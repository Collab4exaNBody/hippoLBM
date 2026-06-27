/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */

#include <mpi.h>

#include <fstream>
#include <iomanip>
#include <sstream>

// Onika
#include <onika/log.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/memory/allocator.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

namespace hippoLBM {

/** @brief Per-point profile statistics, stored as separate contiguous buffers (SoA)
 * so they can be MPI-reduced directly without an intermediate host copy.
 */
struct ProfileStatistics {
  onika::memory::CudaMMVector<double> sum_;
  onika::memory::CudaMMVector<int> fluid_points_;
  onika::memory::CudaMMVector<double> min_;
  onika::memory::CudaMMVector<double> max_;

  inline size_t size() const { return sum_.size(); }

  inline void resize(size_t n) {
    sum_.resize(n);
    fluid_points_.resize(n);
    min_.resize(n);
    max_.resize(n);
  }
};

template <int DIM>
struct Profile2D {
  Box3D bx_;
  Point3D offset_;
  double ratio_dx_dtLB_;
  int* const obst_;  // Pointer to the obstacle field in the LBM grid.
  FieldView<3> density_;
  double* const sum_;
  int* const fluid_points_;
  double* const min_;
  double* const max_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const {
    using namespace onika::math;
    const int idx = bx_(coord.x, coord.y, coord.z);
    if (obst_[idx] == FLUIDE_) {
      int out;
      if constexpr (DIM == DIMX) out = coord.x + offset_[0];
      if constexpr (DIM == DIMY) out = coord.y + offset_[1];
      if constexpr (DIM == DIMZ) out = coord.z + offset_[2];

      // ||U|| = velocity
      double v = norm(ratio_dx_dtLB_ * Vec3d{density_(idx, 0), density_(idx, 1), density_(idx, 2)});
      ONIKA_CU_ATOMIC_ADD(sum_[out], v);
      ONIKA_CU_ATOMIC_ADD(fluid_points_[out], 1);
      ONIKA_CU_ATOMIC_MIN(min_[out], v);
      ONIKA_CU_ATOMIC_MAX(max_[out], v);
    }
  }
};

struct ResetStatistics {
  double* const sum_;
  int* const fluid_points_;
  double* const min_;
  double* const max_;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    sum_[idx] = 0.0;
    fluid_points_[idx] = 0;
    min_[idx] = std::numeric_limits<double>::max();
    max_[idx] = std::numeric_limits<double>::lowest();
  }
};

}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int DIM>
struct ParallelForFunctorTraits<hippoLBM::Profile2D<DIM>> {
  static inline constexpr bool RequiresBlockSynchronousCall = true;
  static inline constexpr bool CudaCompatible = true;
};

template <>
struct ParallelForFunctorTraits<hippoLBM::ResetStatistics> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika

namespace hippoLBM {

using namespace onika::scg;

template <int Q>
class PlaneVelocityProfile : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator."});
  ADD_SLOT(std::string, dimension, INPUT, REQUIRED,
           DocString{"The dimension along which the profile is computed: \"X\", \"Y\" or \"Z\"."});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain containing the simulation data."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
  ADD_SLOT(std::string, dump_file, INPUT, "profile_%010d.csv",
           DocString{"The name of the CSV file, formatted with the current timestep."});
  ADD_SLOT(std::string, output_directory, INPUT, "hippoLBMOutputDir",
           DocString{"The base directory for the profile output."});
  ADD_SLOT(long, timestep, INPUT, 0, DocString{"The current timestep."});
  ADD_SLOT(bool, display_filename, true, DocString{"Print filename"});

  // scratch
  ADD_SLOT(ProfileStatistics, scratch, PRIVATE,
           DocString{"Per-point profile statistics (sum, fluid point count, min, max), reused across calls."});

  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
    This operator computes the velocity profile of the LBM simulation along a given dimension (X, Y or Z).
    For each plane perpendicular to that dimension, it averages the velocity norm over every fluid point,
    and also tracks the minimum and maximum velocity norm reached on that plane. These statistics are
    reduced across all MPI ranks, then dumped to a CSV file with the columns "position avg min max".

    YAML example:

      - plane_velocity_profile:
         dimension: "Z"
         dump_file: "profile_%010d.csv"

    )EOF";
  }

  template <int DIM>
  void run() {
    ProfileStatistics& points = *scratch;
    int3d domain_size = domain->domain_size_;  // local domain size
    LBMGrid& grid = domain->grid();

    int size = domain_size[DIM];
    points.resize(size);

    ResetStatistics reset = {points.sum_.data(), points.fluid_points_.data(), points.min_.data(), points.max_.data()};
    onika::parallel::parallel_for(points.size(), reset, parallel_execution_context("reset_statistics"),
                                  onika::parallel::ParallelForOptions{});

    Box3D real = grid.build_box<Area::Local, Traversal::Real>();
    onika::parallel::ParallelExecutionSpace<3> parallel_range = set(real);
    Profile2D<DIM> func = {grid.bx_,          grid.offset_,       domain->dx() / Params->dtLB_, fields->obstacles(),
                           fields->flux(),    points.sum_.data(), points.fluid_points_.data(),  points.min_.data(),
                           points.max_.data()};

    ONIKA_CU_DEVICE_SYNCHRONIZE();
    parallel_for(parallel_range, func, parallel_execution_context("plane_velocity_profile"));

    int rank;
    MPI_Comm_rank(*mpi, &rank);
    int master = 0;
    if (rank == master) {
      MPI_Reduce(MPI_IN_PLACE, points.sum_.data(), size, MPI_DOUBLE, MPI_SUM, master, *mpi);
      MPI_Reduce(MPI_IN_PLACE, points.fluid_points_.data(), size, MPI_INT, MPI_SUM, master, *mpi);
      MPI_Reduce(MPI_IN_PLACE, points.min_.data(), size, MPI_DOUBLE, MPI_MIN, master, *mpi);
      MPI_Reduce(MPI_IN_PLACE, points.max_.data(), size, MPI_DOUBLE, MPI_MAX, master, *mpi);
    } else {
      MPI_Reduce(points.sum_.data(), nullptr, size, MPI_DOUBLE, MPI_SUM, master, *mpi);
      MPI_Reduce(points.fluid_points_.data(), nullptr, size, MPI_INT, MPI_SUM, master, *mpi);
      MPI_Reduce(points.min_.data(), nullptr, size, MPI_DOUBLE, MPI_MIN, master, *mpi);
      MPI_Reduce(points.max_.data(), nullptr, size, MPI_DOUBLE, MPI_MAX, master, *mpi);
    }

    std::string OutPutDirectory = *output_directory + "/Profile/";

    // dump data, only on the master rank since the reduction above is only valid there
    if (rank == master) {
      // mean velocity per profile point, computed from the reduced sum and fluid point count
      std::vector<double> mean(size, 0.0);
      for (int i = 0; i < size; i++) {
        if (points.fluid_points_[i] > 0) {
          mean[i] = points.sum_[i] / points.fluid_points_[i];
        }
      }

      std::filesystem::create_directories(OutPutDirectory);
      std::string file_name = OutPutDirectory + (*dump_file);
      file_name = onika::format_string(file_name, *timestep);

      if (*display_filename) {
        lout << "writing profile file: " << file_name << std::endl;
      }

      const double dx = domain->dx();
      const onika::math::Vec3d& bmin = domain->bounds_.bmin;
      double origin = 0.0;
      if constexpr (DIM == DIMX) origin = bmin.x;
      if constexpr (DIM == DIMY) origin = bmin.y;
      if constexpr (DIM == DIMZ) origin = bmin.z;

      constexpr int significant_digits = 8;
      constexpr int column_width = significant_digits + 8;  // sign + digits + decimal point + exponent

      std::ostringstream buffer;
      buffer << std::setw(column_width) << "position" << std::setw(column_width) << "avg" << std::setw(column_width)
             << "min" << std::setw(column_width) << "max" << '\n';
      buffer << std::setprecision(significant_digits);
      for (int i = 0; i < size; i++) {
        const double position = origin + i * dx;
        buffer << std::setw(column_width) << position << std::setw(column_width) << mean[i] << std::setw(column_width)
               << points.min_[i] << std::setw(column_width) << points.max_[i] << '\n';
      }

      std::ofstream out(file_name);
      out << buffer.str();
    }
  }

  inline void execute() final {
    if (*dimension == "X") {
      run<DIMX>();
    } else if (*dimension == "Y") {
      run<DIMY>();
    } else if (*dimension == "Z") {
      run<DIMZ>();
    } else {
      lout << "[plane_velocity_profile] Please, select a valid dimension \"X\", \"Y\", or \"Z\"." << std::endl;
      lout << "[plane_velocity_profile] is skipped" << std::endl;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(plane_velocity_profile) {
  OperatorNodeFactory::instance()->register_factory("plane_velocity_profile",
                                                      make_variant_operator<PlaneVelocityProfile>);
}
}  // namespace hippoLBM
