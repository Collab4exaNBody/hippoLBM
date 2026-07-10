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

/** @brief
 */
struct PlotLine {
  std::vector<onika::math::Vec3d> velocity_;
  inline size_t size() const { return velocity_.size(); }
  inline double ux(int i) const { return velocity_[i].x; }
  inline double uy(int i) const { return velocity_[i].y; }
  inline double uz(int i) const { return velocity_[i].z; }
  inline bool resize(Point3D start, Point3D end) {
    // check that the start and end points are aligned along a single dimension
    int count = 0;
    int size = 0;
    for (int d = 0; d < 3; d++) {
      if (start[d] != end[d]) {
        count++;
        int diff = end[d] - start[d];
        if (diff <= 0) {
          throw std::runtime_error("PlotLine::resize: end point must be greater than start point");
        }
        size = diff + 1;  // Box3D bounds are inclusive, so the line spans diff+1 points
      }
    }
    if (count != 1) {
      throw std::runtime_error("PlotLine::resize: start and end points must be aligned along a single dimension");
    }
    velocity_.resize(size);
    std::fill(velocity_.begin(), velocity_.end(), onika::math::Vec3d{0.0, 0.0, 0.0});
    return true;
  }
};

struct ExtractVelocityFunctor {
  Box3D line_;            // The local grid box
  Box3D bx_;              // The local grid box, used to compute the linear index of a point.
  double ratio_dx_dtLB_;  // Conversion factor from LBM units to physical units (dx / dtLB).
  FieldView<3> m1_;       // View of the velocity field (first-order moments).
  PlotLine& points;       // Reference to the PlotLine object where the extracted velocities will be stored.

  inline void operator()(const onikaInt3_t&& coord) const {
    int line_idx = line_.get(coord.x, coord.y, coord.z);
    int grid_idx = bx_(coord.x, coord.y, coord.z);
    points.velocity_[line_idx] = ratio_dx_dtLB_ * m1_.get(grid_idx);
  }
};

using namespace onika::scg;

template <int Q>
class PlotLineVelocityOp : public OperatorNode {
 public:
  // LBM data structures
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain containing the simulation data."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});

  // file
  ADD_SLOT(std::string, dump_file, INPUT, "line_%010d.csv",
           DocString{"The name of the CSV file, formatted with the current timestep."});
  ADD_SLOT(std::string, output_directory, INPUT, "hippoLBMOutputDir",
           DocString{"The base directory for the profile output."});
  ADD_SLOT(long, timestep, INPUT, 0, DocString{"The current timestep."});

  // real input
  ADD_SLOT(onika::math::AABB, line, INPUT, OPTIONAL,
           DocString{"The bounding box defining the line (Real World) along which to compute the velocity profile."});
  ADD_SLOT(Box3D, line_lbm, INPUT, OPTIONAL,
           DocString{"The bounding box defining the line (LBM World) along which to compute the velocity profile."});

  // option
  ADD_SLOT(bool, display_filename, INPUT, true, DocString{""});

  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
    Extracts the velocity profile along an axis-aligned line and writes it to a CSV file (rx, ry, rz, ux, uy, uz).
    The line is specified either in physical coordinates (`line`) or in LBM grid indices (`line_lbm`).

    YAML example:

    plot_line_velocity:
      line:
        min: [[ 0.0, 0.05, 0.0 ],[ 0.0, 0.05, 1.0 ]]

    plot_line_velocity:
      line_lbm: [[0, 5, 0 ], [ 0, 5, 30 ]]

    global:
      simulation_analysis_freq: 100  

    )EOF";
  }

  void execute() final {
    PlotLine points;
    LBMGrid& grid = domain->grid();
    Box3D global_line;

    if (line_lbm.has_value() && line.has_value()) {
      lout << "You can't define both slots: line and line_lbm" << std::endl;
      return;
    } else if (line_lbm.has_value()) {
      global_line = *line_lbm;  //
    } else if (line.has_value()) {
      auto [inf, sup] = *line;
      global_line = Box3D{grid.project_to_grid<Area::Global>(inf), grid.project_to_grid<Area::Global>(sup)};
    } else {
      lout << "Error [plot_line_velocity], you need to specify slot line or line_lbm" << std::endl;
      return;
    }

    points.resize(global_line.lower(), global_line.upper());

    auto [is_inside_subdomain, local_line] = grid.restrict_box_to_grid<Area::Local, Traversal::Real>(global_line);

    if (is_inside_subdomain) {
      // Same shape as global_line, translated to local coordinates (not clipped to this
      // rank's subdomain), so line_.get() yields indices relative to the *global* line
      // start, matching the layout of points.velocity_ across all ranks.
      Box3D local_full_line = grid.convert<Area::Local>(global_line);
      onika::parallel::ParallelExecutionSpace<3> parallel_range = set(local_line);
      ExtractVelocityFunctor func = {local_full_line, grid.bx_, domain->dx() / Params->dtLB_, fields->flux(), points};
      parallel_for(parallel_range, func, parallel_execution_context("plot_line"));
    }  // else do nothing, the line is outside the local subdomain

    int rank;
    MPI_Comm_rank(*mpi, &rank);
    int master = 0;
    if (rank == master) {
      MPI_Reduce(MPI_IN_PLACE, points.velocity_.data(), 3 * points.size(), MPI_DOUBLE, MPI_SUM, master, *mpi);
    } else {
      MPI_Reduce(points.velocity_.data(), nullptr, 3 * points.size(), MPI_DOUBLE, MPI_SUM, master, *mpi);
    }

    std::string OutPutDirectory = *output_directory + "/Profile/";

    // dump data, only on the master rank since the reduction above is only valid there
    if (rank == master) {
      std::filesystem::create_directories(OutPutDirectory);
      std::string file_name = OutPutDirectory + (*dump_file);
      file_name = onika::format_string(file_name, *timestep);

      if (*display_filename) {
        lout << "writing profile file: " << file_name << std::endl;
      }

      constexpr int significant_digits = 8;
      constexpr int column_width = significant_digits + 8;  // sign + digits + decimal point + exponent

      std::ostringstream buffer;
      buffer << std::setw(column_width) << "rx" << std::setw(column_width) << "ry" << std::setw(column_width) << "rz"
             << std::setw(column_width) << "ux" << std::setw(column_width) << "uy" << std::setw(column_width) << "uz"
             << '\n';
      buffer << std::setprecision(significant_digits);

      for (int i = 0; i < global_line.number_of_points(); i++) {
        onika::math::Vec3d r = grid.compute_position<Area::AsIs>(global_line.get(i));
        buffer << std::setw(column_width) << r.x << std::setw(column_width) << r.y << std::setw(column_width) << r.z
               << std::setw(column_width) << points.ux(i) << std::setw(column_width) << points.uy(i)
               << std::setw(column_width) << points.uz(i) << '\n';
      }

      std::ofstream out(file_name);
      out << buffer.str();
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(plot_line_velocity) {
  OperatorNodeFactory::instance()->register_factory("plot_line_velocity", make_variant_operator<PlotLineVelocityOp>);
}
}  // namespace hippoLBM
