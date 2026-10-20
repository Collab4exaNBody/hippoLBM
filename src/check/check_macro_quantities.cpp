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
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/io/simulation_state.hpp>

namespace hippoLBM {

using namespace onika::scg;

template <int Q>
class CheckerMacroQuantitiesOp : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(SimulationStatistics, simulation_statistics, INPUT, REQUIRED,
           DocString{"Contains macro quatities computes by the simulation (log)"});
  ADD_SLOT(double, density, INPUT, OPTIONAL, DocString{"Sum of all densities."});
  ADD_SLOT(double, tol, INPUT, OPTIONAL, DocString{"Tolerance relative to the sum of all densities."});

  ADD_SLOT(double, vmin, INPUT, 0, DocString{"Minimal velocity (norm). Always >= 0"});
  ADD_SLOT(double, vmax, INPUT, OPTIONAL, DocString{"Max velocity (norm)"});

  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
    This operator checks some macro quantities for CI.

    YAML configuration example:
    
      - check_macro_quantities:
         density: 1e6
         tol: 1e-6
         vmax: 0.1
         vmin: 0


        )EOF";
  }

  inline void execute() final {
    SimulationStatistics stats = *simulation_statistics;

    int rank;
    MPI_Comm_rank(*mpi, &rank);

    // simulation_statistics is MPI_Reduce'd onto rank 0 only (see SimulationState::execute()),
    // so it is only meaningful there: the verdict can only be computed on rank 0.
    bool error_spotted = false;
    std::string msg = "Regression test checker [macro quantities]\n";

    if (rank == 0) {
      if (stats.min_velocity_norm_ < *vmin - 1e-15) {
        msg += "min vel (simulation): " + std::to_string(stats.min_velocity_norm_) +
               " is below (user-defined check): " + std::to_string(*vmin) + "\n";
        error_spotted = true;
      }

      if (vmax.has_value()) {
        if (stats.max_velocity_norm_ > *vmax + 1e-15) {
          msg += "max vel (simulation): " + std::to_string(stats.max_velocity_norm_) +
                 " is upper (user-defined check): " + std::to_string(*vmax) + "\n";
          error_spotted = true;
        }
      }

      if (density.has_value()) {
        if (!tol.has_value()) {
          msg += "Please add a tolerance if you use the density INPUT slot.\n";
          error_spotted = true;
        } else if (std::abs(stats.sum_density_ - (*density)) > *tol) {
          msg += "Sum of densities (simulation): " + std::to_string(stats.sum_density_) +
                 " is not close to (user-defined check): " + std::to_string(*density) + " +/- " +
                 std::to_string(*tol) + "\n";
          error_spotted = true;
        }
      }

      if (error_spotted) {
        lout << msg << std::endl;
      }
    }

    // Broadcast the rank-0 verdict and have every rank abort together: std::exit() on rank 0
    // alone would leave the other ranks running and liable to hang on the next MPI collective.
    int error_flag = error_spotted ? 1 : 0;
    MPI_Bcast(&error_flag, 1, MPI_INT, 0, *mpi);

    if (error_flag) {
      MPI_Abort(*mpi, EXIT_FAILURE);
    }

    if (rank == 0) {
      lout << "Test Pass." << std::endl;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(check_macro_quantities) {
  OperatorNodeFactory::instance()->register_factory("check_macro_quantities",
                                                    make_variant_operator<CheckerMacroQuantitiesOp>);
}
}  // namespace hippoLBM
