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
#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/io/simulation_state.hpp>

// reduce
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/compute/reduce.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;
using namespace onika::memory;

template <int Q>
class SimulationState : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(LBMFields<Q>, fields, INPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(SimulationStatistics, simulation_statistics, OUTPUT,
           DocString{"Contains general information about the LBM grid, such as minimum and maximum fluid velocity."});
  ADD_SLOT(CudaMMVector<SimulationStatistics>, scratch, PRIVATE);

  inline std::string documentation() const final {
    return R"EOF(
    This operator computes the simulation state of the LBM grid,
    including statistics such as the sum of density and the minimum and maximum velocity norms.

    YAML configuration example:
    
      - simulation_state

        )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& region = *grid_region;

    auto& buffer = *scratch;
    reset_scratch(buffer, parallel_execution_context("reset_scratch"));
    // buffer[0].display();

    // get fields
    double* const pm0 = data.densities();
    FieldView<3> pm1 = data.flux();

    ComputeSimulationStateFunc func = {pm0, pm1};

    local_reduce(func, buffer, parallel_execution_context("compute_simulation_state"), region);
    local_reduce_sync();
    // reduce on master
    SimulationStatistics local = buffer[0];
    SimulationStatistics global = {};
    int master = 0;
    MPI_Reduce(&local.sum_density_, &global.sum_density_, 1, MPI_DOUBLE, MPI_SUM, master, *mpi);
    MPI_Reduce(&local.min_velocity_norm_, &global.min_velocity_norm_, 1, MPI_DOUBLE, MPI_MIN, master, *mpi);
    MPI_Reduce(&local.max_velocity_norm_, &global.max_velocity_norm_, 1, MPI_DOUBLE, MPI_MAX, master, *mpi);

    *simulation_statistics = global;
  }
};

using SimulationState3D19Q = SimulationState<19>;

// === register factories ===
ONIKA_AUTORUN_INIT(SimulationState) {
  OperatorNodeFactory::instance()->register_factory("simulation_state", make_variant_operator<SimulationState>);
}
}  // namespace hippoLBM
