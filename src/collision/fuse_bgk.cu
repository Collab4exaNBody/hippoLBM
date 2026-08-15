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

// onika
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

// hippoLBM
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/update_ghost.hpp>

// implementation files
#include <hippoLBM/collision/fuse_bgk.hpp>
#include <hippoLBM/collision/streaming.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

/** @brief Fuses macro_variables + bgk collision into a single kernel launch, then runs
 * streaming (step1, ghost update, step2) exactly as the standalone streaming operator does.
 * Combines three OperatorNode executions into one. */
template <int Q>
class FuseBGK : public OperatorNode {
 public:
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString{"Defines the computational domain and its properties for the LBM simulation."});

  inline std::string documentation() const final {
    return R"EOF(
    Fuses macro_variables, bgk, and streaming into a single operator, to avoid the overhead
    of three separate OperatorNode executions per timestep. macro_variables and bgk are fused
    into one kernel (they run over the same points, with no communication in between);
    streaming still runs as its own pass afterwards, since it needs every local point's
    post-collision distributions before it can swap values with neighbors, and involves an
    MPI ghost exchange in between its two steps.

    YAML example:

      - fuse_bgk
    )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;
    auto& params = *Params;
    LBMGrid& Grid = domain->grid();

    // get fields
    FieldView<3> pm1 = data.flux();
    int* const pobst = data.obstacles();
    FieldView<Q> pf = data.distributions();
    double* const pm0 = data.densities();

    // shared by the fused macro_variables+bgk kernel and streaming: both iterate the same
    // per-point traversal levels array.
    auto [ptr, size] = traversals.get_levels();

    // --- fused macro_variables + bgk ---
    fuse_macro_bgk<Q, Traversal::Real> macro_bgk = {ptr, params.Fext_, pm1, pobst, pf, pm0, params.tau_};
    parallel_for_simple(size, macro_bgk, parallel_execution_context("fuse_bgk_macro_bgk"));

    // --- streaming ---
    streaming_step1<Q, Traversal::Real> step1 = {ptr, pf};
    streaming_step2<Q, Traversal::Extend> step2 = {ptr, Grid, pf};
    auto par_exec_ctx = [this](const char* exec_name) { return this->parallel_execution_context(exec_name); };

    parallel_for_simple(size, step1, parallel_execution_context("fuse_bgk_streaming_step1"));
    update_ghost(*domain, pf, par_exec_ctx);
    parallel_for_simple(size, step2, parallel_execution_context("fuse_bgk_streaming_step2"));
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(fuse_bgk) {
  OperatorNodeFactory::instance()->register_factory("fuse_bgk", make_variant_operator<FuseBGK>);
}
}  // namespace hippoLBM
