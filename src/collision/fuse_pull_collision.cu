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
#include <hippoLBM/collision/fuse_collision.hpp>
#include <hippoLBM/collision/pull_streaming.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

template <int Q, typename CollisionModel>
class FusePullCollision : public OperatorNode {
 public:
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString{"Defines the computational domain and its properties for the LBM simulation."});
  ADD_SLOT(LBMScratchStreamingBuffer, scratch, PRIVATE,
           DocString{"Scratch buffer receiving the post-streaming distributions computed by the pull (gather) "
                     "step. Swapped with the live field's buffer at the end of each call, so it holds the "
                     "previous (now stale) distributions afterwards. Allocated once and reused every timestep."});

  inline std::string documentation() const final {
    return R"EOF(

    YAML example:

      - fuse_pull_bgk
      - fuse_pull_mrt
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

    // shared by the fused macro_variables+collision kernel and streaming: both iterate the same
    // per-point traversal levels array.
    auto [ptr, size] = traversals.get_levels();

    // --- fused macro_variables + collision ---
    fuse_macro_collision<Q, Traversal::Real, CollisionModel> macro_collide = {ptr,   params.Fext_, pm1,
                                                                               pobst, pf,           pm0,
                                                                               params.tau_};
    parallel_for_simple(size, macro_collide, parallel_execution_context("fuse_macro_collide"));

    // capture the parallel execution context
    auto par_exec_ctx = [this](const char* exec_name) { return this->parallel_execution_context(exec_name); };

    // ghost cells must hold this timestep's post-collision data before we can gather from them
    update_ghost(*domain, pf, par_exec_ctx);

    // gather-based streaming: read from the live (untouched) field, write into the scratch buffer
    auto& buffer = *scratch;
    buffer.resize(data.size(), Q);
    FieldView<Q> tmp = buffer.view<Q>(data.size());
    pull_streaming_step<Q, Traversal::Real> step = {ptr, Grid, pf, tmp};
    parallel_for_simple(size, step, parallel_execution_context("fuse_pull_collision_streaming"));
    std::swap(data.f_, buffer.f_);
  }
};

template <int Q>
using FusePullBGK = FusePullCollision<Q, BGKCollisionModel>;

template <int Q>
using FusePullMRT = FusePullCollision<Q, MRTCollisionModel>;

// === register factories ===
ONIKA_AUTORUN_INIT(fuse_pull_bgk) {
  OperatorNodeFactory::instance()->register_factory("fuse_pull_bgk", make_variant_operator<FusePullBGK>);
}
ONIKA_AUTORUN_INIT(fuse_pull_mrt) {
  OperatorNodeFactory::instance()->register_factory("fuse_pull_mrt", make_variant_operator<FusePullMRT>);
}
}  // namespace hippoLBM
