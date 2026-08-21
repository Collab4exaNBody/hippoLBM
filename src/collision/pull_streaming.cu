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
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/compute/reduce.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/update_ghost.hpp>

// Implementation
#include <hippoLBM/collision/pull_streaming.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

template <int Q>
class PullStreamingLBM : public OperatorNode {
 public:
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED);
  ADD_SLOT(LBMScratchStreamingBuffer, scratch, PRIVATE,
           DocString{"Scratch buffer receiving the post-streaming distributions computed."});

  inline std::string documentation() const final {
    return R"EOF(

    YAML example:

      - pull_streaming
    )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;
    LBMGrid& Grid = domain->grid();
    auto [ptr, size] = traversals.get_levels();

    // get fields
    FieldView<Q> pf = data.distributions();

    // capture the parallel execution context
    auto par_exec_ctx = [this](const char* exec_name) { return this->parallel_execution_context(exec_name); };

    // ghost cells must hold this timestep's post-collision data before we can gather from them
    update_ghost(*domain, pf, par_exec_ctx);

    // gather-based streaming: read from the live (untouched) field, write into the scratch buffer
    auto& buffer = *scratch;
    buffer.resize(data.size(), Q);
    FieldView<Q> tmp = buffer.view<Q>(data.size());
    pull_streaming_step<Q, Traversal::Real> step = {ptr, Grid, pf, tmp};
    parallel_for_simple(size, step, parallel_execution_context("pull_streaming_step"));
    std::swap(data.f_, buffer.f_);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(pull_streaming) {
  OperatorNodeFactory::instance()->register_factory("pull_streaming", make_variant_operator<PullStreamingLBM>);
}
}  // namespace hippoLBM
