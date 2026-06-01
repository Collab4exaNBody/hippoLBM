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
#include <onika/math/basic_types.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/bcs/bounce_back_manager.hpp>
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

// impl
#include <hippoLBM/bcs/bounce_back.hpp>

namespace hippoLBM {
using namespace onika;
using namespace onika::parallel;
using namespace scg;
using namespace onika::cuda;

template <int Q>
class PostBounceBack : public OperatorNode {
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(bounce_back_manager<Q>, bbmanager, INPUT_OUTPUT);

 public:
  inline std::string documentation() const final {
    return R"EOF(
    This operator applies the post-collision bounce-back boundary condition to the distribution functions at the boundary points of the grid.

    YAML example:
    
      - post_bounce_back
    
        )EOF";
  }

  // TODO simplify template usasge in this operator, we don't need to template on the dimension and side, we can just
  // loop over them in the operator and call the appropriate kernel

  /** @brief Launches the post-bounce-back operation for a specific dimension and direction
   * @tparam dim The dimension (0 for x, 1 for y, 2 for z).
   * @tparam dir The side (Left or Right) indicating the direction of the bounce-back
   * @param traversals The grid region containing the traversal indexes for different categories of points.
   * @param pf The field view for the distribution functions.
   * @param bbm The bounce_back_manager containing the data for the bounce-back operation.
   */
  template <int dim, Side dir>
  void launcher(LBMGridRegion& traversals, FieldView<Q>& pf, bounce_back_manager<Q>& bbm) {
    constexpr int idx = helper_dim_idx<dim, dir>();
    FieldView<bounce_back_manager<Q>::Un> pfi = bbm.get_data(idx);
    if (pfi.num_elements > 0) {
      constexpr Traversal Tr = get_traversal<dim, dir>();
      auto [ptr, size] = traversals.get_data<Tr>();

      assert(ptr != nullptr);
      assert(pfi.num_elements == int(size));

      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      post_bounce_back<dim, dir, Q> kernel = {ptr};
      auto params = make_tuple(pf, pfi);
      parallel_for_id_runner runner = {kernel, params};  // pf, pfi};
      parallel_for(size, runner, parallel_execution_context(), opts);
    }
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;

    // storage
    auto& bb = *bbmanager;

    // define functors

    // get fields
    FieldView<Q> pf = data.distributions();

    // for clarity
    constexpr int dim_x = 0;
    constexpr int dim_y = 1;
    constexpr int dim_z = 2;
    launcher<dim_x, Side::Left>(traversals, pf, bb);
    launcher<dim_x, Side::Right>(traversals, pf, bb);
    launcher<dim_y, Side::Left>(traversals, pf, bb);
    launcher<dim_y, Side::Right>(traversals, pf, bb);
    launcher<dim_z, Side::Left>(traversals, pf, bb);
    launcher<dim_z, Side::Right>(traversals, pf, bb);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT() {
  OperatorNodeFactory::instance()->register_factory("post_bounce_back", make_variant_operator<PostBounceBack>);
}
}  // namespace hippoLBM
