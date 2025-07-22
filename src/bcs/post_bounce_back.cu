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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <grid/make_variant_operator.hpp>
#include <onika/math/basic_types.h>
#include <grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/traversal_lbm.hpp>
#include <hippoLBM/bcs/bounce_back.hpp>
#include <hippoLBM/bcs/bounce_back_manager.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class PostBounceBack : public OperatorNode
  {
    ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
    ADD_SLOT( bounce_back_manager<Q>, bbmanager, INPUT_OUTPUT);
    public:
    inline std::string documentation() const override final
    {
      return R"EOF( 
        )EOF";
    }

    template<int dim, Side dir> 
      void launcher(traversal_lbm& traversals, FieldView<Q>& pf, bounce_back_manager<Q>& bbm)
      {
        constexpr int idx = helper_dim_idx<dim,dir>();
        FieldView<bounce_back_manager<Q>::Un> pfi = bbm.get_data(idx);
        if( pfi.num_elements> 0 )
        {
          constexpr Traversal Tr = get_traversal<dim, dir>();
          auto [ptr, size] = traversals.get_data<Tr>();

          assert(ptr != nullptr);
          assert(pfi.num_elements == int(size));

          ParallelForOptions opts;
          opts.omp_scheduling = OMP_SCHED_STATIC;
          post_bounce_back<dim, dir, Q> kernel = {ptr};
          auto params = make_tuple(pf, pfi);
          parallel_for_id_runner runner = {kernel, params}; //pf, pfi};
          parallel_for(size, runner, parallel_execution_context(), opts);
      }
  }

  inline void execute () override final
  {
    auto& data = *LBMFieds;
    auto& traversals = *Traversals;

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
ONIKA_AUTORUN_INIT()
{
  OperatorNodeFactory::instance()->register_factory( "post_bounce_back", make_variant_operator<PostBounceBack>);
}
}

