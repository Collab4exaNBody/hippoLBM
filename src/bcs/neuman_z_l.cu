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

#include <hippoLBM/grid/make_variant_operator.hpp>
#include <onika/math/basic_types.h>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/bcs/neumann.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class NeumannZL : public OperatorNode
  {
    typedef std::array<double,3> readVec3;
    ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( LBMGridRegion, grid_region, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
    ADD_SLOT( readVec3, U, INPUT, REQUIRED, DocString{"Prescribed velocity at the boundary (z = lz), enforcing the Neumann condition."});

    public:
    inline std::string documentation() const override final
    {
      return R"EOF( This operator enforces a Neumann boundary condition at z = lz in an LBM simulation. 
                      The Neumann boundary condition ensures that the gradient of the distribution function 
                      follows a prescribed value
        )EOF";
    }

    inline void execute () override final
    {
      auto& data = *fields;
      auto& traversals = *grid_region;

      // define functors
      neumann_z_l<Q> neumann = {};

      auto [ux,uy,uz] = *U;

      // get fields
      FieldView<Q> pf = data.distributions();
      int * const pobst = data.obstacles();

      // get traversal
      auto [ptr, size] = traversals.get_data<Traversal::Plan_xy_l>();
      if( size == 0) return;
      // run kernel
      parallel_for_id(ptr, size, neumann, parallel_execution_context(), pobst, pf, ux, uy, uz);
    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(neumann_z_l)
  {
    OperatorNodeFactory::instance()->register_factory( "neumann_z_l", make_variant_operator<NeumannZL>);
  }
}

