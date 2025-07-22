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

#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <grid/make_variant_operator.hpp>
#include <grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/init_obst.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class InitObstLBM : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_domain<Q>, LBMDomain, INPUT, REQUIRED);
      ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT);

      inline void execute () override final
      {
        auto& data = *LBMFieds;
        auto& domain = *LBMDomain;
        init_obst func = {onika::cuda::vector_data(data.obst)};
        constexpr Area A = Area::Local;
        constexpr Traversal Tr = Traversal::All;
        parallel_for_id<A,Tr>(domain.m_grid, func, parallel_execution_context());       
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_obstacles)
  {
    OperatorNodeFactory::instance()->register_factory( "init_obst", make_variant_operator<InitObstLBM>);
  }
}

