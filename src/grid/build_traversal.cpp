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
#include <grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/traversal_lbm.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class BuildTraversalLBM : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_domain<Q>, LBMDomain, INPUT);
      ADD_SLOT( traversal_lbm, Traversals, OUTPUT);
      inline void execute () override final
      {
        auto& domain = *LBMDomain;
        traversal_lbm traversal;
        traversal.build_traversal(domain.m_grid, domain.MPI_coord, domain.MPI_grid_size);
        *Traversals = traversal;
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(build_traversal)
  {
    OperatorNodeFactory::instance()->register_factory( "build_traversal", make_variant_operator<BuildTraversalLBM>);
  }
}

