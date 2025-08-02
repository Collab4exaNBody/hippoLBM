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
#include <onika/math/basic_types_yaml.h>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/make_domain.hpp>
#include <hippoLBM/grid/comm.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using onika::math::AABB;
  using BoolVector = std::vector<bool>;

  template<int Q>
    class InitDomainLBM : public OperatorNode
  {
    public:
      ADD_SLOT( MPI_Comm, mpi, INPUT , MPI_COMM_WORLD);
      ADD_SLOT( LBMDomain<Q>, domain, OUTPUT);
      ADD_SLOT( BoolVector, periodic   , INPUT_OUTPUT , REQUIRED );
      ADD_SLOT( double, resolution, INPUT_OUTPUT, REQUIRED, DocString{"Resolution"});
      ADD_SLOT( AABB, bounds, INPUT_OUTPUT, REQUIRED, DocString{"Domain's bounds"});

      inline void execute () override final
      {
        *domain = make_domain<Q>(*bounds, *resolution, *periodic, *mpi);
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_domain)
  {
    OperatorNodeFactory::instance()->register_factory( "domain", make_variant_operator<InitDomainLBM>);
  }
}

