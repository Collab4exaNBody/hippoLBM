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

#include <grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/lbm_domain.hpp>
#include <grid/update_ghost.hpp>
#include <grid/make_variant_operator.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class UpdateGhost : public OperatorNode
  {
    ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( lbm_domain<Q>, LBMDomain, INPUT, REQUIRED);

    public:

    inline std::string documentation() const override final
    {
      return R"EOF(  A functor for computing macroscopic variables (densities and flux) for lattice Boltzmann method.
        )EOF";
    }

    inline void execute () override final
    {
      auto& data = *LBMFieds;
      auto& domain = *LBMDomain;

      // capture the parallel execution context
      auto par_exec_ctx = [this] (const char* exec_name)
      { 
        return this->parallel_execution_context(exec_name);
      };

      // get fields
      FieldView<Q> pf = data.distributions();
      update_ghost(domain, pf, par_exec_ctx);
    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(update_ghost)
  {
    OperatorNodeFactory::instance()->register_factory( "update_ghost", make_variant_operator<UpdateGhost>);
  }
}

