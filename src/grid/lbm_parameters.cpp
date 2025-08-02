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
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/domain.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::math;

  template<int Q>
    class LBMParametersOp : public OperatorNode
  {
    public:
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
      ADD_SLOT( Vec3d, Fext, INPUT, Vec3d{0,0,0});
      ADD_SLOT( double, celerity, INPUT, 1);
      ADD_SLOT( double, nuth, INPUT, 1e-4);
      ADD_SLOT( double, avg_rho, INPUT, 1000.0);

      ADD_SLOT( LBMParameters, Params, OUTPUT);
      ADD_SLOT( double , dtLB, OUTPUT);

      inline void execute () override final
      {
        double Dx = domain->dx();
        LBMParameters params;
        params.Fext = *Fext;
        params.celerity = *celerity;

        if( dtLB.has_value() )
        {
          params.dtLB = *dtLB;
          if( params.dtLB > Dx / params.celerity )
          {
            lout << "\033[31m[lbm_parameters, Error] The LBM time step is not set correctly for this LBM mesh size. Please set a time step below: " << Dx / params.celerity << " s" << std::endl;
          } 
        }
        else
        {
          params.dtLB = Dx / params.celerity;
        }

        params.nuth = *nuth;
        params.nu = params.nuth * params.dtLB / (Dx * Dx);
        params.tau = 3. * params.nu + 0.5;
        params.avg_rho = *avg_rho;
        params.print();
        *dtLB = params.dtLB;
        *Params = params;
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(lbm_parameters)
  {
    OperatorNodeFactory::instance()->register_factory( "lbm_parameters", make_variant_operator<LBMParametersOp>);
  }
}

