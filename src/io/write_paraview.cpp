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

#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/io/write_paraview.hpp>

#include <onika/string_utils.h>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class WriteParaviewLBM : public OperatorNode
  {
    public:
      ADD_SLOT( MPI_Comm, mpi, INPUT , MPI_COMM_WORLD);
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
      ADD_SLOT( LBMFields<Q>, fields, INPUT, REQUIRED);
      ADD_SLOT( LBMGridRegion, grid_region, INPUT, REQUIRED);
      ADD_SLOT( LBMParameters, Params, INPUT, REQUIRED);
      ADD_SLOT( std::string, filename, INPUT, "hippoLBM_%010d");
      ADD_SLOT( std::string, basedir, INPUT, "hippoLBMOutputDir/ParaviewOutput/");
      ADD_SLOT( long, timestep, INPUT, 0);
      ADD_SLOT( bool, distributions, INPUT, false);
      inline void execute () override final
      {
        ExternalParaviewFields external_paraview_fields;
        if( *distributions ) { 
          FieldView<Q> distributions = fields->distributions();
          external_paraview_fields.register_field("Fi", distributions.data, Q, distributions.num_elements);        
        }
        write_paraview(*mpi, *filename, *basedir, *timestep, *fields, *Params, *grid_region, *domain, external_paraview_fields);
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(write_paraview)
  {
    OperatorNodeFactory::instance()->register_factory( "write_paraview", make_variant_operator<WriteParaviewLBM>);
    OperatorNodeFactory::instance()->register_factory( "hippolbm_write_paraview", make_variant_operator<WriteParaviewLBM>);
  }
}

