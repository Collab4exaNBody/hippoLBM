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
#include <filesystem>

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>

#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/traversal_lbm.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
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
      ADD_SLOT( LBMDomain<Q>, domain, INPUT);
      ADD_SLOT( LBMFields<Q>, fields, INPUT);
      ADD_SLOT( traversal_lbm, Traversals, INPUT);
      ADD_SLOT( LBMParameters, Params, INPUT);
      ADD_SLOT( MPI_Comm, mpi, INPUT , MPI_COMM_WORLD);
      ADD_SLOT( std::string, filename, INPUT, "hippoLBM_%010d");
      ADD_SLOT( std::string, basedir, INPUT, "hippoLBMOutputDir/ParaviewOutput/");
      ADD_SLOT( long, timestep, INPUT, 0);
      ADD_SLOT( bool, distributions, INPUT, false);
      inline void execute () override final
      {
/*
        auto& comm = *mpi;
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        std::string file_name = *filename;
        file_name = onika::format_string(file_name, *timestep);
        std::string fullname = *basedir + file_name;

        if(rank == 0)
        {
          std::filesystem::create_directories( fullname );
        }
        fullname += "/%06d";
        fullname = onika::format_string(fullname, rank);

        auto& data = *fields;
        auto& traversals = *Traversals;
*/
        ExternalParaviewFields external_paraview_fields;
        if( *distributions ) { 
          FieldView<Q> distributions = fields->distributions();
          external_paraview_fields.register_field("Fi", distributions.data, Q, distributions.num_elements);        
        }
/*
        MPI_Barrier(comm);
        write_pvtr(*basedir, file_name, size, *domain, external_paraview_fields);
        write_vtr( fullname, *domain, data, traversals, *Params, external_paraview_fields);*/
        write_paraview(*mpi, *filename, *basedir, *timestep, *fields, *Params, *Traversals, *domain, external_paraview_fields);
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(write_paraview)
  {
    OperatorNodeFactory::instance()->register_factory( "write_paraview", make_variant_operator<WriteParaviewLBM>);
    OperatorNodeFactory::instance()->register_factory( "hippolbm_write_paraview", make_variant_operator<WriteParaviewLBM>);
  }
}

