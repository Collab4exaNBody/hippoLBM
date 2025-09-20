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
      ADD_SLOT( IJK, grid_dims, INPUT, REQUIRED, DocString{"Grid dims"});
			ADD_SLOT( AABB, bounds, INPUT_OUTPUT, REQUIRED, DocString{"Domain's bounds"});

			inline void execute () override final
			{
				GridDetails grid;
        grid.dims = *grid_dims;
				grid.bounds = *bounds;
				grid.periodic = convert<std::array<bool,3>>(*periodic);

				IJK grid_size = grid.dims;
				auto [inf, sup] = grid.bounds;

        Vec3d resolution_dims;
				resolution_dims.x = (sup.x - inf.x) / double(grid_size.i);
				resolution_dims.y = (sup.y - inf.y) / double(grid_size.j);
				resolution_dims.z = (sup.z - inf.z) / double(grid_size.k);

				// check
				bool check_grid_size = false;
        if(resolution_dims.x != resolution_dims.y || resolution_dims.x != resolution_dims.z)
        {
          lout << "[Error, domain], Dx is not the same for all dimension" << std::endl;
          lout << "Dx: [ " << resolution_dims << " ] " << std::endl;
					std::exit(EXIT_FAILURE);  
        }

        double reso = resolution_dims.x;

				if( inf.x + grid_size.i * reso != sup.x) { check_grid_size = true; }
				if( inf.y + grid_size.j * reso != sup.y) { check_grid_size = true; }
				if( inf.z + grid_size.k * reso != sup.z) { check_grid_size = true; }
				if( check_grid_size ) 
				{
					lout << "[Error, domain], The resolution slot and bounds slot mismatch." << std::endl;
          lout << "Bound inf:  " << inf << std::endl; 
          lout << "Bound sup:  " << sup << std::endl;
          lout << "Grid size:  " << grid_size << std::endl; 
          lout << "Resolution: " << reso << std::endl; 
					std::exit(EXIT_FAILURE);  
				}

				SubGridDetails sub_grid = load_balancing(grid, *mpi);
				*domain = make_domain<Q>(grid, sub_grid);
			}
	};

	// === register factories ===  
	ONIKA_AUTORUN_INIT(init_domain)
	{
		OperatorNodeFactory::instance()->register_factory( "domain", make_variant_operator<InitDomainLBM>);
	}
}

