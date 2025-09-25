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
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/parallel_for_core.cu>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/prepro/double_couette.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class InitDoubleCouette : public OperatorNode
  {
    ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
    ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( LBMGridRegion, grid_region, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
    ADD_SLOT( Vec3d, U, INPUT, REQUIRED, DocString{"Prescribed velocity at the boundary (z = 0), enforcing the Neumann condition."});
    ADD_SLOT( LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
    ADD_SLOT( std::string, dimension, INPUT, REQUIRED, DocString{"Choose the dimension."});
    public:
    inline std::string documentation() const override final
    {
      return R"EOF( 
        )EOF";
    }

    inline void execute () override final
    {
      auto& data = *fields;
      auto& params = *Params;
      int3d domain_size = domain->size();
      LBMGrid& grid = domain->grid();

      // define variables
      Vec3d Uc = (*U) / params.celerity;
      
      // get fields 
      FieldView<Q> pf = data.distributions();
      auto [pex, pey, pez] = data.exyz();
      const double * const pw = data.weights();

      // get traversal
      Box3D real = grid.build_box<Area::Local, Traversal::Real>();
      onika::parallel::ParallelExecutionSpace<3> parallel_range = set(real);

			if( *dimension == "X")
			{
        // define variables
        Vec3d dU = Uc / (0.5 * (domain_size[DIMX] - 1));
				// define functors
				InitDoubleCouetteFunc<Q,DIMX> func = { grid, pf, dU, Uc, pex, pey, pez, pw };
				// run kernel
				parallel_for(parallel_range, func, parallel_execution_context("init_double_couette_dim_x"));
			}
      else if(*dimension == "Y")
			{
        // define variables
        Vec3d dU = Uc / (0.5 * (domain_size[DIMY] - 1));
				// define functors
				InitDoubleCouetteFunc<Q,DIMY> func = { grid, pf, dU, Uc, pex, pey, pez, pw };
				// run kernel
				parallel_for(parallel_range, func, parallel_execution_context("init_double_couette_dim_y"));
			}
      else if(*dimension == "Z")
			{
        lout << "Prepro double couette starting ... dim Z" << std::endl;
        // define variables
        Vec3d dU = Uc / (0.5 * (domain_size[DIMZ] - 1));
        lout << "Uc: [" << Uc << "]" << std::endl;
        lout << "dU: [" << dU << "]" << std::endl;
				// define functors
				InitDoubleCouetteFunc<Q,DIMZ> func = { grid, pf, dU, Uc, pex, pey, pez, pw };
				// run kernel
				parallel_for(parallel_range, func, parallel_execution_context("init_double_couette_dim_z"));
        lout << "Prepro double couette ending ... dim Z " << std::endl;
			}
      else
      {
        lout << "[init_double_couette] Please, select a valide dimension \"X\", \"Y\", or \"Z\"." << std::endl; 
        std::exit(EXIT_FAILURE);
      }
		}
	};

	// === register factories ===  
	ONIKA_AUTORUN_INIT(init_double_couette)
	{
		OperatorNodeFactory::instance()->register_factory( "init_double_couette", make_variant_operator<InitDoubleCouette>);
	}
}
