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
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/bcs/bounce_back.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class WallBounceBack : public OperatorNode
  {
    public:
      ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);

      inline std::string documentation() const override final
      {
        return R"EOF(  The WallBounceBack class is described as part of the Lattice Boltzmann Method (LBM) implementation, specifically the wall bounce back steps.)EOF";
      }


      inline void execute () override final
      {
        auto& data = *fields;
        LBMGrid& Grid = domain->m_grid;

        // get fields
        const int* const pobst = data.obstacles();
        FieldView<Q> pf = data.distributions();
        auto [pex, pey, pez] = data.exyz();

        // define functors
        wall_bounce_back<Q> func = {Grid, pobst, pf, pex, pey, pez};

        // run kernel
        Box3D extend = Grid.build_box<Area::Local, Traversal::Extend>();
        onika::parallel::ParallelExecutionSpace<3> parallel_range = set(extend);
        parallel_for(parallel_range, func, parallel_execution_context("wall_bounce_back"));
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(wall_bounce_back)
  {
    OperatorNodeFactory::instance()->register_factory( "wall_bounce_back", make_variant_operator<WallBounceBack>);
  }
}

