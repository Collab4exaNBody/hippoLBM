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
#include <hippoLBM/grid/parallel_for_core.cu>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/set_distribution.hpp>
#include <hippoLBM/grid/update_ghost.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class SetDistributionsLBM : public OperatorNode
  {
    public:
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
      ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT);
      ADD_SLOT( LBMGridRegion, grid_region, INPUT, REQUIRED);
      ADD_SLOT( AABB, bounds, INPUT, OPTIONAL, DocString{"Domain's bounds"});
      ADD_SLOT( double, value, INPUT, double(1) );
      ADD_SLOT( bool, do_update, INPUT, false);

      inline void execute () override final
      {
        auto& data = *fields;
        auto& traversals = *grid_region;
        LBMDomain<Q>& Domain = *domain;
        LBMGrid& Grid = Domain.m_grid;
        GridIJKtoIdx ijk_to_idx(Grid);

        FieldView pf = data.distributions();
        const double * const pw = data.weights();

        // define kernel
        init_distributions<Q> func = {*value, ijk_to_idx};

        // capture the parallel execution context
        auto par_exec_ctx = [this] (const char* exec_name)
        { 
          return this->parallel_execution_context(exec_name);
        };

        if(bounds.has_value())
        {

          auto& bound = *bounds;
          Vec3d min = bound.bmin;
          Vec3d max = bound.bmax;
          double Dx = Grid.dx;
          Point3D _min = {int(min.x/Dx), int(min.y/Dx), int(min.z/Dx)};
          Point3D _max = {int(max.x/Dx), int(max.y/Dx), int(max.z/Dx)};

          Box3D global_wall_box = {_min, _max};

          auto [is_inside_subdomain, wall_box] = Grid.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_wall_box);
          if( !is_inside_subdomain ) return;

          parallel_for(wall_box, func, parallel_execution_context(), pf, pw);

        }
        else  // all domain
        { 
          if( *do_update )
          {
            auto [ptr, size] = traversals.get_data<Traversal::Real>();
            parallel_for_id(ptr, size, func, parallel_execution_context(), pf, pw);
            update_ghost(Domain, pf, par_exec_ctx);
          }
          else
          {
            auto [ptr, size] = traversals.get_data<Traversal::All>();
            parallel_for_id(ptr, size, func, parallel_execution_context(), pf, pw);
          }
        }
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_distributions)
  {
    OperatorNodeFactory::instance()->register_factory( "set_distribution", make_variant_operator<SetDistributionsLBM>);
  }
}
