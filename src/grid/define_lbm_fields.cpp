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

#include <onika/math/basic_types.h>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class DefineLBMFields : public OperatorNode
  {
    public:
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
      ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT);

      inline void execute () override final
      {
        constexpr Area L = Area::Local;
        constexpr Traversal Tr = Traversal::All;
        LBMFields<Q>& grid_data = *fields;
        LBMGrid& Grid = domain->m_grid;
        Box3D& Box = domain->m_box;

        // compute sizes
        constexpr int Un = 5;
        auto bx = Grid.build_box<L, Tr>();
        int size_XYU = bx.get_length(0) * bx.get_length(1) * Un;
        int size_YZU = bx.get_length(1) * bx.get_length(2) * Un;
        int size_XZU = bx.get_length(0) * bx.get_length(2) * Un;
        const size_t np = Box.number_of_points();

        grid_data.grid_size = np;
        if(grid_data.obst.size() != np)
        {
          grid_data.f.resize(np*Q, 0);
          grid_data.obst.resize(np);
          grid_data.m0.resize(np, 0);
          grid_data.m1.resize(np*3, 0);
          grid_data.fi_x_0.resize(size_YZU);
          grid_data.fi_x_l.resize(size_YZU);
          grid_data.fi_y_0.resize(size_XZU);
          grid_data.fi_y_l.resize(size_XZU);
          grid_data.fi_z_0.resize(size_XYU);
          grid_data.fi_z_l.resize(size_XYU);
        }
      }
  };



  // === register factories ===  
  ONIKA_AUTORUN_INIT(define_grid)
  {
    OperatorNodeFactory::instance()->register_factory( "define_grid_3dq19", make_simple_operator<DefineLBMFields<19>>);
    //OperatorNodeFactory::instance()->register_factory( "define_grid_3dq15", make_compatible_operator<DefineLBMFields<Q><15>>);
  }
}

