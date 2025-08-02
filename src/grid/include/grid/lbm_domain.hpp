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

#pragma once

#include <grid/box.hpp>
#include <grid/ghost_manager.hpp>
#include <grid/grid.hpp>

namespace hippoLBM
{
  constexpr int DIM = 3;

  template<int Q>
    struct lbm_domain
    {
      ghost_manager<Q,DIM> m_ghost_manager;
      box<DIM> m_box;
      grid<DIM> m_grid;
      onika::math::AABB bounds;
      int3d domain_size;
      onika::math::IJK MPI_coord;
      onika::math::IJK MPI_grid_size;
      lbm_domain() {};
      lbm_domain(ghost_manager<Q,DIM>& g, box<DIM>& b, grid<DIM>& gr, onika::math::AABB& bd, int3d& ds, onika::math::IJK& mc, onika::math::IJK& mgs)
        : m_ghost_manager(g), m_box(b), m_grid(gr), bounds(bd), domain_size(ds), MPI_coord(mc), MPI_grid_size(mgs) {} 
    };
};
