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
