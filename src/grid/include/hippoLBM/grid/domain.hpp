#pragma once

#include <hippoLBM/grid/box3d.hpp>
#include <hippoLBM/grid/ghost_manager.hpp>
#include <hippoLBM/grid/grid.hpp>

namespace hippoLBM
{
  template<int Q>
    struct LBMDomain
    {
      LBMGhostManager<Q> m_ghost_manager;
      Box3D m_box;
      LBMGrid m_grid;
      onika::math::AABB bounds;
      int3d domain_size;
      onika::math::IJK MPI_coord;
      onika::math::IJK MPI_grid_size;
      LBMDomain() {};
      LBMDomain(LBMGhostManager<Q>& g, Box3D& b, LBMGrid& gr, onika::math::AABB& bd, int3d& ds, onika::math::IJK& mc, onika::math::IJK& mgs)
	: m_ghost_manager(g), m_box(b), m_grid(gr), bounds(bd), domain_size(ds), MPI_coord(mc), MPI_grid_size(mgs) {} 

      double dx() { return m_grid.dx; }
      LBMGrid& grid() { return m_grid; }
    };
};
