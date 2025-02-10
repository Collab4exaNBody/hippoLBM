#pragma once

#include <grid_lbm/box.hpp>
#include <grid_lbm/ghost_manager.hpp>
#include <grid_lbm/grid.hpp>

namespace hipoLBM
{
	constexpr int DIM = 3;

	template<int Q>
		struct domain_lbm
		{
			ghost_manager<Q,DIM> m_ghost_manager;
			box<DIM> m_box;
			grid<DIM> m_grid;
      onika::math::AABB bounds;
      int3d domain_size;
      onika::math::IJK MPI_coord;
      onika::math::IJK MPI_grid_size;
		};
};
