#pragma once

#include <onika/math/basic_types_def.h>
#include <grid_lbm/grid.hpp>

namespace hipoLBM
{
  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
  using ::onika::math::IJK;
  using namespace onika::cuda;

  struct traversal_data
  {
    const int * const ptr;
    const size_t size;
  };

	struct traversal_lbm
	{
		vector_t<int> ghost_edge;
		vector_t<int> inside;
		vector_t<int> real;
		vector_t<int> all;
		vector_t<int> edge;
		vector_t<int> extend;
		vector_t<int> plan_xy_0, plan_xy_l;

    template<Traversal Tr> traversal_data get_data();
 

		void build_traversal(grid<DIM>& G, const IJK MPI_coord, const IJK MPI_grid)
		{
			constexpr Area L = Area::Local;
			constexpr Traversal A = Traversal::All;
			constexpr Traversal R = Traversal::Real;
			constexpr Traversal I = Traversal::Inside;
			constexpr Traversal E = Traversal::Extend;
			auto ba = G.build_box<L, A>();
			auto br = G.build_box<L, R>();
			auto bi = G.build_box<L, I>();
			auto ex = G.build_box<L, E>();
			all.resize(ba.number_of_points());
			real.resize(br.number_of_points());
			inside.resize(bi.number_of_points());
			ghost_edge.resize(all.size() - inside.size());
			extend.resize(ex.number_of_points());

			size_t shift_a(0), shift_r(0), shift_i(0), shift_ge(0), shift_ex(0);
			for (int z = ba.start(2); z <= ba.end(2); z++) {
				for (int y = ba.start(1); y <= ba.end(1); y++) {
					for (int x = ba.start(0); x <= ba.end(0); x++) {
						point<3> p = {x, y, z};
						int idx = G(x, y, z);
						all[shift_a++] = idx;
						if (br.contains(p)) {
							real[shift_r++] = idx;
							if (bi.contains(p)) {
								inside[shift_i++] = idx;
							}
						}

						if (!bi.contains(p)) {
							ghost_edge[shift_ge++] = idx;
						}
						if (ex.contains(p))
							extend[shift_ex++] = idx;
					}
				}
			}

			assert(shift_ex == extend.size());
			assert(shift_i == inside.size());
			assert(shift_r == real.size());
			assert(shift_a == all.size());
			assert(shift_ge == ghost_edge.size());

			// used by neumann z functors
			int plan_size = ba.get_length(0) * ba.get_length(1);
			int idx_xy0(0), idx_xyl(0);

			int plan_0z = br.start(2);
			int plan_lz = br.end(2);

      onika::lout << "PLANE LZ "<< plan_lz << std::endl;
			if (MPI_coord.k == 0)
				plan_xy_0.resize(plan_size);
			if (MPI_coord.k == MPI_grid.k - 1)
				plan_xy_l.resize(plan_size);

			for (int y = ba.start(1); y <= ba.end(1); y++) {
				for (int x = ba.start(0); x <= ba.end(0); x++) {
					if (MPI_coord.k == 0)
						plan_xy_0[idx_xy0++] = G(x, y, plan_0z);
					if (MPI_coord.k == MPI_grid.k - 1)
						plan_xy_l[idx_xyl++] = G(x, y, plan_lz);
				}
			}
			if (MPI_coord.k == 0)
				assert(idx_xy0 == plan_size);
			if (MPI_coord.k == MPI_grid.k - 1)
				assert(idx_xyl == plan_size);
		}
	};


	template<> inline traversal_data traversal_lbm::get_data<Traversal::All>()    { return{ vector_data(all), vector_size(all)}; }  
	template<> inline traversal_data traversal_lbm::get_data<Traversal::Real>()   { return{ vector_data(real), vector_size(real)}; }  
	template<> inline traversal_data traversal_lbm::get_data<Traversal::Extend>() { return{ vector_data(extend), vector_size(extend)}; }  
	template<> inline traversal_data traversal_lbm::get_data<Traversal::Inside>() { return{ vector_data(inside), vector_size(inside)}; }  
	template<> inline traversal_data traversal_lbm::get_data<Traversal::Edge>() { return{ vector_data(edge), vector_size(edge)}; }  
	template<> inline traversal_data traversal_lbm::get_data<Traversal::Ghost_Edge>() { return{ vector_data(ghost_edge), vector_size(ghost_edge)}; }  
	template<> inline traversal_data traversal_lbm::get_data<Traversal::Plan_xy_0>() { return{ vector_data(plan_xy_0), vector_size(plan_xy_0)}; }  
	template<> inline traversal_data traversal_lbm::get_data<Traversal::Plan_xy_l>() { return{ vector_data(plan_xy_l), vector_size(plan_xy_l)}; }  

};
