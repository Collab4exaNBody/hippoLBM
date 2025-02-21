#pragma once

#include <grid_lbm/enum.hpp>

namespace hippoLBM
{
  using namespace onika;
	template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

	template<int dim, Direction dir> 
		inline constexpr int helper_dim_idx() 
		{
			static_assert(dim < DIM);
			if constexpr (dir == Direction::Right) return dim*2 + 1;
			else return dim * 2;
		}

	template<int Q>
		struct bounce_back_manager{};


	template<> struct bounce_back_manager<19>
	{
		static constexpr int  Un = 5; 
		static constexpr int DIM = 3;
		// data[0] : x left
		// data[1] : x right
		// data[2] : y left
		// data[3] : y right
		// data[4] : z bottom
		// data[5] : z top
		std::array<vector_t<double>,6> _data;


		WrapperF<Un> get_data(int i)
		{
			assert( onika::cuda::vector_size(_data[i]) % Un == 0 );
			int size = onika::cuda::vector_size(_data[i]) / Un;
			double * ptr = onika::cuda::vector_data(_data[i]);
			return WrapperF<Un>{ptr, size}; 
		}

		template<int dim>
			int get_size(const onika::math::IJK lgs)
			{
				if constexpr(dim == 0) return lgs.j * lgs.k;
				if constexpr(dim == 1) return lgs.i * lgs.k;
				if constexpr(dim == 2) return lgs.i * lgs.j;
			}

		template<int dim, Direction dir>
			void resize_data(const onika::math::IJK& lgs)
			{
				const size_t size_dim = get_size<dim>(lgs) * Un; 
				int i = helper_dim_idx<dim,dir>();
				auto& data = _data[i];
        if(size_dim != onika::cuda::vector_size(data))
        {
				  data.resize(size_dim); 
        }
			}

		void resize_data(const std::vector<bool>& periodic, const onika::math::IJK& lgs /* local grid size*/, const onika::math::IJK& MPI_coord, const onika::math::IJK& MPI_grid_size)
		{
			if(periodic[0] == false) // not periodic
			{
				constexpr int D = 0; // dimension Z
				if(MPI_coord.i == 0) resize_data<D,Left>(lgs);
				if(MPI_coord.i == MPI_grid_size.i-1) resize_data<D,Right>(lgs);
			}

			if(periodic[1] == false) // not periodic
			{
				constexpr int D = 1; // dimension Z
				if(MPI_coord.j == 0) resize_data<D,Left>(lgs);
				if(MPI_coord.j == MPI_grid_size.j-1) resize_data<D,Right>(lgs);
			}

			if(periodic[2] == false) // not periodic
			{
				constexpr int D = 2; // dimension Z
				if(MPI_coord.k == 0) resize_data<D,Left>(lgs); 
				if(MPI_coord.k == MPI_grid_size.k-1) resize_data<D,Right>(lgs);
			}
		}
	};
}
