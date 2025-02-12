#pragma once

#include <grid_lbm/box.hpp>

namespace hipoLBM
{
	template<int N, int DIM>
		struct packer
		{
			/**
			 * @brief Operator for copying data from a source box to a destination box.
			 *
			 * This operator copies data efficiently from a source box to a destination box.
			 *
			 * @param dst     A pointer to the destination data array.
			 * @param src     A pointer to the source data array.
			 * @param dst_box The destination box where data will be copied.
			 * @param mesh_box The source box from which data will be copied.
			 */
			inline void operator()(double* dst, double* src, const box<DIM>& dst_box, const box<DIM>& mesh_box) const
			{
				static_assert(DIM == 3);
				for(int dim = 0 ; dim < DIM ; dim++) 
				{
					assert( dst_box.get_length(dim) <= mesh_box.get_length(dim) );
				}

				const auto inf = dst_box.inf;
				const auto sup = dst_box.sup;
				const int nb_bytes = N * sizeof(double);

#pragma omp parallel for collapse(3)
				for(int z = inf[2] ; z <= sup[2] ; z++)
					for(int y = inf[1] ; y <= sup[1] ; y++)
						for(int x = inf[0] ; x <= sup[0] ; x++)
						{
							const int dst_idx = compute_idx<DIM>(dst_box, x-inf[0], y-inf[1], z-inf[2]) * N;
							const int src_idx = compute_idx<DIM>(mesh_box, x, y, z) * N;
							// void* memcpy( void* dest, const void* src, std::size_t count );
							std::memcpy(&dst[dst_idx], &src[src_idx], nb_bytes);
						}
			}
		};

	template<int N, int DIM>
		struct unpacker
		{
			/**
			 * @brief Operator for unpacking data from a source box to a destination box.
			 *
			 * This method is responsible for unpacking (copying) data from a source box to a destination
			 * box. It is designed to efficiently handle this data transfer. 
			 * The operator is templated on the number of
			 * elements per point (`N`) and the dimensionality of the space (`DIM`).
			 *
			 * @param dst      A pointer to the destination data array.
			 * @param src      A pointer to the source data array.
			 * @param src_box  The source box from which data will be unpacked.
			 * @param mesh_box The destination box where data will be copied.
			 *
			 * @note This operator assumes that the dimensionality is 3 (DIM == 3).
			 */
			inline void operator()(double* dst, double* src, const box<DIM>& src_box, const box<DIM>& mesh_box) const
			{
				static_assert(DIM == 3);
				for(int dim = 0 ; dim < DIM ; dim++) 
				{
					assert( src_box.get_length(dim) <= mesh_box.get_length(dim) );
				}

				const auto inf = src_box.inf;
				const auto sup = src_box.sup;
				const int nb_bytes = N * sizeof(double);

#pragma omp parallel for collapse(3)
				for(int z = inf[2] ; z <= sup[2] ; z++)
					for(int y = inf[1] ; y <= sup[1] ; y++)
						for(int x = inf[0] ; x <= sup[0] ; x++)
						{
							const int dst_idx = compute_idx(mesh_box, x , y , z) * N;
							const int src_idx = compute_idx(src_box, x - inf[0], y - inf[1], z - inf[2]) * N;
							std::memcpy(&dst[dst_idx], &src[src_idx], nb_bytes);
							std::memcpy(&dst[dst_idx], &src[src_idx], nb_bytes);
						}
			}
		};
}
