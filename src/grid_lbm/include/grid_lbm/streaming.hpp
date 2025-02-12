#pragma once

namespace hipoLBM
{
	template<int Q>
		struct streaming_step1
		{
			/**
			 * @brief Operator for performing the first step of streaming at a given index.
			 *
			 * @param idx The index.
			 * @param f Pointer to an array of doubles representing distribution functions.
			 */
			inline void operator()(int idx, double * const f) const
			{
				const int idxQ = idx * Q;
				for (int iLB = 1; iLB < Q; iLB += 2)
				{
					std::swap(f[idxQ+iLB], f[idxQ+iLB+1]);
				}
			}
		};

	/**
	 * @brief A functor for the second step of streaming in the lattice Boltzmann method for XYZ directions.
	 *
	 * @tparam perX Whether periodic boundary conditions are applied in the X-direction.
	 * @tparam perY Whether periodic boundary conditions are applied in the Y-direction.
	 * @tparam perZ Whether periodic boundary conditions are applied in the Z-direction.
	 */
	template<int Q>
		struct streaming_step2
		{
			grid<3>& g;
			/**
			 * @brief Operator for performing the second step of streaming at given coordinates (x, y, z).
			 *
			 * @param x The x-coordinate.
			 * @param y The y-coordinate.
			 * @param z The z-coordinate.
			 * @param f Pointer to an array of doubles representing distribution functions.
			 * @param ex Pointer to an array of integers for X-direction.
			 * @param ey Pointer to an array of integers for Y-direction.
			 * @param ez Pointer to an array of integers for Z-direction.
			 */
			inline void operator()(int x, int y, int z,
					double * const f, const int* ex, const int* ey, const int* ez) const
			{
				const int idx = g(x,y,z) * Q;
				for (int iLB = 1; iLB < Q; iLB += 2)
				{
					const int next_x = x + ex[iLB];
					const int next_y = y + ey[iLB];
					const int next_z = z + ez[iLB];
					point<3> next = {next_x, next_y, next_z};

					if(g.is_defined(next))
					{
						const int next_idx = g(next) * Q + iLB;
						{
							std::swap(f[idx+iLB+1], f[next_idx]);
						}
					}
				}
			}
		};
}
