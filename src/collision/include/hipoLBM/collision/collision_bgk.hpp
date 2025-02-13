#pragma once

#define FLUIDE_ -1

namespace hipoLBM
{
	using namespace onika::math;
	/**
	 * @brief A functor for collision operations in the lattice Boltzmann method.
	 */
	template<int Q>
		struct collision_bgk
		{
			const Vec3d m_Fext;
			/**
			 * @brief Operator for performing collision operations at a given index.
			 *
			 * @param idx The index.
			 * @param m1 Pointer to a vec3d.
			 * @param obst Pointer to an array of ints.
			 * @param f Pointer to an array of doubles.
			 * @param m0 Pointer to an array of doubles.
			 * @param ex Pointer to an array of ints.
			 * @param ey Pointer to an array of ints.
			 * @param ez Pointer to an array of ints.
			 * @param w Pointer to an array of doubles.
			 * @param tau The relaxation time.
			 */
			ONIKA_HOST_DEVICE_FUNC inline void operator()(
					int idx, 
					Vec3d * m1,
					int * const obst, 
					double * const f,
					double * const m0, 
					const int* ex, 
					const int* ey,
					const int* ez, 
					const double* const w, 
					double tau) const
			{
				const int idxQ = idx * Q;

				if (obst[idx] == FLUIDE_) {

					const double rho = m0[idx];
					const double ux = m1[idx].x;
					const double uy = m1[idx].y;
					const double uz = m1[idx].z;
					const double u_squ = ux * ux + uy * uy + uz * uz;

					for (int iLB = 0; iLB < Q; iLB++) {
						double eu, feq, ef;
						ef = ex[iLB] * m_Fext.x + ey[iLB] * m_Fext.y + ez[iLB] * m_Fext.z;
						eu = ex[iLB] * ux + ey[iLB] * uy + ez[iLB] * uz;
						feq = w[iLB] * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
						f[idxQ+iLB] += 1. / tau * (feq - f[idxQ+iLB]) + 3. * rho * w[iLB] * ef;
					}
				}
			}
		};
}

