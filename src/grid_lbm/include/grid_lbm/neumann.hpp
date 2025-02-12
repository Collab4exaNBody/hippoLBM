#pragma once

#define FLUIDE_ -1

namespace hipoLBM
{
	/**
	 * @brief A functor for handling Neumann boundary conditions at z=lz in the lattice Boltzmann method.
	 */
	template<int Q>
		struct neumann_z_l
		{
			/**
			 * @brief Operator for applying Neumann boundary conditions at z=0.
			 *
			 * @param idxQ The index.
			 * @param obst Pointer to an array of integers representing obstacles.
			 * @param f Pointer to an array of doubles representing distribution functions.
			 * @param ux The x-component of velocity.
			 * @param uy The y-component of velocity.
			 * @param uz The z-component of velocity.
			 */
			ONIKA_HOST_DEVICE_FUNC inline void operator()(
					int idx, 
					int * const obst, 
					double * const f,
					const double ux, 
					const double uy, 
					const double uz) const
			{
				const int idxQ=idx*Q;
				if (obst[idx] == FLUIDE_) {
					const double rho = (f[idxQ] + f[idxQ+1] + f[idxQ+2] + f[idxQ+3] + f[idxQ+4] + f[idxQ+7] + f[idxQ+9] + f[idxQ+10] + f[idxQ+8] + 2. * (f[idxQ+5] + f[idxQ+11] + f[idxQ+14] + f[idxQ+15] + f[idxQ+18])) / (1. + uz);
					const double nxz = (1. / 2.) * (f[idxQ+1] + f[idxQ+7] + f[idxQ+9] - (f[idxQ+2] + f[idxQ+10] + f[idxQ+8])) - (1. / 3.) * rho * ux;
					const double nyz = (1. / 2.) * (f[idxQ+3] + f[idxQ+7] + f[idxQ+10] - (f[idxQ+4] + f[idxQ+9] + f[idxQ+8])) - (1. / 3.) * rho * uy;

					f[idxQ+6] = f[idxQ+5] - (1. / 3.) * rho * uz;
					f[idxQ+13] = f[idxQ+14] + (1. / 6.) * rho * (-uz + ux) - nxz;
					f[idxQ+12] = f[idxQ+11] + (1. / 6.) * rho * (-uz - ux) + nxz;
					f[idxQ+17] = f[idxQ+18] + (1. / 6.) * rho * (-uz + uy) - nyz;
					f[idxQ+16] = f[idxQ+15] + (1. / 6.) * rho * (-uz - uy) + nyz;
				}
			}
		};

	/**
	 * @brief A functor for handling Neumann boundary conditions at z=0 in the lattice Boltzmann method.
	 */
	template<int Q>
		struct neumann_z_0
		{
			/**
			 * @brief Operator for applying Neumann boundary conditions at z=0.
			 *
			 * @param idx The index.
			 * @param obst Pointer to an array of integers representing obstacles.
			 * @param f Pointer to an array of doubles representing distribution functions.
			 * @param ux The x-component of velocity.
			 * @param uy The y-component of velocity.
			 * @param uz The z-component of velocity.
			 */
			ONIKA_HOST_DEVICE_FUNC inline void operator()(
					int idx, 
					int * const obst, 
					double * const f,
					const double &ux, 
					const double &uy, 
					const double &uz) const
			{
				const int idxQ = idx * Q;
				if (obst[idx] == FLUIDE_) {
					const double rho = (f[idxQ] + f[idxQ+1] + f[idxQ+2] + f[idxQ+3] + f[idxQ+4] + f[idxQ+7] + f[idxQ+9] + f[idxQ+10] + f[idxQ+8] + 2. * (f[idxQ+6] + f[idxQ+13] + f[idxQ+12] + f[idxQ+17] + f[idxQ+16])) / (1. - uz);
					const double nxz = (1. / 2.) * (f[idxQ+1] + f[idxQ+7] + f[idxQ+9] - (f[idxQ+2] + f[idxQ+10] + f[idxQ+8])) - (1. / 3.) * rho * ux;
					const double nyz = (1. / 2.) * (f[idxQ+3] + f[idxQ+7] + f[idxQ+10] - (f[idxQ+4] + f[idxQ+9] + f[idxQ+8])) - (1. / 3.) * rho * uy;

					f[idxQ+5] = f[idxQ+6] + (1. / 3.) * rho * uz;
					f[idxQ+11] = f[idxQ+12] + (1. / 6.) * rho * (uz + ux) - nxz;
					f[idxQ+14] = f[idxQ+13] + (1. / 6.) * rho * (uz - ux) + nxz;
					f[idxQ+15] = f[idxQ+16] + (1. / 6.) * rho * (uz + uy) - nyz;
					f[idxQ+18] = f[idxQ+17] + (1. / 6.) * rho * (uz - uy) + nyz;
				}
			}
		};

}
