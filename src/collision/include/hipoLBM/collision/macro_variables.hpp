#pragma once

#define FLUIDE_ -1

namespace hipoLBM
{
  using namespace onika::math;
	/**
	 * @brief A functor for computing macroscopic variables for lattice Boltzmann method.
	 */
	template<int Q>
		struct macro_variables
		{
			const Vec3d Fext_2;
			ONIKA_HOST_DEVICE_FUNC inline void operator()(
					const int idx, 
					Vec3d * pm1, 
					int * const pobst, 
					double * const pf,
					double * const pm0,
					const int* pex, 
					const int* pey, 
					const int* pez) const
			{
				double rho, ux, uy, uz;
				int idxQ = idx * Q;
				if (pobst[idx] == FLUIDE_) {
					rho = ux = uy = uz = 0.;
					for (int iLB = 0; iLB < Q; iLB++) {
						const double s = pf[idxQ+iLB];
						ux += s * pex[iLB];
						uy += s * pey[iLB];
						uz += s * pez[iLB];
						rho += s;
					}

					if (rho != 0.) {
						ux /= rho;
						uy /= rho;
						uz /= rho;
					}

					pm0[idx] = rho;
					pm1[idx] = onika::math::Vec3d(ux, uy, uz); // + Fext_2; // why?
				}
			}
		};
}
