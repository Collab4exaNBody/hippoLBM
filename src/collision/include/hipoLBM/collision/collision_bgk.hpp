#pragma once

#include <grid_lbm/wrapper_f.hpp>
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
       */
      ONIKA_HOST_DEVICE_FUNC inline void operator()(
	  int idx, 
	  Vec3d * m1,
	  int * const obst, 
	  const WrapperF& f,
	  double * const m0, 
	  const int* ex, 
	  const int* ey,
	  const int* ez, 
	  const double* const w, 
	  double tau) const
      {
	if (obst[idx] == FLUIDE_) 
	{
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
	    f(idx,iLB) += 1. / tau * (feq - f(idx,iLB)) + 3. * rho * w[iLB] * ef;
	  }
	}
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Q> struct ParallelForFunctorTraits<hipoLBM::collision_bgk<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
