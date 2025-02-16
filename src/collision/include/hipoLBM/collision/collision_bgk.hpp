#pragma once

#include <grid_lbm/wrapper_f.hpp>
#define FLUIDE_ -1

namespace hippoLBM
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
          Vec3d * __restrict__ m1,
          int * const __restrict__ obst, 
          const WrapperF<Q>& f,
          double * const __restrict__ m0, 
          const int* __restrict__ ex, 
          const int* __restrict__ ey,
          const int* __restrict__ ez, 
          const double* const __restrict__ w, 
          const double tau) const
      {
        if (obst[idx] == FLUIDE_) 
        {
          const double rho = m0[idx];
          const double ux = m1[idx].x;
          const double uy = m1[idx].y;
          const double uz = m1[idx].z;
          const double u_squ = ux * ux + uy * uy + uz * uz;
 
          const double inv_tau = 1. / tau;
#pragma omp simd
          for (int iLB = 0; iLB < Q; iLB++) 
          {
            double eu, feq, ef;
            const double exiLB = ex[iLB];
            const double eyiLB = ey[iLB];
            const double eziLB = ez[iLB];
            const double wiLB = w[iLB] * rho;
            ef = exiLB * m_Fext.x + ey[iLB] * m_Fext.y + eziLB * m_Fext.z;
            eu = exiLB * ux       + eyiLB * uy         + eziLB * uz;
            //feq = w[iLB] * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
            feq = w[iLB] * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
            //f(idx,iLB) += inv_tau * (feq - f(idx,iLB)) + 3. * rho * w[iLB] * ef;
            double& fiLB = f(idx,iLB); 
            fiLB += inv_tau * (feq - fiLB) + 3. * wiLB * ef;
          }
        }
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Q> struct ParallelForFunctorTraits<hippoLBM::collision_bgk<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
