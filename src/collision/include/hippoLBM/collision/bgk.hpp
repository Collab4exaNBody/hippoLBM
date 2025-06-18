#pragma once

#include <hippoLBM/grid/field_view.hpp>
#define FLUIDE_ -1

namespace hippoLBM
{
  /**
   * @brief A functor for collision operations in the lattice Boltzmann method.
   */
  template<int Q, Traversal TR>
    struct bgk
    {
      const int * __restrict__ levels; // It contains the traversal level (0 inside, 0 1 Real, 0 1 2 Extend, and 0 1 2 3 All 
      const Vec3d m_Fext;
      const FieldView<3> m1;
      int * const __restrict__ obst;
      const FieldView<Q> f;
      double * const __restrict__ m0;
      const int* const __restrict__ ex; 
      const int* const __restrict__ ey;
      const int* const __restrict__ ez; 
      const double* const __restrict__ w; 
      const double tau;

      /**
       * @brief Operator for performing collision operations at a given index.
       */
      ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const 
      {

	bool update = check_level<TR>(levels[idx]) && (obst[idx] == FLUIDE_); 
	const double& rho = m0[idx];
	const double& ux = m1(idx,0);
	const double& uy = m1(idx,1);
	const double& uz = m1(idx,2);
	const double ___u_squ = -1.5 * (ux * ux + uy * uy + uz * uz);

	for (int iLB = 0; iLB < Q; iLB++) 
	{
	  const int &exiLB = ex[iLB];
	  const int &eyiLB = ey[iLB];
	  const int &eziLB = ez[iLB];
	  const double &wiLB = w[iLB];
	  double &fiLB = f(idx,iLB);
	  double ef  = exiLB * m_Fext.x + eyiLB * m_Fext.y + eziLB * m_Fext.z;
	  double eu  = exiLB * ux + eyiLB * uy + eziLB * uz;
	  //double feq = wiLB * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
	  double feq = wiLB * rho * (1. +  eu * (3. + 4.5 * eu) + ___u_squ);
	  fiLB += update * ((feq - fiLB) + 3. * wiLB * ef)/tau;
	}
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Q, hippoLBM::Traversal Tr> struct ParallelForFunctorTraits<hippoLBM::bgk<Q,Tr>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
