#pragma once

#include <grid_lbm/wrapper_f.hpp>
#define FLUIDE_ -1

namespace hipoLBM
{
  template<int Q> struct neumann_z_0 {};
  template<int Q> struct neumann_z_l {};

  /**
   * @brief A functor for handling Neumann boundary conditions at z=lz in the lattice Boltzmann method.
   */
  template<>
    struct neumann_z_l<19>
    {
      static constexpr int Q = 19;
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
	  const WrapperF& f,
	  const double ux, 
	  const double uy, 
	  const double uz) const
      {
	const int idxQ=idx*Q;
	if (obst[idx] == FLUIDE_) {
	  const double rho = (f(idx,0) + f(idx,1) + f(idx,2) + f(idx,3) + f(idx,4) + f(idx,7) + f(idx,9) + f(idx,10) + f(idx,8) + 2. * (f(idx,5) + f(idx,11) + f(idx,14) + f(idx,15) + f(idx,18))) / (1. + uz);
	  const double nxz = (1. / 2.) * (f(idx,1) + f(idx,7) + f(idx,9) - (f(idx,2) + f(idx,10) + f(idx,8))) - (1. / 3.) * rho * ux;
	  const double nyz = (1. / 2.) * (f(idx,3) + f(idx,7) + f(idx,10) - (f(idx,4) + f(idx,9) + f(idx,8))) - (1. / 3.) * rho * uy;

	  f(idx,6) = f(idx,5) - (1. / 3.) * rho * uz;
	  f(idx,13) = f(idx,14) + (1. / 6.) * rho * (-uz + ux) - nxz;
	  f(idx,12) = f(idx,11) + (1. / 6.) * rho * (-uz - ux) + nxz;
	  f(idx,17) = f(idx,18) + (1. / 6.) * rho * (-uz + uy) - nyz;
	  f(idx,16) = f(idx,15) + (1. / 6.) * rho * (-uz - uy) + nyz;
	}
      }
    };

  /**
   * @brief A functor for handling Neumann boundary conditions at z=0 in the lattice Boltzmann method.
   */
  template<>
    struct neumann_z_0<19>
    {
      static constexpr int Q = 19;
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
	  const WrapperF& f,
	  const double &ux, 
	  const double &uy, 
	  const double &uz) const
      {
	if (obst[idx] == FLUIDE_) {
	  const double rho = (f(idx,0) + f(idx,1) + f(idx,2) + f(idx,3) + f(idx,4) + f(idx,7) + f(idx,9) + f(idx,10) + f(idx,8) + 2. * (f(idx,6) + f(idx,13) + f(idx,12) + f(idx,17) + f(idx,16))) / (1. - uz);
	  const double nxz = (1. / 2.) * (f(idx,1) + f(idx,7) + f(idx,9) - (f(idx,2) + f(idx,10) + f(idx,8))) - (1. / 3.) * rho * ux;
	  const double nyz = (1. / 2.) * (f(idx,3) + f(idx,7) + f(idx,10) - (f(idx,4) + f(idx,9) + f(idx,8))) - (1. / 3.) * rho * uy;

	  f(idx,5) = f(idx,6) + (1. / 3.) * rho * uz;
	  f(idx,11) = f(idx,12) + (1. / 6.) * rho * (uz + ux) - nxz;
	  f(idx,14) = f(idx,13) + (1. / 6.) * rho * (uz - ux) + nxz;
	  f(idx,15) = f(idx,16) + (1. / 6.) * rho * (uz + uy) - nyz;
	  f(idx,18) = f(idx,17) + (1. / 6.) * rho * (uz - uy) + nyz;
	}
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Q> struct ParallelForFunctorTraits<hipoLBM::neumann_z_0<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
    template<int Q> struct ParallelForFunctorTraits<hipoLBM::neumann_z_l<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
