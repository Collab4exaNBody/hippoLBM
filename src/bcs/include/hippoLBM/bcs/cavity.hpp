#pragma once

#include <grid_lbm/wrapper_f.hpp>
#define FLUIDE_ -1

namespace hippoLBM
{
  template<int Q> struct cavity_z_l {};

	template<int dim, Direction dir, int Q> struct cavity_coeff{};
  template<> struct cavity_coeff<2, Direction::Right, 19> { int fid[5] = {6,13,12,17,16};};
  //template<> struct cavity_coeff<2, Direction::Right, 19> { int fid[5] = {5,14,11,18,15};};

  template<>
    struct cavity_z_l<19>
    {
      static constexpr int Q = 19;
      const cavity_coeff<2, Direction::Right, Q> coeff;
      static constexpr int Un = 5;

      /**
       * @brief operator for applying neumann boundary conditions at z=0.
       *
       * @param idxq the index.
       * @param obst pointer to an array of integers representing obstacles.
       * @param f pointer to an array of doubles representing distribution functions.
       * @param ux the x-component of velocity.
       * @param uy the y-component of velocity.
       * @param uz the z-component of velocity.
       */
      ONIKA_HOST_DEVICE_FUNC inline void operator()(
          int idx, 
          int * const obst, 
          const WrapperF<Un>& fi,
          const double ux, 
          const double uy, 
          const double uz,
					const double * const w,
					const int* ex, 
					const int* ey, 
					const int* ez,
					const int lx, 
					const int ly, 
					const int lz) const
			{
				assert(uz == 0 && "uz should be equal to 0");
				if (obst[idx] == FLUIDE_) {

					const double uxx = ux * ( 1 + 0.5/ ( lz - 1 ));
					const double uyy = uy * ( 1 + 0.5/ ( lz - 1 ));
					const double uzz = 0; // uz * ( 1 + 0.5/ ( lz - 1 ));
          
#pragma GCC unroll 5
          for(int i = 0 ; i < Un ; i++)
          {
            const int fid = coeff.fid[i];
            fi(idx,i) += 6. * w[fid] * (ex[fid] * uxx + ey[fid] * uyy ); // + ez[fid] * uzz);
          }
				}
			}
		};
}

namespace onika
{
	/*
		 namespace parallel
		 {
		 template<int Q> struct ParallelForFunctorTraits<hippoLBM::neumann_z_0<Q>>
		 {
		 static inline constexpr bool RequiresBlockSynchronousCall = false;
		 static inline constexpr bool CudaCompatible = true;
		 };
	 */
}
