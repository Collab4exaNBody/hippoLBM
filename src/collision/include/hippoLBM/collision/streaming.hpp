#pragma once

#include <grid/field_view.hpp>

namespace hippoLBM
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
      ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, const FieldView<Q>& f) const
      {
        for (int iLB = 1; iLB < Q; iLB += 2)
        {
          std::swap(f(idx,iLB), f(idx,iLB+1));
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
      grid<3> g;
      const FieldView<Q> f;
      const int* __restrict__ const ex;
      const int* __restrict__ const ey; 
      const int* __restrict__ const ez;
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
      ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const
      {

        const int idx = g(coord.x,coord.y,coord.z);
        for (int iLB = 1; iLB < Q; iLB += 2)
        {
          const int next_x = coord.x + ex[iLB];
          const int next_y = coord.y + ey[iLB];
          const int next_z = coord.z + ez[iLB];

          if(g.is_defined(next_x, next_y, next_z))
          {
            const int next_idx = g(next_x, next_y, next_z);
            std::swap(f(idx,iLB+1), f(next_idx, iLB));
          } 
        }
      }
    };
}


namespace onika
{
  namespace parallel
  {
    template<int Q> struct ParallelForFunctorTraits<hippoLBM::streaming_step1<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = true;
      static inline constexpr bool CudaCompatible = true;
    };

    template<int Q> struct ParallelForFunctorTraits<hippoLBM::streaming_step2<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
