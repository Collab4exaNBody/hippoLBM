#pragma once

#include<hippoLBM/grid/field_view.hpp>
#include<hippoLBM/grid/grid_helper.hpp>

namespace hippoLBM
{
  /**
   * @brief Initializes the distributions in a lattice Boltzmann model.
   */
  template<int Q>
    struct init_distributions
    {
      double coeff;
      GridIJKtoIdx ijk_to_idx;;
      /**
       * @brief Operator to initialize distributions at a given index.
       *
       * @param idx The index to initialize distributions.
       * @param f Pointer to the distribution function.
       * @param w Pointer to the weight coefficients.
       */
      ONIKA_HOST_DEVICE_FUNC inline void operator()(const int idx, const FieldView<Q>& f, const double* const w) const
      {
        for (int iLB = 0; iLB < Q; iLB++)
        {
          f(idx,iLB) = coeff * w[iLB];
        }
      };

      ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k, const FieldView<Q>& f, const double* const w) const
      {
        this->operator()(ijk_to_idx(i,j,k), f, w);
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template <int Q> struct ParallelForFunctorTraits<hippoLBM::init_distributions<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true
        ;
    };
  }
}
