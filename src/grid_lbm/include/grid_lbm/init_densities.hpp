#pragma once

namespace hipoLBM
{
	/**
	 * @brief Initializes the densities in a lattice Boltzmann model.
	 */
	template<int Q>
		struct init_densities
		{
			/**
			 * @brief Operator to initialize densities at a given index.
			 *
			 * @param idx The index to initialize densities.
			 * @param f Pointer to the distribution function.
			 * @param w Pointer to the weight coefficients.
			 */
			ONIKA_HOST_DEVICE_FUNC void operator()(const int idx, double* const f, const double* const w) const
			{
        int i = idx * Q;
				for (int iLB = 0; iLB < Q; iLB++)
				{
					f[i+iLB]=w[iLB];
				}
			};
		};
}

namespace onika
{
  namespace parallel
  {
    template <int Q> struct ParallelForFunctorTraits<hipoLBM::init_densities<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
