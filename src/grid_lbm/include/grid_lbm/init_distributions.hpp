#pragma once

namespace hipoLBM
{
	/**
	 * @brief Initializes the distributions in a lattice Boltzmann model.
	 */
	template<int Q>
		struct init_distributions
		{
			/**
			 * @brief Operator to initialize distributions at a given index.
			 *
			 * @param idx The index to initialize distributions.
			 * @param f Pointer to the distribution function.
			 * @param w Pointer to the weight coefficients.
			 */
			ONIKA_HOST_DEVICE_FUNC void operator()(const int idx, double* const f, const double* const w) const
			{
        int idxQ = idx * Q;
				for (int iLB = 0; iLB < Q; iLB++)
				{
					f[idxQ+iLB]=w[iLB];
//          std::cout << "f " << f[idxQ+iLB] << std::endl; 
				}
			};
		};
}

namespace onika
{
  namespace parallel
  {
    template <int Q> struct ParallelForFunctorTraits<hipoLBM::init_distributions<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true
;
    };
  }
}
