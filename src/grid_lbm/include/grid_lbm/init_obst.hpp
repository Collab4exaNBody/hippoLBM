#pragma once

#define FLUIDE_ -1

namespace hipoLBM
{
	/**
	 * @brief Initializes the obst in a lattice Boltzmann model.
	 */
	template<int Q>
		struct init_obst
		{
      int * obst;
			ONIKA_HOST_DEVICE_FUNC inline void operator()(const int idx) const
			{
				obst[idx] = FLUIDE_;
			};
		};
}

namespace onika
{
  namespace parallel
  {
    template <int Q> struct ParallelForFunctorTraits<hipoLBM::init_obst<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
