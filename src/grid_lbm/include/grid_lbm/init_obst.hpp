#pragma once

#define FLUIDE_ -1

namespace hippoLBM
{
	/**
	 * @brief Initializes the obst in a lattice Boltzmann model.
	 */
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
		template<> struct ParallelForFunctorTraits<hippoLBM::init_obst>
		{
			static inline constexpr bool RequiresBlockSynchronousCall = false;
			static inline constexpr bool CudaCompatible = true;
		};
	}
}
