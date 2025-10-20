/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */

#include <onika/math/basic_types.h>
#include <hippoLBM/compute/reduce.hpp>

namespace hippoLBM
{
	struct SimulationStatistics
	{
		double sum_density = 0.0;
		double min_velocity_norm = std::numeric_limits<double>::max();
		double max_velocity_norm = std::numeric_limits<double>::lowest();
    void display()
    {
			lout << "sum_density: " << sum_density 
				<< ", min_velocity_norm: " << min_velocity_norm 
				<< ", max_velocity_norm: " << max_velocity_norm << std::endl;
		}
	};

	struct ComuteSimulationStateFunc
	{
		double * const density;
		FieldView<3> velocity;

		ONIKA_HOST_DEVICE_FUNC inline void operator()( 
				SimulationStatistics& local, 
				const uint64_t idx, 
				reduce_thread_local_t = {}) const
		{
			local.sum_density += density[idx];
			onika::math::Vec3d v = velocity.get(idx);
			double v_norm = onika::math::norm(v);
			local.min_velocity_norm = std::min(local.min_velocity_norm, v_norm);
			local.max_velocity_norm = std::max(local.max_velocity_norm, v_norm);
      //std::cout << "max_velocity_norm " << local.max_velocity_norm << " v_norm " << v_norm << std::endl;
		}

		ONIKA_HOST_DEVICE_FUNC inline void operator()( 
				SimulationStatistics& global, 
				SimulationStatistics& local, 
				reduce_thread_block_t) const
		{
			ONIKA_CU_ATOMIC_ADD(global.sum_density, local.sum_density);
			ONIKA_CU_ATOMIC_MIN(global.min_velocity_norm, local.min_velocity_norm);
			ONIKA_CU_ATOMIC_MAX(global.max_velocity_norm, local.max_velocity_norm);
		}

		ONIKA_HOST_DEVICE_FUNC inline void operator()( 
				SimulationStatistics& global, 
				SimulationStatistics& local, 
				reduce_global_t) const
		{
			ONIKA_CU_ATOMIC_ADD(global.sum_density, local.sum_density);
			ONIKA_CU_ATOMIC_MIN(global.min_velocity_norm, local.min_velocity_norm);
			ONIKA_CU_ATOMIC_MAX(global.max_velocity_norm, local.max_velocity_norm);
		}
	};
}

namespace onika
{
	namespace parallel
	{
		template <> struct ParallelForFunctorTraits<hippoLBM::ComuteSimulationStateFunc>
		{
			static inline constexpr bool CudaCompatible = true;
		};
	} // namespace parallel
} // namespace onika


