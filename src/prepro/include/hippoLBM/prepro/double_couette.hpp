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

#pragma once

#include <hippoLBM/grid/enum.hpp>

namespace hippoLBM
{
	template<int Q, int DIM> struct InitDoubleCouetteFunc
	{
		LBMGrid g;
		FieldView<Q> f;
		const Vec3d dU_lbm; // u_lbm.x / (.5 * (l_dir (x,y,z) - 1));
		const Vec3d U; // U_real / c;
		const int* ex;
		const int* ey;
		const int* ez;
		const double * const w;

		ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const
		{
			const int idx = g(coord.x, coord.y, coord.z);
            double value;
			if constexpr (DIM == DIMX ) value = coord.x + g.offset[0];
			if constexpr (DIM == DIMY ) value = coord.y + g.offset[1];
			if constexpr (DIM == DIMZ ) value = coord.z + g.offset[2];

      Vec3d uii = U - dU_lbm * value;
   
			double eu;
		    double u_squ = dot(uii, uii);
			for(int iLB = 0 ; iLB < Q ; iLB++)
			{
				eu = uii.x * double(ex[iLB]) + uii.y * ey[iLB] + uii.z * ez[iLB];
				f(idx, iLB) = 1. * w[iLB] * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
			}
		}
	};
}

namespace onika
{
	namespace parallel
	{
		template<int Q, int DIM> struct ParallelForFunctorTraits<hippoLBM::InitDoubleCouetteFunc<Q, DIM>>
		{
			static inline constexpr bool RequiresBlockSynchronousCall = false;
			static inline constexpr bool CudaCompatible = true;
		};
	}
}
