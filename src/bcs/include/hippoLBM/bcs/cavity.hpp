#pragma once

#include <grid_lbm/enum.hpp>
#include <grid_lbm/field_view.hpp>

namespace hippoLBM
{
	template<int DIM, Side S, int Q> struct cavity{};
	template<int dim, Side dir, int Q> struct cavity_coeff{};

	template<> struct cavity_coeff<DIMZ, Side::Left, 19> { int fid[5] = {5,14,11,18,15};};
	template<> struct cavity_coeff<DIMZ, Side::Right, 19> { int fid[5] = {6,13,12,17,16};};

	template<int Dim, Side S>
		struct cavity<Dim, S, 19>
		{
			static constexpr int Q = 19;
			static constexpr int Un = 5;
			double coeff[Un];

			void compute_coeff(
					double ux, double uy, double uz,
					const double * const w,
					const int* ex, const int* ey, const int* ez,
					int lx, int ly, int lz)
			{
				const cavity_coeff<DIMZ, S, Q> c_coeff;
				double L = 0;
				if constexpr (Dim == DIMZ) L = lx;
				if constexpr (Dim == DIMY) L = ly;
				if constexpr (Dim == DIMZ) L = lz;
				const double uxx = ux * ( 1 + 0.5/ ( L - 1 ));
				const double uyy = uy * ( 1 + 0.5/ ( L - 1 ));
				const double uzz = uz * ( 1 + 0.5/ ( L - 1 ));
#pragma GCC unroll 5
				for(int i = 0 ; i < Un ; i++)
				{
					const int fid = c_coeff.fid[i];
					coeff[i] = 6. * w[fid] * (ex[fid] * uxx + ey[fid] * uyy + ez[fid] * uzz);
				}
			}

			ONIKA_HOST_DEVICE_FUNC inline void operator()(
					int idx, 
					int * const obst, 
					const FieldView<Un>& fi) const
			{
				if (obst[idx] == FLUIDE_) {
#pragma GCC unroll 5
					for(int i = 0 ; i < Un ; i++)
					{
						fi(idx,i) += coeff[i];
					}
				}
			}
		};
}

namespace onika
{
	namespace parallel
	{
		template<int Dim, hippoLBM::Side S, int Q> struct ParallelForFunctorTraits<hippoLBM::cavity<Dim, S, Q>>
		{
			static inline constexpr bool RequiresBlockSynchronousCall = false;
			static inline constexpr bool CudaCompatible = true;
		};
	}
}
