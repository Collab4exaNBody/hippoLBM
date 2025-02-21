#pragma once

#include <grid_lbm/enum.hpp>

namespace hippoLBM
{
  constexpr int DIMX = 0;
	constexpr int DIMY = 1;
	constexpr int DIMZ = 2;
	constexpr Direction LEFT = Direction::Left;
	constexpr Direction RIGHT = Direction::Right;

	template<int dim, Direction dir>
		inline constexpr Traversal get_traversal();

	template<> inline constexpr Traversal get_traversal<DIMX,  LEFT>() { return Traversal::Plan_yz_0; }
	template<> inline constexpr Traversal get_traversal<DIMX, RIGHT>() { return Traversal::Plan_yz_l; }
	template<> inline constexpr Traversal get_traversal<DIMY,  LEFT>() { return Traversal::Plan_xz_0; }
	template<> inline constexpr Traversal get_traversal<DIMY, RIGHT>() { return Traversal::Plan_xz_l; }
	template<> inline constexpr Traversal get_traversal<DIMZ,  LEFT>() { return Traversal::Plan_xy_0; }
	template<> inline constexpr Traversal get_traversal<DIMZ, RIGHT>() { return Traversal::Plan_xy_l; }

  ////////////////////// Pre streaming ///////////////////////////

	template<int dim, Direction dir, int Q>
		struct pre_bounce_back {};

	template<int dim, Direction dir, int Q> struct pre_bounce_back_coeff{};
	template<> struct pre_bounce_back_coeff<DIMX, LEFT, 19> { int fid[5] = {2,10,8,12,14};};
	template<> struct pre_bounce_back_coeff<DIMX, RIGHT,19> { int fid[5] = {1,9,7,11,13};	};

	template<> struct pre_bounce_back_coeff<DIMY, LEFT, 19> { int fid[5] = {4,8,9,16,18};	};
	template<> struct pre_bounce_back_coeff<DIMY, RIGHT,19> { int fid[5] = {3,7,10,15,17};};

	template<> struct pre_bounce_back_coeff<DIMZ, LEFT, 19> { int fid[5] = {6,13,12,17,16};};
	template<> struct pre_bounce_back_coeff<DIMZ, RIGHT,19> { int fid[5] = {5,14,11,18,15};};


	template<int dim, Direction dir> struct pre_bounce_back<dim, dir, 19>
	{
		const int * const traversal; 
		static constexpr int Un = 5;
		pre_bounce_back_coeff<dim, dir, 19> coeff;
		ONIKA_HOST_DEVICE_FUNC inline void operator()(
				int idx, 
				const WrapperF<19>& f, // data could be modified, but the ptr inside WrapperF can't be modified
				const WrapperF<Un>& fi) const
		{
			const int fidx = traversal[idx];
#pragma GCC unroll 5
			for(int i = 0 ; i < 5 ; i++)
			{
				fi(idx, i) = f(fidx, coeff.fid[i]);
			}
		}
	};

  ////////////////////// Post streaming ///////////////////////////

	template<int dim, Direction dir, int Q>
		struct post_bounce_back {};

	template<int dim, Direction dir, int Q> struct post_bounce_back_coeff{};
	template<> struct post_bounce_back_coeff<DIMX, LEFT, 19> { int fid[5] = {1,9,7,11,13};};
	template<> struct post_bounce_back_coeff<DIMX, RIGHT,19> { int fid[5] = {2,10,8,12,14};	};

	template<> struct post_bounce_back_coeff<DIMY, LEFT, 19> { int fid[5] = {3,7,10,15,17};	};
	template<> struct post_bounce_back_coeff<DIMY, RIGHT,19> { int fid[5] = {4,8,9,16,18};};

	template<> struct post_bounce_back_coeff<DIMZ, LEFT, 19> { int fid[5] = {5,14,11,18,15};};
	template<> struct post_bounce_back_coeff<DIMZ, RIGHT,19> { int fid[5] = {6,13,12,17,16};};

	template<int dim, Direction dir> struct post_bounce_back<dim, dir, 19>
	{
		const int * const traversal; 
		static constexpr int Un = 5;
		post_bounce_back_coeff<dim, dir, 19> coeff;
		ONIKA_HOST_DEVICE_FUNC inline void operator()(
				int idx, 
				const WrapperF<19>& f, // data could be modified, but the ptr inside WrapperF can't be modified
				const WrapperF<Un>& fi) const
		{
			const int fidx = traversal[idx];
      //onika::lout << " idx " << idx << " <-> f idx " << fidx << std::endl;
#pragma GCC unroll 5 
			for(int i = 0 ; i < 5 ; i++)
			{
				f(fidx, coeff.fid[i]) = fi(idx, i);
			}
		}
	};
}
