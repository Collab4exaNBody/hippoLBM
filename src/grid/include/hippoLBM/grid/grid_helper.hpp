#pragma once

#include <onika/math/basic_types_def.h>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/core/point3d.hpp>
#include <hippoLBM/core/box3d.hpp>
#include <hippoLBM/grid/grid.hpp>
#include <cassert>
#include <tuple>

namespace hippoLBM
{
  // helper
  struct GridIJKtoIdx // local
  {
    Box3D bx;
    GridIJKtoIdx() = delete;
    GridIJKtoIdx(const LBMGrid& grid) { bx = grid.bx; }
    GridIJKtoIdx(const Box3D& in) { bx = in; }
		ONIKA_HOST_DEVICE_FUNC int operator()(Point3D& p) const  { return bx(p[0], p[1], p[2]); }  
		ONIKA_HOST_DEVICE_FUNC int operator()(Point3D&& p) const { return bx(p[0], p[1], p[2]); }  
		ONIKA_HOST_DEVICE_FUNC int operator()(int x, int y, int z) const { return bx(x, y, z); }  
		ONIKA_HOST_DEVICE_FUNC inline std::tuple<int,int,int> operator()(int idx) const { return bx(idx);	}  
  };
}
