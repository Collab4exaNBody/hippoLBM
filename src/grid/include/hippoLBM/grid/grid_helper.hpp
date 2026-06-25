#pragma once

#include <onika/math/basic_types_def.h>

#include <hippoLBM/core/box3d.hpp>
#include <hippoLBM/core/point3d.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/grid.hpp>

namespace hippoLBM {
// helper
struct GridIJKtoIdx  // local
{
  Box3D bx_;  // local box to compute the index
  GridIJKtoIdx() = delete;
  GridIJKtoIdx(const LBMGrid& grid) { bx_ = grid.bx_; }
  GridIJKtoIdx(const Box3D& in) { bx_ = in; }
  ONIKA_HOST_DEVICE_FUNC int operator()(Point3D& p) const { return bx_(p[0], p[1], p[2]); }
  ONIKA_HOST_DEVICE_FUNC int operator()(Point3D&& p) const { return bx_(p[0], p[1], p[2]); }
  ONIKA_HOST_DEVICE_FUNC int operator()(int x, int y, int z) const { return bx_(x, y, z); }
  ONIKA_HOST_DEVICE_FUNC inline std::tuple<int, int, int> operator()(int idx) const { return bx_(idx); }
};
}  // namespace hippoLBM
