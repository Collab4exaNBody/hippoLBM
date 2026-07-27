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
#include <onika/parallel/block_parallel_for.h>

#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/grid.hpp>

namespace hippoLBM {
template <class Obj>
struct SetObstacleFunc {
  Obj obj_;                       // The obstacle object to be applied.
  LBMGrid grid_;                  // Computes grid indices from (i,j,k) coordinates in the LBM domain.
  int* const __restrict__ obst_;  // Pointer to the obstacle field.
  int value_ = WALL_;             // The value to set for obstacle cells in the obstacle field.

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k) const {
    if (obj_.solid(grid_.compute_position<hippoLBM::Area::Global>(i, j, k))) {
      const int idx = grid_(i, j, k);
      obst_[idx] = value_;  // Mark the cell as an obstacle (e.g., WALL_)
    }
  }
};

template <typename ParExecCtxFunc>
struct ApplyUpdateObstaclesFunc {
  LBMGrid grid_;                 // The LBM grid containing the simulation data.
  double dx_;                    // The grid spacing of the LBM simulation.
  int* const obst_;              // Pointer to the obstacle field in the LBM grid.
  ParExecCtxFunc par_exec_ctx_;  // Function to obtain the parallel execution context.
  int value_ = WALL_;            // The value to set for obstacle cells in the obstacle field.

  template <typename Obj>
  inline void operator()(Obj& obj) const {
    // convert bounds in box
    onika::math::AABB bounds = obj.covered();
    onika::math::Vec3d min = bounds.bmin;
    onika::math::Vec3d max = bounds.bmax;
    Point3D _min = {int(min.x / dx_), int(min.y / dx_), int(min.z / dx_)};
    Point3D _max = {int(max.x / dx_), int(max.y / dx_), int(max.z / dx_)};
    Box3D global_box = {_min, _max};

    auto [is_inside_subdomain, local_box] = grid_.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);

    if (is_inside_subdomain) {
      SetObstacleFunc func = {obj, grid_, obst_, value_};
      hippoLBM::parallel_for(local_box, func, par_exec_ctx_("update_obstacles"));
    }
  }

  inline void operator()(RShape& obj) const {
    apply_to_grid(obj, grid_, obst_, par_exec_ctx_("update_obstacles_rshape"), value_);
  }
};

struct ApplyRShapeToGridFunctor {
  const onika::math::Vec3d* const __restrict__ vertices_;
  const uint32_t* const __restrict__ offset_;
  const uint32_t* const __restrict__ size_;
  double minkowski_;
  LBMGrid grid_;
  int* const __restrict__ obst_;
  int value_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const {
    const size_t f = coord.x;
    std::span<const onika::math::Vec3d> vertices(vertices_ + offset_[f], size_[f]);
    onika::math::AABB bounds = compute_aabb(vertices, minkowski_);
    Point3D pmin = {int(bounds.bmin.x / grid_.dx_), int(bounds.bmin.y / grid_.dx_), int(bounds.bmin.z / grid_.dx_)};
    Point3D pmax = {int(bounds.bmax.x / grid_.dx_), int(bounds.bmax.y / grid_.dx_), int(bounds.bmax.z / grid_.dx_)};
    Box3D global_box = {pmin, pmax};

    auto [is_inside_subdomain, local_box] = grid_.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);
    if (!is_inside_subdomain) return;

    for (int k = local_box.start(2) + ONIKA_CU_THREAD_COORD.z; k <= local_box.end(2); k += ONIKA_CU_BLOCK_DIMS.z) {
      for (int j = local_box.start(1) + ONIKA_CU_THREAD_COORD.y; j <= local_box.end(1); j += ONIKA_CU_BLOCK_DIMS.y) {
        for (int i = local_box.start(0) + ONIKA_CU_THREAD_COORD.x; i <= local_box.end(0); i += ONIKA_CU_BLOCK_DIMS.x) {
          onika::math::Vec3d p = grid_.compute_position<Area::Global>(i, j, k);
          if (intersect_point_face(p, vertices, minkowski_)) {
            obst_[grid_(i, j, k)] = value_;
          }
        }
      }
    }
  }
};

inline void apply_to_grid(const RShape& rshape, const LBMGrid& grid, int* const obst,
                          onika::parallel::ParallelExecutionContext* exec_ctx, int value) {
  if (rshape.faces_ != 0) {
    ApplyRShapeToGridFunctor func = {
        rshape.vertices_.data(), rshape.offset_.data(), rshape.size_.data(), rshape.minkowski_, grid, obst, value};
    onika::parallel::ParallelExecutionSpace<3> space = {{0, 0, 0}, {rshape.faces_, 1, 1}};
    onika::parallel::block_parallel_for(space, func, exec_ctx);
  }
}
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <>
struct BlockParallelForFunctorTraits<hippoLBM::ApplyRShapeToGridFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <typename Obj>
struct ParallelForFunctorTraits<hippoLBM::SetObstacleFunc<Obj>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
