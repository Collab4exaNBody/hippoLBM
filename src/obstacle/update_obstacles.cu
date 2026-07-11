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
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/obstacle/obstacles.hpp>

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
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <typename Obj>
struct ParallelForFunctorTraits<hippoLBM::SetObstacleFunc<Obj>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika

namespace hippoLBM {
using namespace onika;
using namespace scg;
template <int Q>
class UpdateObstacles : public OperatorNode {
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain containing the simulation data."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, DocString{"The LBM fields containing the simulation data."});
  ADD_SLOT(Obstacles, obstacles, INPUT_OUTPUT, REQUIRED, DocString{"List of Obstacles"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator updates the obstacle field in the LBM grid based on the defined obstacles in the simulation.

        YAML example:

		  - update_obstacles

        )EOF";
  }

  inline void execute() final {
    auto& obs = *obstacles;
    LBMFields<Q>& grid_data = *fields;

    // capture the parallel execution context
    auto par_exec_ctx = [this](const char* exec_name) { return this->parallel_execution_context(exec_name); };

    ApplyUpdateObstaclesFunc func = {domain->grid(), domain->dx(), grid_data.obstacles(), par_exec_ctx,
                                     hippoLBM::WALL_};

    for (size_t i = 0; i < obs.size(); i++) {
      obs.apply(i, func);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(update_obstacles) {
  OperatorNodeFactory::instance()->register_factory("update_obstacles", make_variant_operator<UpdateObstacles>);
}
}  // namespace hippoLBM
