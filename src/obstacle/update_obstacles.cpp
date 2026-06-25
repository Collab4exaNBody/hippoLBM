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

#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/obstacle/obstacles.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
struct UpdateObstaclesFunc {
  LBMGrid _grid_;       // The LBM grid containing the simulation data.
  double _dx_;          // The grid spacing of the LBM simulation.
  int* const _obst_;    // Pointer to the obstacle field in the LBM grid.
  int _value_ = WALL_;  // The value to set for obstacle cells in the obstacle field.

  template <typename Obj>
  inline void operator()(Obj& obj) const {
    // convert bounds in box
    onika::math::AABB bounds = obj.covered();
    onika::math::Vec3d min = bounds.bmin;
    onika::math::Vec3d max = bounds.bmax;
    Point3D _min = {int(min.x / _dx_), int(min.y / _dx_), int(min.z / _dx_)};
    Point3D _max = {int(max.x / _dx_), int(max.y / _dx_), int(max.z / _dx_)};
    Box3D global_box = {_min, _max};

    auto [is_inside_subdomain, local_box] = _grid_.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);
    for (int z = local_box.start(2); z <= local_box.end(2); z++)
      for (int y = local_box.start(1); y <= local_box.end(1); y++)
        for (int x = local_box.start(0); x <= local_box.end(0); x++) {
          if (obj.solid(_grid_.compute_position<Area::Global>(x, y, z))) {
            const int idx = _grid_(x, y, z);
            _obst_[idx] = _value_;
          }
        }
  }
};

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

    UpdateObstaclesFunc<Q> func = {domain->m_grid_, domain->dx(), grid_data.obstacles()};

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
