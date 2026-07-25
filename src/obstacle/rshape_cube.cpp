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

#include <onika/math/quaternion.h>
#include <onika/math/quaternion_operators.h>
#include <onika/math/quaternion_yaml.h>

#include <span>

#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/obstacle/rshape.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
class RShapeCube : public OperatorNode {
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain containing the simulation data."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"The LBM fields containing the simulation data."});
  ADD_SLOT(onika::math::Vec3d, center, INPUT, REQUIRED, DocString{"Center of the cube."});
  ADD_SLOT(double, length, INPUT, REQUIRED, DocString{"Side length of the cube."});
  ADD_SLOT(onika::math::Quaternion, orientation, INPUT, (onika::math::Quaternion{1, 0, 0, 0}),
           DocString{"Orientation of the cube, as a quaternion [w,x,y,z]. Default: identity."});
  ADD_SLOT(double, minkowski, INPUT, 0.0, DocString{"Minkowski (rounding) radius of the shape."});

 public:
  inline std::string documentation() const override final {
    return R"EOF(
        Initializes an RShape as a cube, defined by its center, side length,
        orientation (quaternion), and a Minkowski (rounding) radius, and marks
        every LBM grid node it covers as WALL in the obstacle field.

        YAML example:

        rshape_cube:
          center: [0.05, 0.05, 0.05]
          length: 0.02
          orientation: [1, 0, 0, 0]   # identity, [w,x,y,z]
          minkowski: 0.001
        )EOF";
  }

  inline void execute() override final {
    const double hl = 0.5 * (*length);
    const onika::math::Quaternion& q = *orientation;
    const onika::math::Vec3d& c = *center;

    // Local (unrotated) cube corners, centered on the origin.
    const onika::math::Vec3d local[8] = {
        {-hl, -hl, -hl}, {hl, -hl, -hl}, {hl, hl, -hl}, {-hl, hl, -hl},
        {-hl, -hl, hl},  {hl, -hl, hl},  {hl, hl, hl},  {-hl, hl, hl},
    };
    onika::math::Vec3d corner[8];
    for (int i = 0; i < 8; i++) corner[i] = c + q * local[i];

    RShape shape{};
    shape.minkowski_ = *minkowski;

    // Each face is listed with outward-facing, consistent CCW winding.
    constexpr int faces[6][4] = {
        {0, 3, 2, 1},  // bottom (z = -hl)
        {4, 5, 6, 7},  // top    (z = +hl)
        {0, 1, 5, 4},  // front  (y = -hl)
        {3, 7, 6, 2},  // back   (y = +hl)
        {1, 2, 6, 5},  // right  (x = +hl)
        {0, 4, 7, 3},  // left   (x = -hl)
    };
    for (const auto& f : faces) {
      onika::math::Vec3d verts[4] = {corner[f[0]], corner[f[1]], corner[f[2]], corner[f[3]]};
      shape.add_face(std::span<onika::math::Vec3d>(verts, 4));
    }

    shape.apply_to_grid(domain->grid(), fields->obstacles(), WALL_);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(rshape_cube) {
  OperatorNodeFactory::instance()->register_factory("rshape_cube", make_variant_operator<RShapeCube>);
}
}  // namespace hippoLBM
