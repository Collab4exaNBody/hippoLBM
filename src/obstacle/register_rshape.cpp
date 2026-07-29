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

#include <onika/math/quaternion.h>
#include <onika/math/quaternion_operators.h>
#include <onika/math/quaternion_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <hippoLBM/obstacle/obstacles.hpp>
#include <string>

namespace hippoLBM {
using namespace onika;
using namespace scg;

class RegisterRShape : public OperatorNode {
  ADD_SLOT(Obstacles, obstacles, INPUT_OUTPUT, REQUIRED, DocString{"List of Obstacles"});
  ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
  ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"Path to the STL file (ASCII or binary)."});
  ADD_SLOT(onika::math::Vec3d, center, INPUT, (onika::math::Vec3d{0, 0, 0}),
           DocString{"Translation applied to the mesh, after scaling and rotation."});
  ADD_SLOT(double, scale, INPUT, 1.0, DocString{"Scale factor applied to the mesh, before rotation and translation."});
  ADD_SLOT(onika::math::Quaternion, orientation, INPUT, (onika::math::Quaternion{1, 0, 0, 0}),
           DocString{"Orientation applied to the mesh, as a quaternion [w,x,y,z]. Default: identity."});
  ADD_SLOT(double, minkowski, INPUT, 0.0, DocString{"Minkowski (rounding) radius of the shape."});

 public:
  inline std::string documentation() const override final {
    return R"EOF(
        Reads an STL mesh into an RShape (scaled, oriented, then shifted),
        and adds it to the obstacles list.

        YAML example:

        set_obstacles:
          - register_rshape:
             id: 0
             filename: "shape.stl"
             center: [0.15, 0.05, 0.05]
             scale: 1.0
             orientation: [1, 0, 0, 0]   # identity, [w,x,y,z]
             minkowski: 0.001
        )EOF";
  }

  inline void execute() override final {
    RShape shape{};
    shape.minkowski_ = *minkowski;
    shape.read_stl(*filename);

    const onika::math::Quaternion& q = *orientation;
    const onika::math::Vec3d& c = *center;
    const double s = *scale;
    for (auto& v : shape.vertices_) {
      v = c + q * (v * s);
    }

    obstacles->add(*id, shape);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(register_rshape) {
  OperatorNodeFactory::instance()->register_factory("register_rshape", make_simple_operator<RegisterRShape>);
}
}  // namespace hippoLBM
