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
#include <onika/math/matrix4d.h>
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
template <int Q>
class RegisterQuadric : public OperatorNode {
  ADD_SLOT(Obstacles, obstacles, INPUT_OUTPUT, REQUIRED, DocString{"List of Obstacles"});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED);

  ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
  ADD_SLOT(onika::math::Mat4d, quadrics, INPUT, REQUIRED, DocString{"Define area."});
  ADD_SLOT(onika::math::Mat4d, transform, INPUT, OPTIONAL, DocString{"Define area."});

 public:
  inline std::string documentation() const override final {
    return R"EOF(
        This operator add a quadric to the obstacles list.

        YAML example:

        setup_obstacles:
          - register_quadrics:
              id: 0
              quadrics: cyly
              transform:
                - scale:     [ 0.05, 1,    0.05 ]
                - translate: [ 0.15, 0.1,  0.1  ]
          - register_quadrics:
              id: 1
              quadrics: sphere
              transform:
                - scale:     [ 0.05, 0.08, 0.05 ]
                - translate: [ 0.35, 0.1,  0.15 ]
          - register_quadrics:
              id: 2
              quadrics: conez
              transform:
                - scale:     [ 0.05, 0.05, 0.05 ]
                - translate: [ 0.55, 0.1,  0.25 ]
        )EOF";
  }

  inline void execute() override final {
    onika::math::Mat4d quadric = *quadrics;

    // transform the quadric if a transform is provided
    if (transform.has_value()) {
      const auto M_inv = onika::math::inverse(*transform);
      quadric = onika::math::transpose(M_inv) * quadric * M_inv;
    }

    // register it
    hippoLBM::Quadric obj(quadric);
    obstacles->add(*id, obj);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(register_quadrics) {
  OperatorNodeFactory::instance()->register_factory("register_quadrics", make_variant_operator<RegisterQuadric>);
}
}  // namespace hippoLBM
