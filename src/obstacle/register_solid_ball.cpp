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
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <hippoLBM/obstacle/obstacles.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  class RegisterSolidBall : public OperatorNode
  {
    ADD_SLOT(Obstacles, obstacles, INPUT_OUTPUT, REQUIRED, DocString{"List of Obstacles"});
    ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
    ADD_SLOT(Vec3d, center, INPUT, REQUIRED, DocString{"Center of the ball."});
    ADD_SLOT(double, radius, INPUT, REQUIRED, DocString{"Radius of the ball."});

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator add a ball to the obstacles list.
 
        YAML example:
 
        setup_obstacles:
          - register_solid_ball:
             center: [1,2,3]
             radius: 1
        )EOF";
    }

    inline void execute() override final
    {
      hippoLBM::Ball obj(*center, *radius);
      obstacles->add(*id, obj);
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(register_solid_ball) { OperatorNodeFactory::instance()->register_factory("register_solid_ball", make_simple_operator<RegisterSolidBall>); }
} // namespace exaDEM


