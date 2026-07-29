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
#include <hippoLBM/obstacle/update_obstacles.hpp>

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
