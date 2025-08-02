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

  class InitObstacles : public OperatorNode
  {

		ADD_SLOT( Obstacles, obstacles, OUTPUT, DocString{"List of Obstacles"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator .

        YAML example:

        )EOF";
		}

		inline void execute() override final
		{
      *obstacles = Obstacles(); 
		}
	};

	// === register factories ===
	ONIKA_AUTORUN_INIT(init_obstacles) { OperatorNodeFactory::instance()->register_factory("init_obstacles", make_simple_operator<InitObstacles>); }
} // namespace exaDEM


