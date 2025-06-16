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

#include <grid/enum.hpp>
#include <grid/make_variant_operator.hpp>
#include <grid/lbm_domain.hpp>
#include <grid/lbm_fields.hpp>
#include <hippoLBM/obstacle/obstacles.hpp>


namespace hippoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		struct UpdateObstaclesFunc
		{
			grid<3> _grid;
			double _dx;
			int * const _obst;

			template<typename Obj>
				inline void operator()(Obj& obj) const
				{
					// convert bounds in box
					AABB bounds = obj.covered();
					Vec3d min = bounds.bmin;
					Vec3d max = bounds.bmax;
					point<3> _min = {int(min.x/_dx), int(min.y/_dx), int(min.z/_dx)};
					point<3> _max = {int(max.x/_dx), int(max.y/_dx), int(max.z/_dx)};
					box<3> global_box = {_min, _max};

					auto [is_inside_subdomain, local_box] = _grid.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);
					for(int z = local_box.start(2) ; z <= local_box.end(2) ; z++)
						for(int y = local_box.start(1) ; y <= local_box.end(1) ; y++)
							for(int x = local_box.start(0) ; x <= local_box.end(0) ; x++)
							{
								if( obj.solid( _grid.compute_position<Area::Global>(x,y,z) ))
								{
									const int idx = _grid(x,y,z);
									_obst[idx] = WALL_;
								}
							}

				}
		};

	template<int Q>
		class UpdateObstacles : public OperatorNode
	{

		ADD_SLOT( lbm_domain<Q>, LBMDomain, INPUT, REQUIRED);
		ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT);
		ADD_SLOT( Obstacles, obstacles, INPUT_OUTPUT, REQUIRED, DocString{"List of Obstacles"});
		ADD_SLOT( double, dx, INPUT, REQUIRED, DocString{"Space step"});


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
			auto& obs = *obstacles;
			lbm_fields<Q>& grid_data = *LBMFieds;
			lbm_domain<Q>& domain = *LBMDomain;

			UpdateObstaclesFunc<Q> func = { domain.m_grid, *dx, grid_data.obstacles() };

			for(size_t i = 0 ; i < obs.size() ; i++)
			{
				obs.apply(i, func);
			}
		}
	};

	// === register factories ===
	ONIKA_AUTORUN_INIT(project_obstacles) { OperatorNodeFactory::instance()->register_factory("update_obstacles", make_variant_operator<UpdateObstacles>); }
} // namespace exaDEM


