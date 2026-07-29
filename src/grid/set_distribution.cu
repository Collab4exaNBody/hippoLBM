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

#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/math/basic_types.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/set_distribution.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::math;
using namespace onika::parallel;

template <int Q>
class SetDistributionsLBM : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString("The domain containing the grid and other simulation parameters."));
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString("The fields to be initialized, including the distribution function."));

  ADD_SLOT(double, value, INPUT, double(1), DocString{"The value to initialize the distribution function with."});
  ADD_SLOT(bool, do_update, INPUT, false, DocString{"Whether to update ghost cells after initialization."});
  ADD_SLOT(onika::math::AABB, bounds, INPUT, OPTIONAL, DocString{"Domain's bounds"});
  ADD_SLOT(onika::math::Mat4d, quadrics, INPUT, OPTIONAL, DocString{"Define area."});
  ADD_SLOT(onika::math::Mat4d, transform, INPUT, OPTIONAL, DocString{"Define area."});

  inline std::string documentation() const final {
    return R"EOF(
    This operator initializes the distribution function for the LBM simulation
    at its equilibrium value. It can be applied to the whole grid, restricted
    to an axis-aligned bounding box, or restricted to a quadric-defined region.

    YAML example:

      - set_distribution:
          value: 1.0

      - set_distribution:
          value: 1.0
          do_update: true
          bounds: [[ 0.0, 0.0, 0.0 ] , [ 1.0, 1.0, 1.0 ]]

      - set_distribution:
          value: 1.0
          quadrics: sphere
          transform:
            - scale:     [ 0.05, 0.08, 0.05 ]
            - translate: [ 0.35, 0.1,  0.15 ]
    )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    LBMDomain<Q>& Domain = *domain;
    LBMGrid& Grid = Domain.grid();

    FieldView<Q> pf = data.distributions();

    // Define kernel
    SetDistributionFunc<Q> func;

    // Define Box
    bool use_bound = bounds.has_value();
    bool use_quadric = quadrics.has_value();
    [[maybe_unused]] hippoLBM::Box3D wall_box;
    [[maybe_unused]] bool is_inside_subdomain = true;
    [[maybe_unused]] onika::math::Mat4d quadric;

    if (*do_update) {
      wall_box = Grid.build_box<Area::Local, Traversal::Real>();
    } else {
      wall_box = Grid.build_box<Area::Local, Traversal::All>();
    }

    if (use_bound) {
      auto& bound = *bounds;
      onika::math::Vec3d min = bound.bmin;
      onika::math::Vec3d max = bound.bmax;
      double Dx = Grid.dx_;
      Point3D _min = {int(min.x / Dx), int(min.y / Dx), int(min.z / Dx)};
      Point3D _max = {int(max.x / Dx), int(max.y / Dx), int(max.z / Dx)};

      Box3D global_wall_box = {_min, _max};

      std::tie(is_inside_subdomain, wall_box) =
          Grid.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_wall_box);
      if (!is_inside_subdomain) return;
    }

    if (use_quadric) {
      quadric = *quadrics;
      // transform the quadric if a transform is provided
      if (transform.has_value()) {
        const auto M_inv = onika::math::inverse(*transform);
        quadric = onika::math::transpose(M_inv) * quadric * M_inv;
      }
      parallel_for(wall_box, func, parallel_execution_context("wall_box"), pf, *value, Grid, quadric);
    } else {
      parallel_for(wall_box, func, parallel_execution_context("wall_box"), pf, *value, Grid);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(set_distributions) {
  OperatorNodeFactory::instance()->register_factory("set_distribution", make_variant_operator<SetDistributionsLBM>);
}
}  // namespace hippoLBM
