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
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/set_distribution.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::math;
using namespace onika::parallel;

template <int Q>
class SetDPPressureLBM : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString("The domain containing the grid and other simulation parameters."));
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString("The fields to be initialized, including the distribution function."));
  ADD_SLOT(LBMParameters, Params, INPUT, DocString{"The computed LBM parameters based on the input values."});

  ADD_SLOT(double, delta_p, INPUT, REQUIRED,
           DocString{"Pressure difference (Pa) relative to the fluid's reference state (avg_rho_, rho_lbm = 1)."});
  ADD_SLOT(onika::math::AABB, bounds, INPUT, OPTIONAL, DocString{"Domain's bounds"});
  ADD_SLOT(onika::math::Mat4d, quadrics, INPUT, OPTIONAL, DocString{"Define area."});
  ADD_SLOT(onika::math::Mat4d, transform, INPUT, OPTIONAL, DocString{"Define area."});

  ADD_SLOT(bool, verbosity, INPUT, false, DocString{"If true, print information about the operator execution."});

  inline std::string documentation() const final {
    return R"EOF(
    This operator initializes the pressure for a LBM simulation as a difference
    (delta_p) relative to the fluid's reference state (rho_lbm = 1, i.e. avg_rho_).
    Unlike set_pressure, delta_p = 0 always leaves the fluid unperturbed,
    regardless of dtLB. It can be applied to the whole grid, restricted to an
    axis-aligned bounding box, or restricted to a quadric-defined region.

    YAML example:

      - set_dp_pressure:
          delta_p: 80e6

      - set_dp_pressure:
          delta_p: 10e6
          bounds: [[ 0.0, 0.0, 0.0 ] , [ 1.0, 1.0, 1.0 ]]

      - set_dp_pressure:
          delta_p: 10e6
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
    LBMParameters& params = *Params;

    FieldView<Q> pf = data.distributions();

    double real_delta_p = *delta_p;
    double lbm_delta_p = real_delta_p * params.dtLB_ * params.dtLB_ / (params.avg_rho_ * Grid.dx_ * Grid.dx_);
    // Isothermal LBM EOS (p = cs^2 * rho, cs^2 = 1/3), as a perturbation around the
    // reference density rho_lbm = 1: delta_p = 0 => rho_lbm = 1 (unperturbed fluid).
    double rho_lbm = 1.0 + 3.0 * lbm_delta_p;

    if (*verbosity) {
      lout << "SetDPPressureLBM: Setting delta_p to " << real_delta_p << " Pa and lbm_delta_p = " << lbm_delta_p
           << ", corresponding to rho_lbm = " << rho_lbm << std::endl;
    }

    // Define kernel
    SetDistributionFunc<Q> func;

    // Define Box
    bool use_bound = bounds.has_value();
    bool use_quadric = quadrics.has_value();
    [[maybe_unused]] hippoLBM::Box3D wall_box;
    [[maybe_unused]] bool is_inside_subdomain = true;
    [[maybe_unused]] onika::math::Mat4d quadric;

    wall_box = Grid.build_box<Area::Local, Traversal::All>();

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
      parallel_for(wall_box, func, parallel_execution_context("set_dp_pressure"), pf, rho_lbm, Grid, quadric);
    } else {
      parallel_for(wall_box, func, parallel_execution_context("set_dp_pressure"), pf, rho_lbm, Grid);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(set_dp_pressures) {
  OperatorNodeFactory::instance()->register_factory("set_dp_pressure", make_variant_operator<SetDPPressureLBM>);
}
}  // namespace hippoLBM
