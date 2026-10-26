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
#include <hippoLBM/grid/set_dp_pressure_gradient.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::math;
using namespace onika::parallel;

template <int Q>
class SetDPPressureGradientLBM : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString("The domain containing the grid and other simulation parameters."));
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString("The fields to be initialized, including the distribution function."));
  ADD_SLOT(LBMParameters, Params, INPUT, DocString{"The computed LBM parameters based on the input values."});

  ADD_SLOT(std::vector<double>, delta_p, INPUT, REQUIRED,
           DocString{"Pressure differences (Pa) relative to the fluid's reference state (avg_rho_, rho_lbm = 1): "
                     "[delta_p at bounds.bmin, delta_p at bounds.bmax] along dimension."});
  ADD_SLOT(onika::math::AABB, bounds, INPUT, REQUIRED,
           DocString{"Box where the pressure is set. The gradient goes from bmin to bmax along dimension."});
  ADD_SLOT(std::string, dimension, INPUT, REQUIRED, DocString{"Dimension of the gradient: \"X\", \"Y\" or \"Z\"."});

  ADD_SLOT(bool, verbosity, INPUT, false, DocString{"If true, print information about the operator execution."});

  inline std::string documentation() const final {
    return R"EOF(
    This operator initializes the pressure for a LBM simulation as a linear
    ramp of delta_p (difference relative to the fluid's reference state,
    rho_lbm = 1, i.e. avg_rho_) inside an axis-aligned bounding box. The ramp
    goes from delta_p[0] at bounds.bmin to delta_p[1] at bounds.bmax along
    the chosen dimension. Typically used to smooth a pressure discontinuity
    set with set_dp_pressure.

    YAML example:

      - set_dp_pressure_gradient:
          delta_p: [ 0.5e6, -0.5e6 ]
          bounds: [[ 0.009, 0.0, 0.0 ] , [ 0.011, 0.02, 0.015 ]]
          dimension: "X"
    )EOF";
  }

  inline void execute() final {
    if (delta_p->size() != 2) {
      lout << "[set_dp_pressure_gradient] delta_p must contain 2 values: [at bounds.bmin, at bounds.bmax]."
           << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    auto& data = *fields;
    LBMDomain<Q>& Domain = *domain;
    LBMGrid& Grid = Domain.grid();
    LBMParameters& params = *Params;
    const AABB& bound = *bounds;

    FieldView<Q> pf = data.distributions();

    // Isothermal LBM EOS (p = cs^2 * rho, cs^2 = 1/3), as a perturbation around rho_lbm = 1, see set_dp_pressure.
    const double to_lbm = params.dtLB_ * params.dtLB_ / (params.avg_rho_ * Grid.dx_ * Grid.dx_);
    const double rho_0 = 1.0 + 3.0 * (*delta_p)[0] * to_lbm;
    const double rho_1 = 1.0 + 3.0 * (*delta_p)[1] * to_lbm;

    if (*verbosity) {
      lout << "SetDPPressureGradientLBM: delta_p from " << (*delta_p)[0] << " Pa to " << (*delta_p)[1]
           << " Pa along " << *dimension << ", corresponding to rho_lbm from " << rho_0 << " to " << rho_1
           << std::endl;
    }

    // project_to_grid uses floor: bmax / dx can land just below an integer (e.g. 89.99999 instead of 90) and
    // drop the last layer, so bmax is shifted by a small fraction of dx.
    const double eps = 1e-6 * Grid.dx_;
    Point3D _min = Grid.project_to_grid<Area::Global>(bound.bmin);
    Point3D _max = Grid.project_to_grid<Area::Global>(bound.bmax + Vec3d{eps, eps, eps});
    Box3D global_box = {_min, _max};
    auto [is_inside_subdomain, box] = Grid.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);
    if (!is_inside_subdomain) return;

    if (*dimension == "X") {
      SetDPGradientFunc<Q, DIMX> func;
      parallel_for(box, func, parallel_execution_context("set_dp_pressure_gradient_x"), pf, Grid, bound.bmin.x,
                   bound.bmax.x, rho_0, rho_1);
    } else if (*dimension == "Y") {
      SetDPGradientFunc<Q, DIMY> func;
      parallel_for(box, func, parallel_execution_context("set_dp_pressure_gradient_y"), pf, Grid, bound.bmin.y,
                   bound.bmax.y, rho_0, rho_1);
    } else if (*dimension == "Z") {
      SetDPGradientFunc<Q, DIMZ> func;
      parallel_for(box, func, parallel_execution_context("set_dp_pressure_gradient_z"), pf, Grid, bound.bmin.z,
                   bound.bmax.z, rho_0, rho_1);
    } else {
      lout << "[set_dp_pressure_gradient] Please, select a valid dimension \"X\", \"Y\", or \"Z\"." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(set_dp_pressure_gradient) {
  OperatorNodeFactory::instance()->register_factory("set_dp_pressure_gradient",
                                                    make_variant_operator<SetDPPressureGradientLBM>);
}
}  // namespace hippoLBM
