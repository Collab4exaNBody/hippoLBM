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

#include <mpi.h>
#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/math/basic_types.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

// impl
#include <hippoLBM/bcs/dispatch_traversal.hpp>
#include <hippoLBM/bcs/rho.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;
using namespace bcs;

template <int Q>
class RhoOp : public OperatorNode {
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString{"The domain containing the grid and other simulation parameters."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(onika::math::Vec3d, U, INPUT, REQUIRED,
           DocString{"Prescribed velocity at the boundary, enforcing the rho condition."});
  ADD_SLOT(std::vector<double>, delta_p, INPUT, REQUIRED,
           DocString{"Pressure difference (Pa) relative to the fluid's reference state (avg_rho_, rho_lbm = 1), "
                     "prescribed at each boundary. Must have the same size as regions."});
  ADD_SLOT(std::vector<std::string>, regions, INPUT, REQUIRED,
           DocString{"Lists of grid regions to apply BCS. Ex: [plan_xy_0]"});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"The computed LBM parameters based on the input values."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
    This operator enforces a rho (density/pressure) boundary condition at x/y/z = 0 or l in
    an LBM simulation. delta_p is a pressure difference (Pa) relative to the fluid's
    reference state (avg_rho_, rho_lbm = 1), converted internally to rho_lbm using the same
    convention as set_dp_pressure.
        )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;
    LBMDomain<Q>& Domain = *domain;
    LBMGrid& Grid = Domain.grid();
    LBMParameters& params = *Params;

    // define functors
    auto [ux, uy, uz] = convert_velocity<LBM_UNITS>(*U, *Params);

    // get fields
    FieldView<Q> pf = data.distributions();
    int* const pobst = data.obstacles();

    // get traversal
    const std::vector<std::string> allowed_tr = {"plan_xy_0", "plan_xy_l", "plan_yz_0",
                                                 "plan_yz_l", "plan_xz_0", "plan_xz_l"};
    const std::vector<std::string>& region_names = *regions;
    const std::vector<double>& delta_ps = *delta_p;

    if (delta_ps.size() != region_names.size()) {
      lout << "[Error, bcs::rho] delta_p size (" << delta_ps.size() << ") does not match regions size ("
           << region_names.size() << ")" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    const std::vector<traversal_data> trs = get_traversal(traversals, region_names, allowed_tr);

    // run kernel
    for (size_t i = 0; i < trs.size(); ++i) {
      const auto& traversal = trs[i];
      const std::string& traversal_names = region_names[i];
      std::string kernel_name = "rho_on_" + traversal_names;

      if (traversal.size_ == 0) continue;  // No LBM point in this subdomain

      // Physical delta_p (Pa) -> rho_lbm, same convention as set_dp_pressure.
      const double real_delta_p = delta_ps[i];
      const double lbm_delta_p = real_delta_p * params.dtLB_ * params.dtLB_ / (params.avg_rho_ * Grid.dx_ * Grid.dx_);
      const double lbm_rho = 1.0 + 3.0 * lbm_delta_p;

      auto make_ctx = [this](const char* name) { return parallel_execution_context(name); };
      dispatch_traversal<Rho, Q>(make_ctx, traversal_from_string(traversal_names), kernel_name, traversal.ptr_,
                                 traversal.size_, pobst, pf, lbm_rho, ux, uy, uz);
    }

    // run kernel
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(rho) { OperatorNodeFactory::instance()->register_factory("rho", make_variant_operator<RhoOp>); }
}  // namespace hippoLBM
