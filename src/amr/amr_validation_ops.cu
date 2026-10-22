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

// Standalone operators usable in a *plain* (non-AMR) simulation: they let a quantitative
// reference run (uniform grid at the fine resolution, no refinement) use the exact same
// initial condition and print profiles in the exact same format as AMRTwoLevel's internal
// diagnostics, so the two can be compared numerically. See docs/AMR_PLAN.md ("Validation
// quantitative (convergence)").

#include <mpi.h>

#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// NOTE: must come after <onika/parallel/parallel_for.h>, which declares the primary
// ParallelForFunctorTraits template that amr_test_init.hpp specializes -- including it first
// made the compiler treat that specialization as a brand new (wrongly-shaped) primary
// template, breaking every other specialization in this translation unit.
#include <hippoLBM/amr/amr_test_diagnostics.hpp>
#include <hippoLBM/amr/amr_test_init.hpp>
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
class AMRInitGaussianBump : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"The fields to initialize."});
  ADD_SLOT(onika::math::Vec3d, center, INPUT, REQUIRED, DocString{"Physical center of the density bump."});
  ADD_SLOT(double, sigma, INPUT, REQUIRED, DocString{"Standard deviation of the density bump."});
  ADD_SLOT(double, amplitude, INPUT, REQUIRED, DocString{"Amplitude of the density bump (rho = 1 + amplitude at the center)."});

  inline std::string documentation() const final {
    return R"EOF(
    Initializes the distribution functions to equilibrium (u=0) at a localized Gaussian
    density bump: rho(x) = 1 + amplitude * exp(-|x - center|^2 / (2*sigma^2)). Used to give a
    plain (non-AMR) reference run the exact same initial condition as an amr_two_level run
    with the same perturb_center/perturb_sigma/perturb_amplitude, for quantitative comparison
    (docs/AMR_PLAN.md).

    YAML example:

      - amr_init_gaussian_bump:
         center: [0.5, 0.5, 0.5]
         sigma: 0.05
         amplitude: 0.1
    )EOF";
  }

  inline void execute() final {
    LBMGrid& grid = domain->grid();
    Box3D all_box = grid.template build_box<Area::Local, Traversal::All>();
    FieldView<Q> pf = fields->distributions();
    GaussianDensityInit<Q> func = {*center, *sigma, *amplitude};
    parallel_for(all_box, func, parallel_execution_context(), pf, grid);
    ONIKA_CU_DEVICE_SYNCHRONIZE();
  }
};

template <int Q>
class AMRPrintLineProfile : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT, REQUIRED, DocString{"The fields to read from."});
  ADD_SLOT(onika::math::Vec3d, point, INPUT, REQUIRED,
           DocString{"Physical point: the profile is printed along x, at the y,z nearest to this point."});
  ADD_SLOT(std::string, tag, INPUT, "ref", DocString{"Label prefixed to the printed line, for identification."});

  inline std::string documentation() const final {
    return R"EOF(
    Prints rho along the x axis (at the y,z nearest to `point`) over this domain's Real
    nodes, in the exact same text format as amr_two_level's internal verbosity diagnostics --
    so the two can be compared directly (docs/AMR_PLAN.md: quantitative convergence
    validation against a plain, uniform-fine reference run).

    YAML example:

      - amr_print_line_profile:
         point: [0.5, 0.5, 0.5]
         tag: "ref"
    )EOF";
  }

  inline void execute() final {
    LBMGrid& grid = domain->grid();
    FieldView<Q> pf = fields->distributions();
    print_rho_line_profile<Q>(tag->c_str(), grid, pf, *point);
  }
};

template <int Q>
class AMRPrintUxProfileZ : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT, REQUIRED, DocString{"The fields to read from."});
  ADD_SLOT(onika::math::Vec3d, point, INPUT, REQUIRED,
           DocString{"Physical point: the profile is printed along z, at the x,y nearest to this point."});
  ADD_SLOT(std::string, tag, INPUT, "ref", DocString{"Label prefixed to the printed line, for identification."});

  inline std::string documentation() const final {
    return R"EOF(
    Prints u_x along the z axis (at the x,y nearest to `point`) over this domain's Real
    nodes, in the exact same text format as amr_two_level's internal verbosity diagnostics
    (when apply_couette_bc: true) -- so the two can be compared directly against the
    analytical Couette profile u_x(z) = U_bottom + (U_top - U_bottom) * z / L
    (docs/AMR_PLAN.md: quantitative convergence validation).

    YAML example:

      - amr_print_ux_profile_z:
         point: [0.5, 0.5, 0.5]
         tag: "ref"
    )EOF";
  }

  inline void execute() final {
    LBMGrid& grid = domain->grid();
    FieldView<Q> pf = fields->distributions();
    print_ux_profile_z<Q>(tag->c_str(), grid, pf, *point);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(amr_init_gaussian_bump) {
  OperatorNodeFactory::instance()->register_factory("amr_init_gaussian_bump", make_variant_operator<AMRInitGaussianBump>);
}
ONIKA_AUTORUN_INIT(amr_print_line_profile) {
  OperatorNodeFactory::instance()->register_factory("amr_print_line_profile", make_variant_operator<AMRPrintLineProfile>);
}
ONIKA_AUTORUN_INIT(amr_print_ux_profile_z) {
  OperatorNodeFactory::instance()->register_factory("amr_print_ux_profile_z", make_variant_operator<AMRPrintUxProfileZ>);
}
}  // namespace hippoLBM
