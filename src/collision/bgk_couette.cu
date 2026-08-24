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

// onika
#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// onika types
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_yaml.h>

// hippoLBM

#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

// implementation files
#include <hippoLBM/collision/bgk.hpp>
#include <hippoLBM/collision/fext.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

template <int Q>
class CollisionBGKCouette : public OperatorNode {
 public:
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString{"Defines the computational domain and its properties for the LBM simulation."});
  ADD_SLOT(onika::math::Vec3d, U_inf, INPUT, REQUIRED,
           DocString{"Physical velocity used to derive the external force term at the lower boundary (k = 0)."});
  ADD_SLOT(
      onika::math::Vec3d, U_sup, INPUT, REQUIRED,
      DocString{
          "Physical velocity used to derive the external force term at the upper boundary (k = domain_size_z - 1)."});

  inline std::string documentation() const override final {
    return R"EOF(
    YAML example:

      - bgk_couette:
          U_inf: [0.0, 0.0, 0.0]
          U_sup: [0.1, 0.0, 0.0]
        )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;
    auto& params = *Params;
    LBMGrid& grid = domain->grid();
    int3d domain_size = domain->size();

    // get fields
    FieldView<3> pm1 = data.flux();
    int* const pobst = data.obstacles();
    FieldView<Q> pf = data.distributions();
    double* const pm0 = data.densities();

    // get traversal
    auto [ptr, size] = traversals.get_levels();
    // define functor
    const double Lz = domain_size[DIMZ] - 1;
    onika::math::Vec3d Uc_inf = convert_velocity<LBM_UNITS>(*U_inf, params);
    onika::math::Vec3d Uc_sup = convert_velocity<LBM_UNITS>(*U_sup, params);
    onika::math::Vec3d dU = (Uc_sup - Uc_inf) / Lz;
    FextCouetteFunc fext = {grid, Uc_inf, dU, 1. / params.tau_};
    bgk<Q, Traversal::Real, FextCouetteFunc> func = {ptr, fext, pm1, pobst, pf, pm0, params.tau_};
    // run kernel over the lbm grid
    parallel_for_simple(size, func, parallel_execution_context("bgk_couette"));
  }
};

using CollisionBGKCouette3D19Q = CollisionBGKCouette<19>;

// === register factories ===
ONIKA_AUTORUN_INIT(CollisionBGKCouette) {
  OperatorNodeFactory::instance()->register_factory("bgk_couette", make_variant_operator<CollisionBGKCouette>);
}
}  // namespace hippoLBM
