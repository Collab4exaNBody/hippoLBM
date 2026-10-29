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
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_yaml.h>
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

// Implementation file
#include <hippoLBM/collision/mrt.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

template <int Q>
class CollisionMRT : public OperatorNode {
 public:
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
  ADD_SLOT(double, s_e, INPUT, MRTRates{}.s_e, DocString{"Relaxation rate of the energy moment."});
  ADD_SLOT(double, s_eps, INPUT, MRTRates{}.s_eps, DocString{"Relaxation rate of the energy square moment."});
  ADD_SLOT(double, s_q, INPUT, MRTRates{}.s_q, DocString{"Relaxation rate of the energy flux moments."});
  ADD_SLOT(double, s_pi, INPUT, MRTRates{}.s_pi, DocString{"Relaxation rate of the stress square moments."});
  ADD_SLOT(double, s_m, INPUT, MRTRates{}.s_m, DocString{"Relaxation rate of the third order moments."});

  inline std::string documentation() const final {
    return R"EOF( The `CollisionMRT` operator implements the MRT collision model for the Lattice Boltzmann Method (LBM).
    Moments of the distribution functions relax towards the moments of the BGK equilibrium. The stress moments relax with
    1/tau (so the viscosity is the same as BGK), the others with the rates s_e, s_eps, s_q, s_pi and s_m
    (default: D'Humieres et al. 2002). With all rates equal to 1/tau, the operator is equivalent to BGK.

    YAML example:

      - mrt:
          s_e: 1.19
          s_eps: 1.4
    )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;
    auto& params = *Params;

    // define functor
    const MRTRates rates = {*s_e, *s_eps, *s_q, *s_pi, *s_m};
    mrt<Q> func = {params.Fext_, rates};

    // get fields
    int* const pobst = data.obstacles();
    FieldView<Q> pf = data.distributions();
    double* const pm0 = data.densities();

    // get traversal
    auto [ptr, size] = traversals.get_data<Traversal::Real>();

    // run kernel
    parallel_for_id(ptr, size, func, parallel_execution_context(), pobst, pf, pm0, params.tau_);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(CollisionMRT) {
  OperatorNodeFactory::instance()->register_factory("mrt", make_variant_operator<CollisionMRT>);
}
}  // namespace hippoLBM
