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
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::math;

template <int Q>
class LBMParametersOp : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED,
           DocString{"The domain containing the grid and other simulation parameters."});
  ADD_SLOT(onika::math::Vec3d, Fext, INPUT, onika::math::Vec3d{0, 0, 0},
           DocString{"External forces applied to the fluid."});
  ADD_SLOT(double, celerity, INPUT, 1, DocString{"The speed of sound in the fluid."});
  ADD_SLOT(double, nuth, INPUT, 1e-4, DocString{"The dynamic viscosity of the fluid."});
  ADD_SLOT(double, avg_rho, INPUT, 1000.0, DocString{"The average density of the fluid."});
  ADD_SLOT(double, tau, INPUT, OPTIONAL, DocString{"Define tau [optional]"});

  ADD_SLOT(LBMParameters, Params, OUTPUT, DocString{"The computed LBM parameters based on the input values."});
  ADD_SLOT(double, dt, INPUT_OUTPUT, 0.0, DocString{"The time step for the LBM simulation."});

  inline std::string documentation() const final {
    return R"EOF(
    This operator computes the parameters required for the LBM simulation based on the input domain and physical properties.

    YAML example:

    - lbm_parameters:
       Fext: [0.0, 0.0, 0.0]   # External forces
       celerity: 1.0           # Speed of sound
       nuth: 1e-4              # Dynamic viscosity
       avg_rho: 1000.0         # Average density
    )EOF";
  }

  inline void execute() final {
    double Dx = domain->dx();
    LBMParameters params;

    params.nuth_ = *nuth;  // Physical Units
    params.avg_rho_ = *avg_rho;

    if (*dt > 0.0 && tau.has_value()) {
      lout << "[lbm_parameters]: You can't define tau AND dt" << std::endl;
    }

    if (*dt > 0.0) {
      lout << "LBM parameters are set according to the LBM timestep (" << *dt << ")." << std::endl;
      params.define_by_dt(*dt, Dx);  // define tau, nu, and c
    } else if (tau.has_value()) {
      lout << "LBM parameters are set according to tau (" << *tau << ")." << std::endl;
      params.define_by_tau(*tau, Dx);  // define nu, dtLB, and c
    } else {                           // default, use celerity (c), dtLB is deduced from c
      lout << "LBM parameters are set according to the celerity (" << *celerity << "), default." << std::endl;
      params.define_by_c(*celerity, Dx);
    }
    params.Fext_ = convert_force<LBM_UNITS>(*Fext, params);
    params.print();

    *dt = params.dtLB_;
    *Params = params;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(lbm_parameters) {
  OperatorNodeFactory::instance()->register_factory("lbm_parameters", make_variant_operator<LBMParametersOp>);
}
}  // namespace hippoLBM
