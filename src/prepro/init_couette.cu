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

// Implementation
#include <hippoLBM/prepro/couette.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

template <int Q>
class InitCouette : public OperatorNode {
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain containing the simulation data."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(onika::math::Vec3d, U_inf, INPUT, REQUIRED,
           DocString{"Prescribed velocity at the lower boundary (value = 0)."});
  ADD_SLOT(onika::math::Vec3d, U_sup, INPUT, REQUIRED,
           DocString{"Prescribed velocity at the upper boundary (value = domain_size - 1)."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
  ADD_SLOT(std::string, dimension, INPUT, REQUIRED, DocString{"Choose the dimension."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator initializes the distribution functions in the LBM grid for a Couette flow setup.
        The velocity profile is defined by a linear gradient between two boundaries, going from U_inf
        (at value = 0) to U_sup (at value = domain_size - 1).

        YAML example:

          - init_couette:
              dimension: "Z"
              U_inf: [0.0, 0.0, 0.0]
              U_sup: [0.1, 0.0, 0.0]
        )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& params = *Params;
    int3d domain_size = domain->size();
    LBMGrid& grid = domain->grid();

    // define variables
    onika::math::Vec3d Uc_inf = (*U_inf) / params.celerity_;
    onika::math::Vec3d Uc_sup = (*U_sup) / params.celerity_;

    // get fields
    FieldView<Q> pf = data.distributions();

    // get traversal
    Box3D real = grid.build_box<Area::Local, Traversal::Real>();
    onika::parallel::ParallelExecutionSpace<3> parallel_range = set(real);

    if (*dimension == "X") {
      // define variables
      onika::math::Vec3d dU = (Uc_sup - Uc_inf) / (domain_size[DIMX] - 1);
      // define functors
      InitCouetteFunc<Q, DIMX> func = {grid, pf, dU, Uc_inf};
      // run kernel
      parallel_for(parallel_range, func, parallel_execution_context("init_couette_dim_x"));
    } else if (*dimension == "Y") {
      // define variables
      onika::math::Vec3d dU = (Uc_sup - Uc_inf) / (domain_size[DIMY] - 1);
      // define functors
      InitCouetteFunc<Q, DIMY> func = {grid, pf, dU, Uc_inf};
      // run kernel
      parallel_for(parallel_range, func, parallel_execution_context("init_couette_dim_y"));
    } else if (*dimension == "Z") {
      lout << "Prepro couette starting ... dim Z" << std::endl;
      // define variables
      onika::math::Vec3d dU = (Uc_sup - Uc_inf) / (domain_size[DIMZ] - 1);
      lout << "Uc_inf: [" << Uc_inf << "]" << std::endl;
      lout << "Uc_sup: [" << Uc_sup << "]" << std::endl;
      lout << "dU: [" << dU << "]" << std::endl;
      // define functors
      InitCouetteFunc<Q, DIMZ> func = {grid, pf, dU, Uc_inf};
      // run kernel
      parallel_for(parallel_range, func, parallel_execution_context("init_couette_dim_z"));
      lout << "Prepro couette ending ... dim Z " << std::endl;
    } else {
      lout << "[init_couette] Please, select a valid dimension \"X\", \"Y\", or \"Z\"." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(init_couette) {
  OperatorNodeFactory::instance()->register_factory("init_couette", make_variant_operator<InitCouette>);
}
}  // namespace hippoLBM
