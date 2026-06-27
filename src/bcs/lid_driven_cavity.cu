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
#include <hippoLBM/bcs/lid_driven_cavity.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

template <int Q>
class LidDrivenCavityBCs : public OperatorNode {
  typedef std::array<double, 3> readVec3;
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED);
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(onika::math::Vec3d, U, INPUT, REQUIRED,
           DocString{"Prescribed velocity at the boundary (z = lz), enforcing the Cavity condition. U is the U real"});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});
  ADD_SLOT(std::vector<std::string>, regions, INPUT, REQUIRED, DocString{"Lists of grid regions to apply BCS."});

 public:
  inline std::string documentation() const final {
    return R"EOF( This operator enforces a Cavity boundary condition at z = lz in an LBM simulation. 
                      The Cavity boundary condition ensures that the gradient of the distribution function 
                      follows a prescribed value
        )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;

    // get fields
    auto [pex, pey, pez] = data.exyz();
    const double* const pw = data.weights();
    // ULBM = UReal / Celerity
    onika::math::Vec3d uLB = *U / Params->celerity_;  // Change to LBM World

    // define functors
    LidDrivenCavityBCsFunctor<Q> bcs = {uLB, data.distributions(), data.obstacles(), pex, pey, pez, pw};

    const std::vector<std::string>& region_names = *regions;
    const std::vector<traversal_data> trs = get_traversal(traversals, region_names);

    auto it = region_names.begin();
    // run kernel
    for (auto& traversal : trs) {
      const std::string traversal_names = *it++;
      std::string kernel_name = "cavity_apply_on_" + traversal_names;
      parallel_for_id(traversal.ptr_, traversal.size_, bcs, parallel_execution_context(kernel_name.c_str()));
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(lid_driven_cavity) {
  OperatorNodeFactory::instance()->register_factory("lid_driven_cavity", make_variant_operator<LidDrivenCavityBCs>);
}
}  // namespace hippoLBM
