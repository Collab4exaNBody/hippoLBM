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
#include <hippoLBM/bcs/neumann.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;

template <int Q>
class NeumannOp : public OperatorNode {
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED,
           DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
  ADD_SLOT(LBMGridRegion, grid_region, INPUT, REQUIRED,
           DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
  ADD_SLOT(onika::math::Vec3d, U, INPUT, REQUIRED,
           DocString{"Prescribed velocity at the boundary, enforcing the Neumann condition."});
  ADD_SLOT(std::vector<std::string>, regions, INPUT, REQUIRED,
           DocString{"Lists of grid regions to apply BCS. Ex: [plan_xy_0]"});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"The computed LBM parameters based on the input values."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
    This operator enforces a Neumann boundary condition at z = 0 or z = l in an LBM simulation. 
    The Neumann boundary condition ensures that the gradient of the distribution function 
    follows a prescribed value.
        )EOF";
  }

  inline void execute() final {
    auto& data = *fields;
    auto& traversals = *grid_region;

    // define functors
    auto [ux, uy, uz] = convert_velocity<LBM_UNITS>(*U, *Params);

    // get fields
    FieldView<Q> pf = data.distributions();
    int* const pobst = data.obstacles();

    // get traversal
    const std::vector<std::string> allowed_tr = {"plan_xy_0", "plan_xy_l", "plan_yz_0",
                                                 "plan_yz_l", "plan_xz_0", "plan_xz_l"};
    const std::vector<std::string>& region_names = *regions;
    const std::vector<traversal_data> trs = get_traversal(traversals, region_names, allowed_tr);

    auto it = region_names.begin();
    // run kernel
    for (auto& traversal : trs) {
      const std::string traversal_names = *it++;
      std::string kernel_name = "neumann_on_" + traversal_names;

      if (traversal.size_ == 0) continue;  // No LBM point in this subdomain

      if (traversal_names == "plan_yz_0") {
        neumann_x_0<Q> neumann = {};
        parallel_for_id(traversal.ptr_, traversal.size_, neumann, parallel_execution_context(kernel_name.c_str()),
                        pobst, pf, ux, uy, uz);
      } else if (traversal_names == "plan_yz_l") {
        neumann_x_l<Q> neumann = {};
        parallel_for_id(traversal.ptr_, traversal.size_, neumann, parallel_execution_context(kernel_name.c_str()),
                        pobst, pf, ux, uy, uz);
      } else if (traversal_names == "plan_xz_0") {
        neumann_y_0<Q> neumann = {};
        parallel_for_id(traversal.ptr_, traversal.size_, neumann, parallel_execution_context(kernel_name.c_str()),
                        pobst, pf, ux, uy, uz);
      } else if (traversal_names == "plan_xz_l") {
        neumann_y_l<Q> neumann = {};
        parallel_for_id(traversal.ptr_, traversal.size_, neumann, parallel_execution_context(kernel_name.c_str()),
                        pobst, pf, ux, uy, uz);
      } else if (traversal_names == "plan_xy_0") {
        neumann_z_0<Q> neumann = {};
        parallel_for_id(traversal.ptr_, traversal.size_, neumann, parallel_execution_context(kernel_name.c_str()),
                        pobst, pf, ux, uy, uz);
      } else if (traversal_names == "plan_xy_l") {
        neumann_z_l<Q> neumann = {};
        parallel_for_id(traversal.ptr_, traversal.size_, neumann, parallel_execution_context(kernel_name.c_str()),
                        pobst, pf, ux, uy, uz);
      }
    }

    // run kernel
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(neumann) {
  OperatorNodeFactory::instance()->register_factory("neumann", make_variant_operator<NeumannOp>);
}
}  // namespace hippoLBM
