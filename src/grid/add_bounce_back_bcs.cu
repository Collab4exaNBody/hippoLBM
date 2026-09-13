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

#include <onika/log.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// HippoLBM
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
class AddBounceBackBCs : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT_OUTPUT, REQUIRED,
           DocString{"The already-initialized LBM domain to update (e.g. after a read_dump restart)."});
  ADD_SLOT(std::vector<std::string>, bounce_back_planes, INPUT, std::vector<std::string>{},
           DocString{"Named planes (plan_xy_0, plan_xy_l, plan_xz_0, plan_xz_l, plan_yz_0, plan_yz_l) where a "
                     "bounce-back wall boundary condition is applied. Each named plane's dimension must not be "
                     "periodic."});
  ADD_SLOT(bool, has_bounce_back, INPUT_OUTPUT, REQUIRED,
           DocString{"True if at least one bounce_back_plane is declared. Used as a condition to skip "
                     "pre_bounce_back/post_bounce_back entirely otherwise."});

  inline std::string documentation() const final {
    return R"EOF(
    This operator (re)configures the bounce-back manager of an already-initialized domain, without
    rebuilding the domain itself.

    YAML example:

      - add_bounce_back_bcs:
         bounce_back_planes: [plan_xy_0, plan_xy_l]
    )EOF";
  }

  inline void execute() final {
    std::array<bool, 2 * DIM_MAX> active_planes{};
    for (const std::string& name : *bounce_back_planes) {
      const int plane_idx = bounce_back_plane_index(name);
      const int dim = bounce_back_plane_dim(plane_idx);
      if (domain->periodic_[dim]) {
        lout << "[Error, add_bounce_back_bcs], bounce_back_planes: \"" << name
             << "\" is on a periodic dimension. A dimension can't be both periodic and have a bounce-back plane."
             << std::endl;
        std::exit(EXIT_FAILURE);
      }
      active_planes[plane_idx] = true;
    }
    *has_bounce_back = !bounce_back_planes->empty();

    auto local_all_box = domain->grid().template build_box<Area::Local, Traversal::All>();
    onika::math::IJK local_grid_size(local_all_box.get_length(0), local_all_box.get_length(1),
                                     local_all_box.get_length(2));
    domain->bb_manager().resize_data(active_planes, local_grid_size, domain->MPI_coord_, domain->MPI_grid_size_);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(add_bounce_back_bcs) {
  OperatorNodeFactory::instance()->register_factory("add_bounce_back_bcs", make_variant_operator<AddBounceBackBCs>);
}
}  // namespace hippoLBM
