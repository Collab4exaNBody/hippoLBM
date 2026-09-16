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
#include <onika/math/basic_types_yaml.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <algorithm>
#include <cmath>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/make_domain.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using onika::math::AABB;
using BoolVector = std::vector<bool>;

template <int Q>
class InitDomainLBM : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(LBMDomain<Q>, domain, OUTPUT, DocString{"The initialized LBM domain."});
  ADD_SLOT(BoolVector, periodic, INPUT_OUTPUT, REQUIRED, DocString{"Periodic boundary conditions for each dimension."});
  ADD_SLOT(onika::math::IJK, cell_dims, INPUT, REQUIRED,
           DocString{"Number of cells in each dimension. Grid dims: cells_dims+1."});
  ADD_SLOT(onika::math::AABB, bounds, INPUT, REQUIRED, DocString{"Domain's bounds"});
  ADD_SLOT(double, tolerance, INPUT, 1e-6,
           DocString{"Relative tolerance used to check consistency between resolution, grid size, and bounds."});
  ADD_SLOT(std::vector<std::string>, bounce_back_planes, INPUT, std::vector<std::string>{},
           DocString{"Named planes (plan_xy_0, plan_xy_l, plan_xz_0, plan_xz_l, plan_yz_0, plan_yz_l) where a "
                     "bounce-back wall boundary condition is applied. Each named plane's dimension must not be "
                     "periodic."});
  ADD_SLOT(bool, has_bounce_back, OUTPUT,
           DocString{"True if at least one bounce_back_plane is declared. Used as a condition to skip "
                     "pre_bounce_back/post_bounce_back entirely otherwise."});

  inline std::string documentation() const final {
    return R"EOF(
		This operator initializes the computational domain for ²the LBM simulation.

		Parameters:

		- cell_dims [IJK] : Number of cells in each dimension. Required. Grid dims: cell_dims+1.
		- bounds [AABB] : Domain's bounds (bmin/bmax). Required.
		- periodic [bool[3]] : Periodic boundary conditions for each dimension. Required.
		- tolerance [double] : Relative tolerance used to check consistency between resolution,
		  grid size, and bounds. Default: 1e-6.
		- bounce_back_planes [string list] : Named planes (plan_xy_0, plan_xy_l, plan_xz_0, plan_xz_l,
		  plan_yz_0, plan_yz_l) where a bounce-back wall boundary condition is applied. Each plane's
		  dimension must not be periodic. Default: none.

		YAML example:

		domain:
		   cell_dims: [100, 100, 100]
		   bounds:
			 bmin: [0.0, 0.0, 0.0]
			 bmax: [1.0, 1.0, 1.0]
		   periodic: [true, true, false]
		   bounce_back_planes: [plan_xy_0, plan_xy_l]
		   tolerance: 1e-6
		)EOF";
  }

  inline void execute() final {
    GridConfig grid;
    grid.periodic_ = convert<std::array<bool, 3>>(*periodic);
    grid.dims_.i = cell_dims->i + (grid.periodic_[0] ? 0 : 1);
    grid.dims_.j = cell_dims->j + (grid.periodic_[1] ? 0 : 1);
    grid.dims_.k = cell_dims->k + (grid.periodic_[2] ? 0 : 1);
    grid.bounds_ = *bounds;

    for (const std::string& name : *bounce_back_planes) {
      const int plane_idx = bounce_back_plane_index(name);
      const int dim = bounce_back_plane_dim(plane_idx);
      if (grid.periodic_[dim]) {
        lout << "[Error, domain], bounce_back_planes: \"" << name
             << "\" is on a periodic dimension. A dimension can't be both periodic and have a bounce-back plane."
             << std::endl;
        std::exit(EXIT_FAILURE);
      }
      grid.bounce_back_planes_[plane_idx] = true;
    }
    *has_bounce_back = !bounce_back_planes->empty();

    onika::math::IJK grid_size = grid.dims_;
    auto [inf, sup] = grid.bounds_;

    auto nb_intervals = [&](int dim, ssize_t n) { return grid.periodic_[dim] ? n : n - 1; };

    onika::math::Vec3d resolution_dims;
    resolution_dims.x = (sup.x - inf.x) / double(nb_intervals(0, grid_size.i));
    resolution_dims.y = (sup.y - inf.y) / double(nb_intervals(1, grid_size.j));
    resolution_dims.z = (sup.z - inf.z) / double(nb_intervals(2, grid_size.k));

    // check
    const double tol = *tolerance;
    bool check_grid_size = false;
    if (!equal_rel_tol(resolution_dims.x, resolution_dims.y, tol) ||
        !equal_rel_tol(resolution_dims.x, resolution_dims.z, tol)) {
      lout << "[Error, domain], Dx is not the same for all dimension" << std::endl;
      lout << "Dx: [ " << resolution_dims << " ] " << std::endl;
      std::exit(EXIT_FAILURE);
    }

    double reso = resolution_dims.x;

    if (!equal_rel_tol(inf.x + nb_intervals(0, grid_size.i) * reso, sup.x, tol)) {
      check_grid_size = true;
    }
    if (!equal_rel_tol(inf.y + nb_intervals(1, grid_size.j) * reso, sup.y, tol)) {
      check_grid_size = true;
    }
    if (!equal_rel_tol(inf.z + nb_intervals(2, grid_size.k) * reso, sup.z, tol)) {
      check_grid_size = true;
    }
    if (check_grid_size) {
      lout << "[Error, domain], The resolution slot and bounds slot mismatch." << std::endl;
      lout << "Bound inf:  " << inf << std::endl;
      lout << "Bound sup:  " << sup << std::endl;
      lout << "Grid size:  " << grid_size << std::endl;
      lout << "Resolution: " << reso << std::endl;
      std::exit(EXIT_FAILURE);
    }

    SubGridConfig sub_grid = load_balancing(grid, *mpi);
    *domain = make_domain<Q>(grid, sub_grid);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(init_domain) {
  OperatorNodeFactory::instance()->register_factory("domain", make_variant_operator<InitDomainLBM>);
}
}  // namespace hippoLBM
