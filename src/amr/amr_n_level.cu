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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

// hippoLBM
#include <hippoLBM/bcs/dispatch_traversal.hpp>
#include <hippoLBM/bcs/neumann.hpp>
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/init_obst.hpp>
#include <hippoLBM/grid/make_domain.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/set_distribution.hpp>
#include <hippoLBM/grid/update_ghost.hpp>
#include <hippoLBM/obstacle/bounce_back_obstacles.hpp>
#include <hippoLBM/obstacle/obstacle.hpp>
#include <hippoLBM/obstacle/update_obstacles.hpp>

// implementation files
#include <hippoLBM/amr/amr_coupling.hpp>
#include <hippoLBM/amr/amr_level.hpp>
#include <hippoLBM/amr/amr_refinement_indicator.hpp>
#include <hippoLBM/amr/amr_test_diagnostics.hpp>
#include <hippoLBM/amr/amr_test_init.hpp>
#include <hippoLBM/collision/bgk.hpp>
#include <hippoLBM/collision/fext.hpp>
#include <hippoLBM/collision/macro_variables.hpp>
#include <hippoLBM/collision/streaming.hpp>
#include <hippoLBM/io/write_paraview.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;
using namespace onika::cuda;
using namespace bcs;
using BoolVector = std::vector<bool>;
using AABBVector = std::vector<onika::math::AABB>;

/** @brief AMR Phase 3 (docs/AMR_PLAN.md): a stack of N recursively-refined levels (ratio 2:1
 * between each level and the next), single rank. Generalizes the Phase 1/2 two-level operator
 * (renamed from amr_two_level): level 0 is the coarsest (the physical domain), and each entry
 * in `patch_bounds_list` adds one more level refining the previous one -- an empty list
 * reduces to a single, unrefined level (the old "enable_fine_patch: false" case).
 *
 * Timestepping is the classic recursive Berger-Colella advance: to advance level L one of its
 * own timesteps, step L itself, then (if it has a child level L+1) couple and recursively
 * advance L+1 `ratio_` times, coupling C2F before each sub-step and F2C once after all of
 * them -- see advance() below. For exactly one level this reduces to a plain single-level
 * step; for two levels it reduces to exactly the Phase 2 sub-cycled sequence this operator
 * used to hardcode. See amr_coupling.hpp for the C2F/F2C scheme between any adjacent pair. */
template <int Q>
class AMRNLevel : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator. Single rank only."});

  // level 0 (coarsest, the physical domain)
  ADD_SLOT(onika::math::IJK, cell_dims, INPUT, REQUIRED,
           DocString{"Level 0: number of cells in each dimension. Grid dims: cell_dims+1."});
  ADD_SLOT(onika::math::AABB, bounds, INPUT, REQUIRED, DocString{"Level 0: domain's bounds."});
  ADD_SLOT(BoolVector, periodic, INPUT, REQUIRED,
           DocString{"Level 0: periodic boundary conditions for each dimension."});

  // refinement hierarchy (ratio 2:1 fixed, as in Phase 1-2): patch_bounds_list[i] refines
  // level i into level i+1, and must nest inside (and 2:1-align with) level i's own grid --
  // which, for i>0, is itself patch_bounds_list[i-1]'s grid, not level 0's. An empty list (the
  // default) means a single, unrefined level.
  ADD_SLOT(AABBVector, patch_bounds_list, INPUT, AABBVector{},
           DocString{"Ordered list of physical sub-boxes to refine, each nested inside and 2:1-aligned with the "
                     "previous level's grid. patch_bounds_list[i] turns level i into level i+1's parent. Empty "
                     "(the default) means no refinement at all: level 0 alone."});

  // physical fluid parameters, shared by every level (same celerity => convective scaling,
  // see docs/AMR_PLAN.md §4: this alone reproduces the Filippova-Haenel tau rescaling at each
  // successive level).
  ADD_SLOT(double, celerity, INPUT, 1.0, DocString{"Shared network celerity (same at every level)."});
  ADD_SLOT(double, nuth, INPUT, 1e-4, DocString{"Physical dynamic viscosity, shared by every level."});
  ADD_SLOT(double, avg_rho, INPUT, 1000.0, DocString{"Physical average density, shared by every level."});
  ADD_SLOT(onika::math::Vec3d, Fext, INPUT, (onika::math::Vec3d{0, 0, 0}),
           DocString{"Physical external force, shared by every level."});

  ADD_SLOT(bool, verbosity, INPUT, false,
           DocString{"If true, print per-step total density and max |u| for every level (debug/validation)."});

  // Validation-only initial condition (docs/AMR_PLAN.md Phase 1 status): a uniform field
  // (amplitude 0, the default) cannot exercise the C2F/F2C interpolation at all.
  ADD_SLOT(double, perturb_amplitude, INPUT, 0.0,
           DocString{"Amplitude of an initial Gaussian density bump (0 = uniform rho=1, u=0). Validation only."});
  ADD_SLOT(onika::math::Vec3d, perturb_center, INPUT, (onika::math::Vec3d{0.5, 0.5, 0.5}),
           DocString{"Physical center of the initial density bump."});
  ADD_SLOT(double, perturb_sigma, INPUT, 0.05, DocString{"Standard deviation of the initial density bump."});

  // Couette validation (docs/AMR_PLAN.md: quantitative convergence against a known analytical
  // profile, u_x(z) = U_bottom + (U_top - U_bottom) * z / L, rather than an acoustic pulse
  // whose transient position is hard to compare precisely). Requires periodic: [true, true,
  // false] (walls in z, like example/lbm_couette.msp).
  ADD_SLOT(bool, apply_couette_bc, INPUT, false,
           DocString{"If true, apply a Neumann (prescribed velocity) BC at the z=0/z=L planes of level 0 (the only "
                     "level with a genuine physical boundary), after each streaming step."});
  ADD_SLOT(onika::math::Vec3d, wall_velocity_bottom, INPUT, (onika::math::Vec3d{0, 0, 0}),
           DocString{"Physical velocity imposed at the z=0 plane (plan_xy_0)."});
  ADD_SLOT(onika::math::Vec3d, wall_velocity_top, INPUT, (onika::math::Vec3d{0, 0, 0}),
           DocString{"Physical velocity imposed at the z=L plane (plan_xy_l)."});

  // Obstacle-in-patch validation (docs/AMR_PLAN.md): a single Ball, stamped at the same
  // physical position/radius on every level, closer to the project's actual target use case
  // (refine near an obstacle) than the synthetic Couette/Gaussian-bump cases above. Reuses
  // the production Ball/SetObstacleFunc/wall_bounce_back machinery from the obstacle plugin
  // directly (not through the Obstacles-container operator pipeline, which this
  // self-contained operator doesn't participate in).
  ADD_SLOT(bool, enable_obstacle, INPUT, false,
           DocString{"If true, stamp a single Ball obstacle (obstacle_center/obstacle_radius) into every level "
                     "and apply wall_bounce_back at its surface every step."});
  ADD_SLOT(onika::math::Vec3d, obstacle_center, INPUT, (onika::math::Vec3d{0.5, 0.5, 0.5}),
           DocString{"Physical center of the Ball obstacle."});
  ADD_SLOT(double, obstacle_radius, INPUT, 0.05, DocString{"Physical radius of the Ball obstacle."});
  ADD_SLOT(onika::math::Vec3d, obstacle_velocity, INPUT, (onika::math::Vec3d{0, 0, 0}),
           DocString{"Physical velocity of the obstacle's own center (displacement per coarse/level-0 step, not a "
                     "true physical dt conversion -- a validation knob, see docs/AMR_PLAN.md Phase 6 sub-problem "
                     "2). {0,0,0} (the default) is a static obstacle: identical cost/behavior to before this slot "
                     "existed, since update_moving_obstacle() early-exits on zero velocity. Does NOT handle the "
                     "obstacle crossing a periodic boundary (Ball::solid() is plain Euclidean distance, not "
                     "periodic-aware) -- keep its path inside the domain's interior."});

  // Flow-driven refinement indicator (docs/AMR_PLAN.md Phase 6, sub-problem 1).
  ADD_SLOT(bool, print_q_criterion, INPUT, false,
           DocString{"If true (with verbosity), also print the Q-criterion (amr_refinement_indicator.hpp) along "
                     "the z-axis through perturb_center, for every level."});

  // Dynamic re-centering (docs/AMR_PLAN.md Phase 6, sub-problem 2): a leaf (childless) patch
  // can MOVE, tracking the peak |Q| in its parent, at a fixed cadence -- patch SIZE and the
  // number of levels stay fixed (no patches appear/disappear, no load rebalancing: both
  // explicitly out of scope for this first increment).
  ADD_SLOT(long, remesh_freq, INPUT, -1,
           DocString{"Phase 6 sub-problem 2: every N coarse (level 0) steps, recenter each leaf patch on the "
                     "peak |Q-criterion| found in its parent's own grid, within remesh_search_margin cells of its "
                     "current position. <=0 disables remeshing (static AMR, the default)."});
  ADD_SLOT(long, remesh_search_margin, INPUT, 2,
           DocString{"Remeshing: half-width, in the PARENT level's own cells, of the neighborhood around a leaf "
                     "patch's current center that's searched for the new peak-|Q| location. Bounds how far a "
                     "patch can move per remesh event (a smoothness/tracking assumption, not a hard limit from "
                     "the coupling scheme itself)."});
  ADD_SLOT(double, remesh_q_threshold, INPUT, 1.0e-3,
           DocString{"Remeshing: a patch only moves if the peak |Q-criterion| found in the search window is above "
                     "this value; otherwise there's no genuine flow feature to track and the patch stays put. "
                     "Needed because Q is never exactly zero in the quiescent background (floating-point/LBM "
                     "discretization noise, ~1e-5 for the Couette validation case) -- without this floor, the "
                     "|Q|-weighted centroid still has a well-defined (nonzero) value even with no real feature "
                     "present, and the patch slowly drifts chasing that noise instead of staying still (observed "
                     "on a no-obstacle Couette run before this threshold was added: the patch wandered from the "
                     "domain center toward z=0.75 over 200 iterations). Our obstacle signal is ~5e-3 to ~5e-2, so "
                     "the default sits comfortably between the two."});

  // Paraview output: writes every level as its own .pvtr (reusing write_paraview() from the
  // io plugin, unmodified) plus a small .vtm multi-block manifest wrapping all of them, so
  // ParaView opens one file and shows every level together, each at its own real resolution.
  ADD_SLOT(long, timestep, INPUT, 0, DocString{"Current timestep, used to number the Paraview output files."});
  ADD_SLOT(long, paraview_freq, INPUT, -1,
           DocString{"Write Paraview output every N timesteps (<=0 disables it)."});
  ADD_SLOT(std::string, output_directory, INPUT, "AMROutputDir",
           DocString{"Base directory for the AMR Paraview output."});

  inline std::string documentation() const final {
    return R"EOF(
    AMR Phase 3 (docs/AMR_PLAN.md): a stack of N recursively 2:1-refined levels, single rank.
    Each call to this operator runs exactly one timestep of level 0, recursively sub-cycling
    every finer level `ratio_` (=2) times per parent step, coupled by C2F before each fine
    sub-step and F2C once after all of them (per level transition).

    YAML example (3 levels: a coarse domain, one patch, and a patch nested inside that patch):

      - amr_n_level:
         cell_dims: [64, 64, 64]
         bounds: { bmin: [0,0,0], bmax: [1,1,1] }
         periodic: [true, true, true]
         patch_bounds_list:
           - { bmin: [0.375,0.375,0.375], bmax: [0.625,0.625,0.625] }
           - { bmin: [0.4375,0.4375,0.4375], bmax: [0.5625,0.5625,0.5625] }
         celerity: 1.0
         nuth: 1e-4
         avg_rho: 1000.0
    )EOF";
  }

 private:
  bool initialized_ = false;
  std::vector<std::unique_ptr<AMRLevel<Q>>> levels_;       // levels_[0] = coarsest
  std::vector<std::unique_ptr<AMRCoupling<Q>>> couplings_;  // couplings_[i]: levels_[i] <-> levels_[i+1]

  // The GridConfig used to build each level (level_cfg_[0] is the physical domain, fixed for
  // the whole run). Needed by remesh_if_needed() (docs/AMR_PLAN.md Phase 6 sub-problem 2) to
  // recompute a moved patch's new GridConfig relative to its PARENT's current one -- build_level()
  // itself discards its GridConfig argument once the grid is built, so without this a level
  // couldn't be rebuilt at a new position without re-deriving its parent's geometry from scratch.
  std::vector<GridConfig> level_cfg_;

  // Snapshot of each level-with-a-child's (rho, u) at the start of its own current timestep
  // (docs/AMR_PLAN.md Phase 2 sub-cycling): its child's first sub-step needs this, blended
  // with the parent's post-step state, since it lands strictly between two parent timesteps.
  // rho_prev_[i]/u_prev_[i] correspond to levels_[i] and are only ever sized/used when
  // levels_[i] has a child (i.e. for i < couplings_.size()). u_prev_[i] is laid out exactly
  // like a FieldView<3> over rho_prev_[i]'s sibling buffer, so it can be wrapped in a real
  // FieldView<3> on demand.
  std::vector<std::vector<double>> rho_prev_;
  std::vector<std::vector<double>> u_prev_;

  static constexpr int ratio_ = 2;

  LBMParameters make_params(double dx) {
    LBMParameters p;
    p.nuth_ = *nuth;
    p.avg_rho_ = *avg_rho;
    p.define_by_c(*celerity, dx);
    p.Fext_ = convert_force<LBM_UNITS>(*Fext, p);
    return p;
  }

  // `is_amr_interface`: make_domain<Q>() shrinks the grid's ext_ box (used by is_defined() /
  // Traversal::Extend to decide which neighbor nodes are "valid") at every non-periodic
  // boundary, on the assumption that there's genuinely nothing beyond it (a physical wall).
  // That's wrong for a patch's boundary: the ghost shell there holds real data, written by
  // apply_c2f() every step, which streaming_step2() must be allowed to read. Widening ext_
  // back to the full local box after construction fixes this -- without touching `periodic_`,
  // which must stay false so make_domain()'s neighbor search finds no (spurious, self-wrapping)
  // MPI neighbor for a single-rank, non-periodic patch (see docs/AMR_PLAN.md Phase 1 notes).
  // Only ever used for i>0 levels: level 0's non-periodic faces, if any, are genuine physical
  // boundaries where the original shrinking behavior is the correct one.
  //
  // This also makes the ghost-shell nodes themselves "active" under Traversal::Extend
  // (check_level<Extend> becomes true for them too, not just for the Real boundary nodes
  // that need to read them) -- checked against streaming_step2()'s swap, which is safe: the
  // odd/even `iopp` pairing (scheme.hpp) means every undirected cell-pair edge in the lattice
  // is only ever initiated from the endpoint for which that direction is odd-numbered, so an
  // edge between two now-"active" ghost nodes is just extra, harmless work (both will be
  // overwritten by the next apply_c2f() call anyway) -- it never double-processes, or
  // corrupts, the Real<->ghost edges this fix exists for.
  template <typename ParExecCtxFunc>
  std::unique_ptr<AMRLevel<Q>> build_level(const GridConfig& grid_cfg, double dx, ParExecCtxFunc& par_exec_ctx,
                                           bool is_amr_interface = false) {
    auto level = std::make_unique<AMRLevel<Q>>();
    SubGridConfig sub_grid = load_balancing(grid_cfg, *mpi, /*use_periodic=*/false);
    level->domain_ = make_domain<Q>(grid_cfg, sub_grid);

    if (is_amr_interface) {
      Box3D full_box = level->domain_.grid().bx_;
      level->domain_.grid().set_ext(full_box);
    }

    resize_lbm_fields<Q>(level->domain_, level->fields_);
    level->region_.build_traversal(level->domain_.grid(), level->domain_.MPI_coord_, level->domain_.MPI_grid_size_);
    level->params_ = make_params(dx);

    init_obst obst_func = {onika::cuda::vector_data(level->fields_.obst_)};
    parallel_for_id<Area::Local, Traversal::All>(level->domain_.grid(), obst_func, parallel_execution_context());
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    // Equilibrium baseline (rho=1, u=0) everywhere, including the ghost shell: otherwise f_
    // starts at exactly zero (resize_lbm_fields fills with 0), so density would stay zero
    // through every step and apply_c2f()/apply_f2c() would never be exercised with real data.
    Box3D all_box = level->domain_.grid().template build_box<Area::Local, Traversal::All>();
    FieldView<Q> pf_init = level->fields_.distributions();
    if (*perturb_amplitude != 0.0) {
      GaussianDensityInit<Q> bump_func = {*perturb_center, *perturb_sigma, *perturb_amplitude};
      parallel_for(all_box, bump_func, parallel_execution_context(), pf_init, level->domain_.grid());
    } else {
      SetDistributionFunc<Q> dist_func;
      parallel_for(all_box, dist_func, parallel_execution_context(), pf_init, 1.0, level->domain_.grid());
    }
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    return level;
  }

  // Current physical center of the Ball obstacle: obstacle_center at t=0 plus
  // obstacle_velocity*timestep (see that slot's own doc for the simplifications). Recomputed
  // from `*timestep` directly rather than accumulated step by step, so it's deterministic and
  // reproducible regardless of call order.
  onika::math::Vec3d obstacle_center_now() { return *obstacle_center + (*obstacle_velocity) * double(*timestep); }

  // Stamps the single Ball obstacle (enable_obstacle) into one level's own obst_ field, at its
  // own resolution and CURRENT position (obstacle_center_now()) -- same pattern as the
  // production ApplyUpdateObstaclesFunc (update_obstacles.hpp): project the ball's AABB to
  // grid indices, restrict to the local subdomain, and mark every node inside the ball
  // WALL_CELL. Called once per level right after build_level() (which has already set every
  // node to FLUID_CELL via init_obst) and again, after a fresh FLUID_CELL reset, every step by
  // update_moving_obstacle() when the obstacle actually moves -- for every level so the
  // obstacle is consistently represented at every resolution, even though only the finest
  // level actually enclosing it matters once F2C starts overwriting its parent's coincident
  // nodes every step.
  void stamp_obstacle(AMRLevel<Q>& level) {
    if (!*enable_obstacle) return;
    LBMGrid& grid = level.domain_.grid();
    Ball ball(obstacle_center_now(), *obstacle_radius);

    onika::math::AABB bounds = ball.covered();
    Point3D pmin = grid.template project_to_grid<Area::Global>(bounds.bmin);
    Point3D pmax = grid.template project_to_grid<Area::Global>(bounds.bmax);
    Box3D global_box = {pmin, pmax};

    auto [is_inside_subdomain, local_box] = grid.template restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);
    if (is_inside_subdomain) {
      SetObstacleFunc<Ball> func = {ball, grid, level.fields_.obstacles(), WALL_CELL};
      hippoLBM::parallel_for(local_box, func, parallel_execution_context("amr_stamp_obstacle"));
      ONIKA_CU_DEVICE_SYNCHRONIZE();
    }
  }

  // Phase 6 sub-problem 2 (docs/AMR_PLAN.md): re-stamps the obstacle at its current position on
  // every level, every coarse step -- a full FLUID_CELL reset (init_obst, same functor
  // build_level() uses) followed by stamp_obstacle(), since stamping only ever ADDS WALL_CELL
  // and never removes it from the obstacle's previous position. Nodes that transition
  // solid->fluid keep whatever distributions wall_bounce_back last wrote there (a reasonable
  // starting guess, since bounce-back values are themselves derived from real neighboring
  // fluid data) rather than being explicitly reinitialized: normal BGK relaxation pulls them
  // to the correct local equilibrium within a few steps, a documented simplification rather
  // than special-cased reinitialization. No-op (and no extra cost at all) when
  // obstacle_velocity is {0,0,0}, so this never affects any of the static-obstacle tests.
  void update_moving_obstacle() {
    if (!*enable_obstacle) return;
    const onika::math::Vec3d v = *obstacle_velocity;
    if (v.x == 0.0 && v.y == 0.0 && v.z == 0.0) return;

    for (auto& level : levels_) {
      init_obst obst_func = {onika::cuda::vector_data(level->fields_.obst_)};
      parallel_for_id<Area::Local, Traversal::All>(level->domain_.grid(), obst_func, parallel_execution_context());
      ONIKA_CU_DEVICE_SYNCHRONIZE();
      stamp_obstacle(*level);
    }
  }

  // Builds one child level's GridConfig from its PARENT's (not necessarily level 0's) grid
  // config and the physical patch bounds refining it: node count derived directly for the
  // non-periodic node convention (intervals + 1), not via the existing grid_resolution()
  // helper, whose dims_ *= resolution is only correct for a periodic-style (dims_ == interval
  // count) grid. `level_index` is only used to make error messages point at the right
  // patch_bounds_list entry.
  GridConfig build_patch_grid_config(const GridConfig& parent_cfg, const onika::math::AABB& patch_bounds,
                                     size_t level_index) {
    const double dx_parent = (parent_cfg.bounds_.bmax.x - parent_cfg.bounds_.bmin.x) / double(parent_cfg.dims_.i - 1);

    auto check_multiple = [&](double value, const char* what) -> int {
      const double n = value / dx_parent;
      const double rounded = std::round(n);
      if (std::fabs(n - rounded) > 1e-6) {
        std::ostringstream oss;
        oss << "[amr_n_level] Error: patch_bounds_list[" << level_index << "] " << what << " (" << value
            << ") is not an exact multiple of its parent level's dx (" << dx_parent
            << "); each patch must align exactly with its parent level's grid lines (level 0's for index 0, the "
               "previous patch's for later indices).";
        throw std::runtime_error(oss.str());
      }
      return int(rounded);
    };

    // origin alignment (not just length): the patch's own bmin must itself fall on its
    // parent's grid line relative to the parent domain's bmin, not just have an aligned length.
    check_multiple(patch_bounds.bmin.x - parent_cfg.bounds_.bmin.x, "bmin.x - parent bmin.x");
    check_multiple(patch_bounds.bmin.y - parent_cfg.bounds_.bmin.y, "bmin.y - parent bmin.y");
    check_multiple(patch_bounds.bmin.z - parent_cfg.bounds_.bmin.z, "bmin.z - parent bmin.z");

    // Node counts AT THE PARENT's dx first (so the "exact multiple" validation above is
    // meaningful), then refined to this level's own dx by hand: NOT via the existing
    // grid_resolution() helper, which does dims_ *= resolution -- correct for a periodic-style
    // grid (dims_ == interval count), but off by one for this non-periodic, node-counted
    // (dims_ == intervals+1) patch grid.
    const int patch_nodes_i = check_multiple(patch_bounds.bmax.x - patch_bounds.bmin.x, "bmax.x - bmin.x") + 1;
    const int patch_nodes_j = check_multiple(patch_bounds.bmax.y - patch_bounds.bmin.y, "bmax.y - bmin.y") + 1;
    const int patch_nodes_k = check_multiple(patch_bounds.bmax.z - patch_bounds.bmin.z, "bmax.z - bmin.z") + 1;

    GridConfig child_cfg;
    child_cfg.bounds_ = patch_bounds;
    child_cfg.dims_.i = (patch_nodes_i - 1) * ratio_ + 1;
    child_cfg.dims_.j = (patch_nodes_j - 1) * ratio_ + 1;
    child_cfg.dims_.k = (patch_nodes_k - 1) * ratio_ + 1;
    child_cfg.periodic_ = {false, false, false};  // a patch is never independently periodic
    return child_cfg;
  }

  template <typename ParExecCtxFunc>
  void step_level(AMRLevel<Q>& level, ParExecCtxFunc& par_exec_ctx, bool is_physical_boundary_level) {
    auto& data = level.fields_;
    auto& traversals = level.region_;
    auto& params = level.params_;
    LBMGrid& grid = level.domain_.grid();

    FieldView<3> pm1 = data.flux();
    int* const pobst = data.obstacles();
    FieldView<Q> pf = data.distributions();
    double* const pm0 = data.densities();

    {
      FextConstantFunc fext = {params.Fext_};
      macro_variables<Q, FextConstantFunc> func = {fext};
      auto [ptr, size] = traversals.template get_data<Traversal::All>();
      parallel_for_id(ptr, size, func, parallel_execution_context(), pm1, pobst, pf, pm0);
    }
    {
      auto [ptr, size] = traversals.get_levels();
      FextConstantFunc fext = {params.Fext_};
      bgk<Q, Traversal::Real, FextConstantFunc> func = {ptr, fext, pm1, pobst, pf, pm0, params.tau_};
      parallel_for_simple(size, func, parallel_execution_context("amr_bgk"));
    }
    if (*enable_obstacle) {
      // Same placement as the production pipeline's pre_stream_bcs (data/config/config_compute_loop.msp):
      // after collision, before streaming -- bcs::wall_bounce_back<19> is a no-op on every node
      // where obst_ != WALL_CELL, so this is safe to call even where stamp_obstacle() found no
      // overlap with this level's subdomain.
      Box3D extend = grid.template build_box<Area::Local, Traversal::Extend>();
      bcs::wall_bounce_back<Q> wbb = {grid, pobst, pf};
      onika::parallel::ParallelExecutionSpace<3> extend_range = set(extend);
      parallel_for(extend_range, wbb, parallel_execution_context("amr_wall_bounce_back"));
    }
    {
      auto [ptr, size] = traversals.get_levels();
      streaming_step1<Q, Traversal::Real> step1 = {ptr, pf};
      streaming_step2<Q, Traversal::Extend> step2 = {ptr, grid, pf};
      parallel_for_simple(size, step1, parallel_execution_context("amr_streaming_step1"));
      update_ghost(level.domain_, pf, par_exec_ctx);
      parallel_for_simple(size, step2, parallel_execution_context("amr_streaming_step2"));
    }

    // Couette walls sit on the global domain boundary, i.e. level 0's own Plan_xy_0/Plan_xy_l.
    // Every other level is an interior AMR block (is_amr_interface in build_level()): its own
    // Plan_xy_0/Plan_xy_l are a coarse/fine interface, not physical walls, and must be left to
    // C2F/F2C -- imposing the wall BC there clobbered a patch's top/bottom nodes with the wall
    // velocity, which apply_f2c() then injected straight into its parent (seen as the parent's
    // ux(z) profile jumping to exactly wall_velocity_top at the patch's far face; see
    // docs/AMR_PLAN.md Couette validation).
    if (*apply_couette_bc && is_physical_boundary_level) apply_couette_walls(level);
  }

  // Couette validation (docs/AMR_PLAN.md): Neumann BC at z=0/z=L, same mechanism and ordering
  // (right after streaming) as the standalone `neumann` operator (src/bcs/neuman.cu).
  void apply_couette_walls(AMRLevel<Q>& level) {
    FieldView<Q> pf = level.fields_.distributions();
    int* const pobst = level.fields_.obstacles();
    auto make_ctx = [this](const char* name) { return this->parallel_execution_context(name); };

    {
      auto [ux, uy, uz] = convert_velocity<LBM_UNITS>(*wall_velocity_bottom, level.params_);
      auto traversal = level.region_.template get_data<Traversal::Plan_xy_0>();
      if (traversal.size_ > 0) {
        dispatch_traversal<Neumann, Q>(make_ctx, Traversal::Plan_xy_0, "amr_neumann_bottom", traversal.ptr_,
                                       traversal.size_, pobst, pf, ux, uy, uz);
      }
    }
    {
      auto [ux, uy, uz] = convert_velocity<LBM_UNITS>(*wall_velocity_top, level.params_);
      auto traversal = level.region_.template get_data<Traversal::Plan_xy_l>();
      if (traversal.size_ > 0) {
        dispatch_traversal<Neumann, Q>(make_ctx, Traversal::Plan_xy_l, "amr_neumann_top", traversal.ptr_,
                                       traversal.size_, pobst, pf, ux, uy, uz);
      }
    }
  }

  // Sums rho directly from f_ (not the m0_ cache, which may be stale relative to the most
  // recent streaming/F2C) so this reflects the true current state regardless of call order.
  void print_level_diagnostics(const char* name, AMRLevel<Q>& level) {
    const uint64_t n = level.fields_.size();
    FieldView<Q> pf = level.fields_.distributions();
    double sum_rho = 0.0;
    double min_rho = std::numeric_limits<double>::max();
    double max_rho = std::numeric_limits<double>::lowest();
    bool has_nan = false;
    for (uint64_t idx = 0; idx < n; idx++) {
      double rho = 0.0;
      for (int iLB = 0; iLB < Q; iLB++) rho += pf(idx, iLB);
      if (std::isnan(rho)) has_nan = true;
      sum_rho += rho;
      min_rho = std::min(min_rho, rho);
      max_rho = std::max(max_rho, rho);
    }
    lout << "[amr_n_level] " << name << ": sum(rho)=" << sum_rho << " avg(rho)=" << (sum_rho / double(n))
         << " min(rho)=" << min_rho << " max(rho)=" << max_rho << (has_nan ? " *** NaN detected ***" : "")
         << std::endl;
  }

  // Prints rho along the x axis at the (local) y,z nearest to `along_point`, over this
  // level's own Real nodes -- lets us eyeball continuity across a coarse/fine interface
  // (no NaN/sum check alone can catch a smooth-looking-but-wrong interpolation).
  void print_line_profile(const char* name, AMRLevel<Q>& level, const onika::math::Vec3d& along_point) {
    LBMGrid& grid = level.domain_.grid();
    FieldView<Q> pf = level.fields_.distributions();
    const std::string tag = std::string("amr_n_level ") + name;
    print_rho_line_profile<Q>(tag.c_str(), grid, pf, along_point);
  }

  // Prints the Q-criterion (docs/AMR_PLAN.md Phase 6, sub-problem 1: flow-driven refinement
  // indicator) along the z axis at the (local) x,y nearest to `along_point`, over this level's
  // own Real nodes. Calls compute_macro() itself first: advance()'s leaf (childless) level
  // only has macro_variables run at the TOP of its last step_level(), before that step's own
  // collision/streaming -- its fields_.flux() would otherwise be stale by one full timestep.
  void print_q_profile(const char* name, AMRLevel<Q>& level, const onika::math::Vec3d& along_point) {
    compute_macro(level);
    LBMGrid& grid = level.domain_.grid();
    FieldView<3> u = level.fields_.flux();
    const uint64_t n = level.fields_.size();
    std::vector<double> q_buf(n, 0.0);

    Box3D real = grid.template build_box<Area::Local, Traversal::Real>();
    QCriterionFunc<Q> func = {grid, u, q_buf.data()};
    parallel_for(real, func, parallel_execution_context());
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    const onika::math::Vec3d proj = (along_point - grid.origin_) / grid.dx_;
    const Point3D global_pt = {int(std::lround(proj.x)), int(std::lround(proj.y)), int(std::lround(proj.z))};
    const Point3D local_pt = grid.template convert<Area::Local>(global_pt);

    std::ostringstream oss;
    oss << "[amr_n_level " << name << "] Q-criterion(z) profile (z=" << real.start(2) << ".." << real.end(2)
        << "): ";
    for (int z = real.start(2); z <= real.end(2); z++) {
      const int idx = grid(local_pt[0], local_pt[1], z);
      oss << q_buf[idx] << " ";
    }
    lout << oss.str() << std::endl;
  }

  // Phase 6 sub-problem 2 (docs/AMR_PLAN.md): finds where to recenter a patch from
  // `parent`'s own Q-criterion field, searched within `old_bounds` padded by
  // `search_margin_cells` parent cells on every side (clamped to the parent's own Real region
  // by restrict_box_to_grid, so this can never propose a position outside the parent even for
  // a large margin). Reuses QCriterionFunc (amr_refinement_indicator.hpp) over the WHOLE
  // parent Real box rather than just the padded window: simpler than a windowed kernel launch,
  // and cheap since this only runs once every remesh_freq steps, not every step.
  //
  // Returns the |Q|-weighted CENTROID of the search window, not the single peak-|Q| node: an
  // obstacle's wake is rarely symmetric around the obstacle's own geometric center (background
  // Couette shear biases it to one side), so centering the patch on the single hottest node
  // tends to push the patch toward the wake and partly off the obstacle, which then shows up
  // as a *new* peak back on the other side next remesh event -- a non-convergent back-and-forth
  // observed empirically on amr_couette_obstacle_remesh.msp before this fix. The centroid
  // naturally balances a one-sided wake against the (smaller but nonzero) signal on the
  // opposite, less-disturbed side, converging to a stable position once the patch is large
  // enough to contain the whole disturbed region.
  onika::math::Vec3d find_peak_q_center(AMRLevel<Q>& parent, const onika::math::AABB& old_bounds,
                                        int search_margin_cells) {
    compute_macro(parent);
    LBMGrid& grid = parent.domain_.grid();
    const uint64_t n = parent.fields_.size();
    std::vector<double> q_buf(n, 0.0);

    Box3D real = grid.template build_box<Area::Local, Traversal::Real>();
    QCriterionFunc<Q> func = {grid, parent.fields_.flux(), q_buf.data()};
    parallel_for(real, func, parallel_execution_context());
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    Point3D pmin = grid.template project_to_grid<Area::Global>(old_bounds.bmin);
    Point3D pmax = grid.template project_to_grid<Area::Global>(old_bounds.bmax);
    Box3D global_search = {{pmin[0] - search_margin_cells, pmin[1] - search_margin_cells, pmin[2] - search_margin_cells},
                           {pmax[0] + search_margin_cells, pmax[1] + search_margin_cells, pmax[2] + search_margin_cells}};
    auto [is_inside, local_search] = grid.template restrict_box_to_grid<Area::Local, Traversal::Real>(global_search);

    const onika::math::Vec3d old_center = (old_bounds.bmin + old_bounds.bmax) * 0.5;
    if (!is_inside) return old_center;  // shouldn't happen (old_bounds is itself inside the parent), stay put

    double weight_sum = 0.0;
    double peak_abs_q = 0.0;
    onika::math::Vec3d weighted_pos_sum = {0.0, 0.0, 0.0};
    for (int z = local_search.start(2); z <= local_search.end(2); z++) {
      for (int y = local_search.start(1); y <= local_search.end(1); y++) {
        for (int x = local_search.start(0); x <= local_search.end(0); x++) {
          const double w = std::fabs(q_buf[grid(x, y, z)]);
          const onika::math::Vec3d pos = grid.template compute_position<Area::Global>(x, y, z);
          weighted_pos_sum += pos * w;
          weight_sum += w;
          peak_abs_q = std::max(peak_abs_q, w);
        }
      }
    }
    // Summing |Q| over many nodes makes weight_sum comfortably nonzero even with NO genuine
    // feature present (plain Couette shear alone gives Q~1e-5 per node, never exactly 0) -- so
    // a bare "weight_sum ~ 0" guard doesn't actually prevent chasing background noise (this
    // was tried and failed empirically, see remesh_q_threshold's own doc). What distinguishes
    // a real feature is its PEAK magnitude relative to that background floor, so require the
    // single highest |Q| in the window to clear remesh_q_threshold before trusting the
    // centroid at all.
    if (peak_abs_q < *remesh_q_threshold) return old_center;
    return weighted_pos_sum / weight_sum;
  }

  // Phase 6 sub-problem 2 (docs/AMR_PLAN.md): every remesh_freq coarse steps, recenters each
  // LEAF (childless) patch on the peak |Q| found in its own parent -- intermediate levels (a
  // patch that itself has a child) are deliberately skipped: rebuilding levels_[i+1] would
  // invalidate couplings_[i+1] (still referencing the just-destroyed old levels_[i+1] grid) and
  // leave levels_[i+2] without a valid parent, a cascading-rebuild problem not solved in this
  // first increment. Patch SIZE never changes, only its center; the number of levels and their
  // nesting structure stay fixed (no patches appear/disappear -- also not implemented yet).
  template <typename ParExecCtxFunc>
  void remesh_if_needed(ParExecCtxFunc& par_exec_ctx) {
    if (*remesh_freq <= 0 || (*timestep % *remesh_freq) != 0) return;

    for (size_t i = 0; i < couplings_.size(); i++) {
      const bool child_has_its_own_child = (i + 2 < levels_.size());
      if (child_has_its_own_child) continue;  // see method comment: not supported yet

      AMRLevel<Q>& parent = *levels_[i];
      const onika::math::AABB old_bounds = level_cfg_[i + 1].bounds_;
      const onika::math::Vec3d old_center = (old_bounds.bmin + old_bounds.bmax) * 0.5;
      const onika::math::Vec3d half_extent = (old_bounds.bmax - old_bounds.bmin) * 0.5;
      const double dx_parent =
          (level_cfg_[i].bounds_.bmax.x - level_cfg_[i].bounds_.bmin.x) / double(level_cfg_[i].dims_.i - 1);

      const onika::math::Vec3d peak_pos = find_peak_q_center(parent, old_bounds, int(*remesh_search_margin));

      auto snap = [&](double v, double origin) { return origin + std::round((v - origin) / dx_parent) * dx_parent; };
      const onika::math::Vec3d new_center = {snap(peak_pos.x, level_cfg_[i].bounds_.bmin.x),
                                             snap(peak_pos.y, level_cfg_[i].bounds_.bmin.y),
                                             snap(peak_pos.z, level_cfg_[i].bounds_.bmin.z)};

      const onika::math::Vec3d d = new_center - old_center;
      const double shift = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
      if (shift < 0.5 * dx_parent) continue;  // negligible move, not worth a rebuild

      const onika::math::AABB new_bounds = {new_center - half_extent, new_center + half_extent};

      // A proposed position too close to the parent's own boundary fails build_patch_grid_config()'s
      // alignment check or AMRCoupling's own "ghost node projects outside the coarse grid" check
      // (amr_coupling.hpp) -- caught here and the patch just stays put for this remesh event,
      // rather than aborting the whole run over a single bad move.
      try {
        GridConfig new_child_cfg = build_patch_grid_config(level_cfg_[i], new_bounds, i);
        const double dx_child = dx_parent / double(ratio_);

        auto new_child = build_level(new_child_cfg, dx_child, par_exec_ctx, /*is_amr_interface=*/true);
        stamp_obstacle(*new_child);

        // Fold the moving patch's fine detail back into its parent before discarding it (uses
        // the OLD coupling_[i]/levels_[i+1], still valid at this point).
        couplings_[i]->apply_f2c(parent, *levels_[i + 1]);

        levels_[i + 1] = std::move(new_child);
        couplings_[i] = std::make_unique<AMRCoupling<Q>>(*levels_[i], *levels_[i + 1]);
        couplings_[i]->seed_from_parent(parent, *levels_[i + 1]);  // whole-patch seed, not just the ghost shell

        level_cfg_[i + 1] = new_child_cfg;

        if (*verbosity) {
          lout << "[amr_n_level] remesh: level" << (i + 1) << " recentered (shift=" << shift << ") to ("
               << new_center.x << "," << new_center.y << "," << new_center.z << ")" << std::endl;
        }
      } catch (const std::exception& err) {
        if (*verbosity) {
          lout << "[amr_n_level] remesh: keeping level" << (i + 1) << " in place (" << err.what() << ")"
               << std::endl;
        }
      }
    }
  }

  // Writes every level (reusing write_paraview() from the io plugin, unmodified, once per
  // level) plus a .vtm multi-block manifest wrapping all the resulting .pvtr files, so
  // ParaView opens one file and shows every level together, each at its own real resolution
  // (not resampled onto a common grid).
  void write_amr_paraview(long ts) {
    const std::string basedir = *output_directory + "/";
    std::filesystem::create_directories(basedir);
    ExternalParaviewFields no_external_fields;

    std::vector<std::string> names(levels_.size());
    for (size_t i = 0; i < levels_.size(); i++) {
      names[i] = "level" + std::to_string(i) + "_%010ld";
      write_paraview(*mpi, names[i], basedir, ts, levels_[i]->fields_, levels_[i]->params_, levels_[i]->domain_,
                     no_external_fields, *verbosity);
    }

    if (levels_.size() < 2) return;  // nothing to wrap in a multi-block manifest

    // Reference the underlying per-rank .vtr piece directly, not the .pvtr wrapper:
    // vtkXMLMultiBlockDataReader could not create a reader for a nested .pvtr (a parallel/
    // composite format) when it's a child of a plain vtkMultiBlockDataSet -- confirmed by
    // trying it in ParaView. It has no trouble with a plain, single-piece .vtr. Phase 1-3 are
    // single-rank only (checked in execute()), so each level has exactly one .vtr piece, at
    // rank 0; this must be revisited if/when a level is ever split across multiple ranks.
    const std::string vtm_path = basedir + onika::format_string("amr_%010ld.vtm", ts);
    std::ofstream vtm(vtm_path);
    if (!vtm) {
      lout << "[amr_n_level] Warning: could not create '" << vtm_path << "', skipping the multi-block manifest."
           << std::endl;
      return;
    }
    vtm << "<?xml version=\"1.0\"?>\n";
    vtm << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\">\n";
    vtm << "  <vtkMultiBlockDataSet>\n";
    for (size_t i = 0; i < levels_.size(); i++) {
      const std::string vtr = onika::format_string(names[i], ts) + "/000000.vtr";
      vtm << "    <DataSet index=\"" << i << "\" name=\"level" << i << "\" file=\"" << vtr << "\"/>\n";
    }
    vtm << "  </vtkMultiBlockDataSet>\n";
    vtm << "</VTKFile>\n";

    if (*verbosity) {
      lout << "[amr_n_level] writing AMR paraview multi-block: " << vtm_path << std::endl;
    }
  }

  // Only macro_variables, needed on a level before C2F can interpolate from it.
  void compute_macro(AMRLevel<Q>& level) {
    auto& data = level.fields_;
    auto& traversals = level.region_;
    FextConstantFunc fext = {level.params_.Fext_};
    macro_variables<Q, FextConstantFunc> func = {fext};
    FieldView<3> pm1 = data.flux();
    int* const pobst = data.obstacles();
    FieldView<Q> pf = data.distributions();
    double* const pm0 = data.densities();
    auto [ptr, size] = traversals.template get_data<Traversal::All>();
    parallel_for_id(ptr, size, func, parallel_execution_context(), pm1, pobst, pf, pm0);
  }

  // Copies levels_[level_idx]'s current (rho, u) -- assumed freshly computed by
  // compute_macro() just before this call -- into rho_prev_[level_idx]/u_prev_[level_idx],
  // for apply_c2f()'s temporal blending (docs/AMR_PLAN.md Phase 2).
  void snapshot_prev(size_t level_idx) {
    AMRLevel<Q>& level = *levels_[level_idx];
    const uint64_t n = level.fields_.size();
    const double* const rho_now = level.fields_.densities();
    FieldView<3> u_now = level.fields_.flux();
    FieldView<3> u_prev_view{u_prev_[level_idx].data(), n};
    for (uint64_t i = 0; i < n; i++) {
      rho_prev_[level_idx][i] = rho_now[i];
      u_prev_view(i, 0) = u_now(i, 0);
      u_prev_view(i, 1) = u_now(i, 1);
      u_prev_view(i, 2) = u_now(i, 2);
    }
  }

  // Classic recursive Berger-Colella advance: steps levels_[level_idx] one of its own
  // timesteps; if it has a child, couples (C2F) and recursively advances that child `ratio_`
  // times first, then restricts (F2C) once the child has caught back up. For a single level
  // (no child), this is just step_level(). For two levels, this is exactly the Phase 2
  // sub-cycled sequence this operator used to hardcode: step coarse -> C2F(alpha=0.5) -> step
  // fine -> C2F(alpha=1.0) -> step fine -> F2C.
  template <typename ParExecCtxFunc>
  void advance(size_t level_idx, ParExecCtxFunc& par_exec_ctx) {
    const bool is_physical_boundary_level = (level_idx == 0);
    const bool has_child = (level_idx + 1 < levels_.size());

    if (has_child) {
      compute_macro(*levels_[level_idx]);
      snapshot_prev(level_idx);
    }

    step_level(*levels_[level_idx], par_exec_ctx, is_physical_boundary_level);

    if (has_child) {
      compute_macro(*levels_[level_idx]);  // this level's (rho, u) at t=n+1, needed by every C2F call below

      AMRLevel<Q>& parent = *levels_[level_idx];
      AMRLevel<Q>& child = *levels_[level_idx + 1];
      AMRCoupling<Q>& coupling = *couplings_[level_idx];
      FieldView<3> u_prev_view{u_prev_[level_idx].data(), parent.fields_.size()};

      for (int sub = 1; sub <= ratio_; sub++) {
        const double alpha = double(sub) / double(ratio_);
        coupling.apply_c2f(parent, child, rho_prev_[level_idx].data(), u_prev_view, alpha);
        advance(level_idx + 1, par_exec_ctx);
      }

      coupling.apply_f2c(parent, child);  // parent and child both now at t=n+1
    }
  }

 public:
  inline void execute() final {
    int mpi_size;
    MPI_Comm_size(*mpi, &mpi_size);
    if (mpi_size != 1) {
      lout << "[amr_n_level] Error: this operator only supports a single MPI rank (got " << mpi_size << ")."
           << std::endl;
      MPI_Abort(*mpi, EXIT_FAILURE);
    }

    auto par_exec_ctx = [this](const char* exec_name) { return this->parallel_execution_context(exec_name); };

    if (!initialized_) {
      GridConfig cfg0;
      cfg0.periodic_ = convert<std::array<bool, 3>>(*periodic);
      cfg0.dims_.i = cell_dims->i + (cfg0.periodic_[0] ? 0 : 1);
      cfg0.dims_.j = cell_dims->j + (cfg0.periodic_[1] ? 0 : 1);
      cfg0.dims_.k = cell_dims->k + (cfg0.periodic_[2] ? 0 : 1);
      cfg0.bounds_ = *bounds;
      const double dx0 = (cfg0.bounds_.bmax.x - cfg0.bounds_.bmin.x) / double(cfg0.dims_.i - 1);

      levels_.push_back(build_level(cfg0, dx0, par_exec_ctx, /*is_amr_interface=*/false));
      stamp_obstacle(*levels_[0]);
      level_cfg_.push_back(cfg0);

      GridConfig parent_cfg = cfg0;
      double parent_dx = dx0;
      for (size_t i = 0; i < patch_bounds_list->size(); i++) {
        GridConfig child_cfg = build_patch_grid_config(parent_cfg, (*patch_bounds_list)[i], i);
        const double child_dx = parent_dx / double(ratio_);

        levels_.push_back(build_level(child_cfg, child_dx, par_exec_ctx, /*is_amr_interface=*/true));
        stamp_obstacle(*levels_[i + 1]);
        level_cfg_.push_back(child_cfg);
        couplings_.push_back(std::make_unique<AMRCoupling<Q>>(*levels_[i], *levels_[i + 1]));

        if (couplings_.back()->f2c_.empty()) {
          lout << "[amr_n_level] Error: patch_bounds_list[" << i
               << "] has no node coincident with its parent level's Real region." << std::endl;
          MPI_Abort(*mpi, EXIT_FAILURE);
        }

        rho_prev_.emplace_back(levels_[i]->fields_.size());
        u_prev_.emplace_back(levels_[i]->fields_.size() * 3);

        parent_cfg = child_cfg;
        parent_dx = child_dx;
      }

      if (*verbosity) {
        for (size_t i = 0; i < levels_.size(); i++) {
          lout << "[amr_n_level] level" << i << " params:" << std::endl;
          levels_[i]->params_.print();
        }
      }

      initialized_ = true;
    }

    update_moving_obstacle();
    advance(0, par_exec_ctx);
    remesh_if_needed(par_exec_ctx);

    if (*verbosity) {
      for (size_t i = 0; i < levels_.size(); i++) {
        const std::string name = "level" + std::to_string(i);
        print_level_diagnostics(name.c_str(), *levels_[i]);
        if (*apply_couette_bc) {
          const std::string tag = std::string("amr_n_level ") + name;
          print_ux_profile_z<Q>(tag.c_str(), levels_[i]->domain_.grid(), levels_[i]->fields_.distributions(),
                                *perturb_center);
        } else {
          print_line_profile(name.c_str(), *levels_[i], *perturb_center);
        }
        if (*print_q_criterion) {
          print_q_profile(name.c_str(), *levels_[i], *perturb_center);
        }
      }
    }

    if (*paraview_freq > 0 && (*timestep % *paraview_freq) == 0) {
      write_amr_paraview(*timestep);
    }
  }
};

using AMRNLevel3D19Q = AMRNLevel<19>;

// === register factories ===
ONIKA_AUTORUN_INIT(amr_n_level) {
  OperatorNodeFactory::instance()->register_factory("amr_n_level", make_variant_operator<AMRNLevel>);
}
}  // namespace hippoLBM
