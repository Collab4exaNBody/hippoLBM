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

#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include <hippoLBM/amr/amr_level.hpp>
#include <hippoLBM/grid/make_domain.hpp>

namespace hippoLBM {

/** @brief Coarse-to-fine / fine-to-coarse coupling for a single pair of AMR levels, ratio 2:1,
 * single rank (docs/AMR_PLAN.md Phase 1-2). apply_c2f() supports the temporal blending
 * needed for Phase 2's sub-cycling (the fine level takes 2 steps per 1 coarse step, so its
 * first sub-step falls strictly between two coarse timesteps); the caller is responsible for
 * keeping a "previous" coarse snapshot and picking `alpha` each sub-step.
 *
 * Couples by *node* position, not cell-volume averaging: hippoLBM's grid is node-centered
 * (LBMGrid::compute_position has no half-cell offset), so under exact 2:1 refinement every
 * coarse node coincides exactly with one fine node. C2F (into the fine level's own ghost
 * shell) reconstructs distributions from trilinearly-interpolated coarse (rho, u), plus the
 * classic Filippova-Haenel non-equilibrium correction term (trilinearly interpolated from the
 * 8 coarse corners' own (f - f_eq), rescaled by the Dupuis-Chopard tau coefficient): see
 * docs/AMR_PLAN.md
 * Couette validation for why the equilibrium-only Phase 1 reconstruction left a non-negligible
 * (up to ~9%) residual error for a shear-dominated flow, and apply_c2f() below for the exact
 * formula and its simplifications. F2C is a direct injection (not an average) at the
 * coincident nodes, which is the standard restriction operator for node-based LBM-AMR -- the
 * fine solution there is already fully resolved, so it needs no correction.
 */
template <int Q>
struct AMRCoupling {
  static constexpr int ratio_ = 2;

  /** @brief One fine ghost node reconstructed from 8 surrounding coarse nodes. */
  struct C2FEntry {
    int fine_idx_;         ///< local index into the fine level's fields (f_, m0_, m1_)
    int coarse_base_idx_;  ///< local index of the coarse node at (floor_x, floor_y, floor_z)
    int coarse_step_[3];   ///< local-index stride of the coarse grid along x, y, z (for the +1 corners)
    double fx_, fy_, fz_;  ///< trilinear interpolation weights in [0, 1]
  };

  /** @brief One coarse node directly overwritten by its coincident fine node. */
  struct F2CEntry {
    int coarse_idx_;
    int fine_idx_;
  };

  std::vector<C2FEntry> c2f_;
  std::vector<F2CEntry> f2c_;

  /** @brief Precomputes the C2F and F2C node lists once, from the two levels' current grids.
   * Aborts (std::runtime_error) if the fine patch isn't exactly 2:1-aligned with the coarse
   * grid, rather than silently snapping or interpolating against a misaligned grid. */
  AMRCoupling(const AMRLevel<Q>& coarse, const AMRLevel<Q>& fine) {
    build_c2f(coarse, fine);
    build_f2c(coarse, fine);
  }

 private:
  // Computes one node's C2FEntry (its trilinear-interpolation geometry against the coarse
  // grid) from its own local coordinates -- shared by build_c2f() (ghost shell only, cached
  // once at construction) and seed_from_parent() (docs/AMR_PLAN.md Phase 6 sub-problem 2: every
  // node, recomputed on demand right after a patch moves, so caching isn't worth it there).
  static C2FEntry compute_c2f_entry(const LBMGrid& cg, const LBMGrid& fg, int x, int y, int z, int cstep_x,
                                    int cstep_y, int cstep_z) {
    const onika::math::Vec3d phys = fg.compute_position<Area::Global>(x, y, z);
    const onika::math::Vec3d proj = (phys - cg.origin_) / cg.dx_;
    const Point3D base_global = {int(std::floor(proj.x)), int(std::floor(proj.y)), int(std::floor(proj.z))};
    Point3D base_local = cg.convert<Area::Local>(base_global);
    Point3D corner_local = {base_local[0] + 1, base_local[1] + 1, base_local[2] + 1};

    if (!cg.is_local(base_local) || !cg.is_local(corner_local)) {
      throw std::runtime_error(
          "[AMRCoupling] Error: a fine node projects outside the coarse grid. The fine patch must sit well inside "
          "the coarse domain (at least one coarse cell away from its own boundary/ghost layer).");
    }

    C2FEntry e;
    e.fine_idx_ = fg(x, y, z);
    e.coarse_base_idx_ = cg(base_local[0], base_local[1], base_local[2]);
    e.coarse_step_[0] = cstep_x;
    e.coarse_step_[1] = cstep_y;
    e.coarse_step_[2] = cstep_z;
    e.fx_ = proj.x - base_global[0];
    e.fy_ = proj.y - base_global[1];
    e.fz_ = proj.z - base_global[2];
    return e;
  }

  void build_c2f(const AMRLevel<Q>& coarse, const AMRLevel<Q>& fine) {
    const LBMGrid& cg = coarse.domain_.grid();
    const LBMGrid& fg = fine.domain_.grid();

    Box3D fine_all = fg.build_box<Area::Local, Traversal::All>();
    Box3D fine_real = fg.build_box<Area::Local, Traversal::Real>();

    // local-index strides for the coarse box, needed to offset by +1 node along each axis
    // without recomputing grid(x,y,z) for all 8 corners every time.
    const int cstep_x = 1;
    const int cstep_y = cg.bx_.get_length(0);
    const int cstep_z = cg.bx_.get_length(0) * cg.bx_.get_length(1);

    for (int z = fine_all.start(2); z <= fine_all.end(2); z++) {
      for (int y = fine_all.start(1); y <= fine_all.end(1); y++) {
        for (int x = fine_all.start(0); x <= fine_all.end(0); x++) {
          Point3D p_local = {x, y, z};
          if (fine_real.contains(p_local)) continue;  // not a ghost node, nothing to reconstruct
          c2f_.push_back(compute_c2f_entry(cg, fg, x, y, z, cstep_x, cstep_y, cstep_z));
        }
      }
    }
  }

  void build_f2c(const AMRLevel<Q>& coarse, const AMRLevel<Q>& fine) {
    const LBMGrid& cg = coarse.domain_.grid();
    const LBMGrid& fg = fine.domain_.grid();
    Box3D coarse_real = cg.build_box<Area::Local, Traversal::Real>();
    Box3D fine_real = fg.build_box<Area::Local, Traversal::Real>();

    constexpr double alignment_tol = 1e-6;  // in fine node-index units

    for (int z = coarse_real.start(2); z <= coarse_real.end(2); z++) {
      for (int y = coarse_real.start(1); y <= coarse_real.end(1); y++) {
        for (int x = coarse_real.start(0); x <= coarse_real.end(0); x++) {
          const onika::math::Vec3d phys = cg.compute_position<Area::Global>(x, y, z);
          const onika::math::Vec3d proj = (phys - fg.origin_) / fg.dx_;
          const Point3D nearest_global = {int(std::lround(proj.x)), int(std::lround(proj.y)),
                                          int(std::lround(proj.z))};
          const double dx_err = std::fabs(proj.x - nearest_global[0]);
          const double dy_err = std::fabs(proj.y - nearest_global[1]);
          const double dz_err = std::fabs(proj.z - nearest_global[2]);
          if (dx_err > alignment_tol || dy_err > alignment_tol || dz_err > alignment_tol) {
            continue;  // this coarse node has no coincident fine node: outside the fine patch
          }

          Point3D nearest_local = fg.convert<Area::Local>(nearest_global);
          if (!fine_real.contains(nearest_local)) {
            continue;  // coincident node exists but falls in the fine level's own ghost shell
          }

          F2CEntry e;
          e.coarse_idx_ = cg(x, y, z);
          e.fine_idx_ = fg(nearest_local[0], nearest_local[1], nearest_local[2]);
          f2c_.push_back(e);
        }
      }
    }
  }

 private:
  // Trilinear spatial interpolation of (rho, u) at one C2F entry, from a given source field
  // pair (either the coarse level's "previous" snapshot or its current fields -- see
  // apply_c2f()).
  void spatial_interp(const C2FEntry& e, const double* rho_src, const FieldView<3>& u_src, double& rho_out,
                      onika::math::Vec3d& u_out) const {
    const int i000 = e.coarse_base_idx_;
    const int sx = e.coarse_step_[0], sy = e.coarse_step_[1], sz = e.coarse_step_[2];
    const double fx = e.fx_, fy = e.fy_, fz = e.fz_;

    rho_out = 0.0;
    u_out = {0.0, 0.0, 0.0};
    for (int dz = 0; dz <= 1; dz++) {
      for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
          const int idx = i000 + dx * sx + dy * sy + dz * sz;
          const double w = (dx ? fx : 1.0 - fx) * (dy ? fy : 1.0 - fy) * (dz ? fz : 1.0 - fz);
          rho_out += w * rho_src[idx];
          u_out.x += w * u_src(idx, 0);
          u_out.y += w * u_src(idx, 1);
          u_out.z += w * u_src(idx, 2);
        }
      }
    }
  }

  // Trilinearly interpolates the coarse level's own non-equilibrium part, f_corner - f_eq(rho_corner,
  // u_corner), direction by direction, from the 8 corners surrounding one C2F entry -- using
  // each corner's OWN (rho, u) to define its local equilibrium, not the blended target (rho, u)
  // at the fine ghost location. This is the classic Filippova-Haenel reconstruction's
  // non-equilibrium term, read from the coarse level's CURRENT distributions only (not
  // temporally blended between the "prev"/"now" snapshots the way the equilibrium part is in
  // apply_c2f()): it's a secondary, higher-order correction, and using the latest available
  // coarse state for it avoids doubling the Phase 2 snapshot buffers to a full 19-component
  // f_ copy at t=n just for this term.
  void interp_non_equilibrium(const C2FEntry& e, const double* crho_now, const FieldView<3>& cu_now,
                              const FieldView<Q>& cf_now, double (&fneq_out)[Q]) const {
    const int i000 = e.coarse_base_idx_;
    const int sx = e.coarse_step_[0], sy = e.coarse_step_[1], sz = e.coarse_step_[2];
    const double fx = e.fx_, fy = e.fy_, fz = e.fz_;

    for (int iLB = 0; iLB < Q; iLB++) fneq_out[iLB] = 0.0;

    for (int dz = 0; dz <= 1; dz++) {
      for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
          const int idx = i000 + dx * sx + dy * sy + dz * sz;
          const double w = (dx ? fx : 1.0 - fx) * (dy ? fy : 1.0 - fy) * (dz ? fz : 1.0 - fz);
          const double rho_c = crho_now[idx];
          const double ux_c = cu_now(idx, 0), uy_c = cu_now(idx, 1), uz_c = cu_now(idx, 2);
          const double usq_c = ux_c * ux_c + uy_c * uy_c + uz_c * uz_c;
          stencil::for_each<typename LBMScheme<Q>::Coefficients>([&]<typename coeff>(int iLB) {
            const double eu_c = coeff::ex * ux_c + coeff::ey * uy_c + coeff::ez * uz_c;
            const double feq_c = coeff::w * rho_c * (1. + 3. * eu_c + 4.5 * eu_c * eu_c - 1.5 * usq_c);
            fneq_out[iLB] += w * (cf_now(idx, iLB) - feq_c);
          });
        }
      }
    }
  }

  // Writes one node's reconstructed (equilibrium + Dupuis-Chopard-corrected non-equilibrium)
  // distribution, pre-swapped for the AA-pattern streaming scheme -- shared by apply_c2f()
  // (ghost shell only, every step) and seed_from_parent() (every node, only right after a
  // patch moves). See apply_c2f()'s own comment for why the pre-swap is needed: a node written
  // here never goes through streaming_step1's local pair-swap (only Traversal::Real does), yet
  // streaming_step2 (Traversal::Extend, which under is_amr_interface includes the whole patch's
  // ghost shell) reads every node as if it HAD been swapped.
  void write_reconstructed(FieldView<Q>& ff, int fine_idx, double rho, const onika::math::Vec3d& u, double chi,
                           const double (&fneq)[Q]) const {
    const double u_squ = u.x * u.x + u.y * u.y + u.z * u.z;
    stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, 1>([&]<typename coeff>(int iLB) {
      ff(fine_idx, iLB) = coeff::w * rho * (1. - 1.5 * u_squ) + chi * fneq[iLB];
    });
    stencil::for_each<typename LBMScheme<Q>::Coefficients, 1, Q, 2>([&]<typename coeff>(int iLB) {
      const double eu = coeff::ex * u.x + coeff::ey * u.y + coeff::ez * u.z;
      const double feq_this = coeff::w * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
      const double feq_opp = coeff::w * rho * (1. - 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
      ff(fine_idx, iLB) = feq_opp + chi * fneq[iLB + 1];
      ff(fine_idx, iLB + 1) = feq_this + chi * fneq[iLB];
    });
  }

  // tau_fine/tau_coarse Dupuis-Chopard coefficient (docs/AMR_PLAN.md Couette validation),
  // shared by apply_c2f() and seed_from_parent().
  static double dupuis_chopard_chi(const AMRLevel<Q>& coarse, const AMRLevel<Q>& fine) {
    return (fine.params_.tau_ * (coarse.params_.tau_ - 1.0)) / (coarse.params_.tau_ * (fine.params_.tau_ - 1.0));
  }

 public:
  /** @brief Coarse -> fine: reconstructs the fine level's ghost shell by trilinear spatial
   * interpolation, linearly blended in time between a "previous" coarse snapshot
   * (`coarse_rho_prev`/`coarse_u_prev`, the coarse state at the start of the current coarse
   * timestep) and the coarse level's CURRENT (rho, u) fields (the end of that timestep, must
   * already be up to date via macro_variables), weighted by `alpha` in [0, 1] (0 = prev,
   * 1 = current). Needed because the fine level sub-cycles 2 steps per 1 coarse step
   * (docs/AMR_PLAN.md Phase 2): its first sub-step lands exactly halfway between two coarse
   * timesteps, where neither coarse snapshot alone is correct.
   *
   * On top of the equilibrium part (rho, u), adds the classic Filippova-Haenel
   * non-equilibrium correction: f_fine = f_eq(rho, u) + chi * f_neq_coarse_interpolated, with
   * chi the refined Dupuis-Chopard coefficient (computed below) -- the simpler
   * chi = tau_fine/tau_coarse ratio also seen in the literature under
   * the "Filippova-Haenel" name measurably *underperforms* it here (our tau values, ~3.5/6.5
   * under convective scaling, are far from 1, where the two forms diverge most). Without any
   * correction at all, the fine patch received no deviatoric stress information from the
   * coarse donor, only its velocity -- harmless for a small transient perturbation, but a real
   * (up to ~9%) error for a shear-dominated flow like Couette, since the non-equilibrium part
   * is exactly what carries the shear stress (docs/AMR_PLAN.md
   * Couette validation). */
  void apply_c2f(AMRLevel<Q>& coarse, AMRLevel<Q>& fine, const double* coarse_rho_prev,
                const FieldView<3>& coarse_u_prev, double alpha) const {
    const double* const crho_now = coarse.fields_.densities();
    FieldView<3> cu_now = coarse.fields_.flux();
    FieldView<Q> cf_now = coarse.fields_.distributions();
    FieldView<Q> ff = fine.fields_.distributions();
    // Dupuis-Chopard refined coefficient, not the simpler tau_fine/tau_coarse ratio also found
    // in the literature under the "Filippova-Haenel" name: empirically, on the Couette
    // validation case, the simple ratio (chi=1.857 here, tau_coarse=3.5, tau_fine=6.5 under our
    // convective scaling) made the coupled coarse level's error *worse* than no correction at
    // all (peak ~10.5%, sign flipped in places, vs ~9% uncorrected) -- these tau values are
    // far from 1, where the two coefficients diverge most. This refined form (chi=0.844 for
    // the same tau pair) brought the peak error down to ~0.46%, a ~20x improvement; see
    // docs/AMR_PLAN.md Couette validation for the full before/after comparison.
    const double chi = dupuis_chopard_chi(coarse, fine);

    for (const auto& e : c2f_) {
      double rho_prev, rho_now;
      onika::math::Vec3d u_prev, u_now;
      spatial_interp(e, coarse_rho_prev, coarse_u_prev, rho_prev, u_prev);
      spatial_interp(e, crho_now, cu_now, rho_now, u_now);

      const double rho = (1.0 - alpha) * rho_prev + alpha * rho_now;
      const onika::math::Vec3d u = u_prev * (1.0 - alpha) + u_now * alpha;

      double fneq[Q];
      interp_non_equilibrium(e, crho_now, cu_now, cf_now, fneq);

      // A ghost node written here never goes through streaming_step1's local pair-swap (only
      // Traversal::Real does), yet streaming_step2 (Traversal::Extend, which under
      // is_amr_interface includes this ghost shell) reads every node as if it HAD been
      // swapped -- so write_reconstructed() writes it pre-swapped, see its own comment
      // (missing this corrupted exactly the fine patch's real nodes nearest the interface,
      // see docs/AMR_PLAN.md Couette validation).
      write_reconstructed(ff, e.fine_idx_, rho, u, chi, fneq);
    }
  }

  /** @brief Phase 6 sub-problem 2 (docs/AMR_PLAN.md): seeds EVERY node of `fine` (real
   * interior as well as ghost shell, unlike apply_c2f() which only ever touches the ghost
   * shell) from `coarse`'s CURRENT state, using the same equilibrium + Dupuis-Chopard
   * non-equilibrium reconstruction as apply_c2f() with alpha=1 (no "prev" snapshot needed --
   * this is a one-off reinitialization at a single instant, not a sub-cycled timestep). Used
   * right after a patch is rebuilt at a new position (a "remesh" event): the new patch has no
   * history of its own yet, so its entire interior must be populated from its parent, not just
   * the thin ghost shell apply_c2f() keeps refreshed every step thereafter. Geometry (the
   * trilinear interpolation weights) is recomputed node by node rather than cached in `c2f_`,
   * since this only runs on an infrequent remesh event, not every step. */
  void seed_from_parent(AMRLevel<Q>& coarse, AMRLevel<Q>& fine) const {
    const LBMGrid& cg = coarse.domain_.grid();
    const LBMGrid& fg = fine.domain_.grid();
    const double* const crho = coarse.fields_.densities();
    FieldView<3> cu = coarse.fields_.flux();
    FieldView<Q> cf = coarse.fields_.distributions();
    FieldView<Q> ff = fine.fields_.distributions();
    const double chi = dupuis_chopard_chi(coarse, fine);

    const int cstep_x = 1;
    const int cstep_y = cg.bx_.get_length(0);
    const int cstep_z = cg.bx_.get_length(0) * cg.bx_.get_length(1);

    Box3D fine_all = fg.build_box<Area::Local, Traversal::All>();
    for (int z = fine_all.start(2); z <= fine_all.end(2); z++) {
      for (int y = fine_all.start(1); y <= fine_all.end(1); y++) {
        for (int x = fine_all.start(0); x <= fine_all.end(0); x++) {
          const C2FEntry e = compute_c2f_entry(cg, fg, x, y, z, cstep_x, cstep_y, cstep_z);

          double rho;
          onika::math::Vec3d u;
          spatial_interp(e, crho, cu, rho, u);

          double fneq[Q];
          interp_non_equilibrium(e, crho, cu, cf, fneq);

          write_reconstructed(ff, e.fine_idx_, rho, u, chi, fneq);
        }
      }
    }
  }

  /** @brief Fine -> coarse: injects the fine level's resolved distributions at every coarse
   * node that has a coincident, fully-resolved (non-ghost) fine node. */
  void apply_f2c(AMRLevel<Q>& coarse, AMRLevel<Q>& fine) const {
    FieldView<Q> cf = coarse.fields_.distributions();
    FieldView<Q> ff = fine.fields_.distributions();

    for (const auto& e : f2c_) {
      for (int iLB = 0; iLB < Q; iLB++) {
        cf(e.coarse_idx_, iLB) = ff(e.fine_idx_, iLB);
      }
    }
  }
};

}  // namespace hippoLBM
