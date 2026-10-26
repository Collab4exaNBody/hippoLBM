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

#include <mpi.h>

#include <array>
#include <cmath>
#include <map>
#include <vector>

#include <hippoLBM/amr/amr_level.hpp>
#include <hippoLBM/amr/amr_rank_owner.hpp>
#include <hippoLBM/grid/make_domain.hpp>

namespace hippoLBM {

/** @brief Phase 4 v1 (docs/AMR_PLAN.md): the inclusive global coarse-node index range that a
 * patch's C2F ghost-reconstruction + F2C coincidence together touch, padded by exactly 1
 * coarse cell on every side (the patch's own boundary already sits at least 1 coarse cell
 * inside the coarse domain, enforced by build_patch_grid_config()'s own alignment checks, so
 * this never needs clamping). Computed purely from shared, YAML-known config -- callable
 * identically on EVERY rank, owner or not, without that rank needing to have actually built
 * the patch level itself (only the owner does, see amr_n_level.cu). This is the "shadow box":
 * AMRMPICoupling's whole job is to keep a small local AMRLevel-shaped buffer covering exactly
 * this range filled with the real, possibly-remote coarse data every step, so the existing,
 * completely unmodified AMRCoupling<Q>::apply_c2f/apply_f2c can run against it exactly as in
 * the single-rank case. */
inline void amr_shadow_box_global_range(const GridConfig& coarse_cfg, const onika::math::AABB& patch_bounds,
                                        onika::math::IJK& lo, onika::math::IJK& hi) {
  const double dx = (coarse_cfg.bounds_.bmax.x - coarse_cfg.bounds_.bmin.x) / double(coarse_cfg.dims_.i - 1);
  auto idx = [&](double p, double origin) { return int(std::lround((p - origin) / dx)); };
  lo.i = idx(patch_bounds.bmin.x, coarse_cfg.bounds_.bmin.x) - 1;
  lo.j = idx(patch_bounds.bmin.y, coarse_cfg.bounds_.bmin.y) - 1;
  lo.k = idx(patch_bounds.bmin.z, coarse_cfg.bounds_.bmin.z) - 1;
  hi.i = idx(patch_bounds.bmax.x, coarse_cfg.bounds_.bmin.x) + 1;
  hi.j = idx(patch_bounds.bmax.y, coarse_cfg.bounds_.bmin.y) + 1;
  hi.k = idx(patch_bounds.bmax.z, coarse_cfg.bounds_.bmin.z) + 1;
}

/** @brief Builds a GridConfig for the shadow box itself (§ amr_shadow_box_global_range),
 * suitable for passing straight into AMRNLevel::build_level() (owner rank only, MPI_COMM_SELF)
 * exactly the way a patch level is built -- same machinery, no new grid-construction code. */
inline GridConfig amr_shadow_box_grid_config(const GridConfig& coarse_cfg, const onika::math::IJK& lo,
                                             const onika::math::IJK& hi) {
  const double dx = (coarse_cfg.bounds_.bmax.x - coarse_cfg.bounds_.bmin.x) / double(coarse_cfg.dims_.i - 1);
  GridConfig cfg;
  cfg.periodic_ = {false, false, false};
  cfg.dims_ = {hi.i - lo.i + 1, hi.j - lo.j + 1, hi.k - lo.k + 1};
  cfg.bounds_.bmin = coarse_cfg.bounds_.bmin + onika::math::Vec3d{double(lo.i), double(lo.j), double(lo.k)} * dx;
  cfg.bounds_.bmax = coarse_cfg.bounds_.bmin + onika::math::Vec3d{double(hi.i), double(hi.j), double(hi.k)} * dx;
  return cfg;
}

/** @brief Phase 4 v1 (docs/AMR_PLAN.md): point-to-point MPI exchange standing in for
 * AMRCoupling<Q> when a patch's coarse donor nodes may live on a different rank than the
 * patch's own owner. Deliberately reuses AMRCoupling's physics completely unmodified: the
 * owner rank keeps a small "shadow" AMRLevel (see amr_shadow_box_global_range/
 * amr_shadow_box_grid_config) built and stepped exactly like a normal coarse level would be,
 * except its fields are filled from the REAL coarse data (local copy where the owner already
 * has it, MPI from each remote donor otherwise) instead of from its own physics -- then a
 * perfectly ordinary AMRCoupling<Q>(shadow, fine) runs against it, and the resolved F2C result
 * is scattered back out to the real donors. No duplicated interpolation/correction math
 * anywhere in this file.
 *
 * Node ownership (which rank owns the N-th node in the shadow box) is resolved once at
 * construction, identically and independently on EVERY involved rank, from shared config
 * (amr_rank_owner.hpp) -- no index-list handshake, only the actual field data is ever
 * exchanged, every coarse step. Assumes cart-rank == world-rank (see amr_rank_owner.hpp's own
 * comment: matches LBMGhostManager's existing, pre-existing assumption, not a new risk).
 *
 * Out of scope for v1 (docs/AMR_PLAN.md Phase 4 v1 plan): more than one patch, a patch split
 * across multiple owner ranks, and combining this with dynamic remeshing (a patch migrating
 * across rank ownership) -- amr_n_level.cu's execute() refuses remesh_freq>0 together with
 * mpi_size>1 rather than attempting any of that. */
template <int Q>
struct AMRMPICoupling {
  static constexpr int kC2FDoubles = 8 + Q;  // rho_prev, u_prev.xyz, rho_now, u_now.xyz, f_now[Q]
  static constexpr int kF2CDoubles = Q;      // resolved f_[Q]
  static constexpr int kTagC2F = 7001;
  static constexpr int kTagF2C = 7002;

  bool is_owner_ = false;
  bool is_donor_ = false;  // true on a non-owner rank that owns part of the shadow footprint
  int owner_rank_ = 0;      // *patch_owner_rank's CART rank (see constructor), not its world rank
  int my_rank_ = 0;         // my own CART rank
  MPI_Comm comm_ = MPI_COMM_WORLD;  // the CART communicator -- every Isend/Irecv/rank comparison
                                    // in this struct uses it, never the world communicator, so
                                    // "rank" always means the same thing make_domain()'s own
                                    // ghost-exchange table (and hence actual data ownership)
                                    // means by it. See constructor for the one-time
                                    // world-rank -> cart-rank resolution this requires.

  /** @brief One peer's worth of node indices. On the owner, `shadow_local_idx_` is always
   * populated (where in the shadow box this peer's data lands) and `local_coarse_idx_` is
   * populated only for the owner's OWN entry (owner_local_, no MPI needed there). On a donor,
   * only `local_coarse_idx_` is populated (where in ITS OWN level 0 grid to read/write). */
  struct PeerNodes {
    int rank_ = -1;
    std::vector<int> shadow_local_idx_;
    std::vector<int> local_coarse_idx_;
  };

  std::vector<PeerNodes> owner_remote_;  // owner-only: one entry per distinct remote donor rank
  PeerNodes owner_local_;                // owner-only: the owner's own locally-held portion
  PeerNodes donor_self_;                 // non-owner: this rank's own contribution (if any)

  onika::math::IJK shadow_lo_, shadow_hi_;  // the global range this coupling covers

  // `patch_owner_rank`: a WORLD rank (`*mpi`'s numbering, matching mpirun -n N semantics and
  // what a user writes in the .msp file). `world_comm`/`cart_comm`: the world communicator and
  // the REAL Cartesian communicator level 0's own domain decomposition uses (captured via
  // amr_n_level.cu's build_level() `out_subgrid` parameter) -- NOT a second, separately-built
  // Cartesian communicator: MPI_Cart_create's `reorder` is not guaranteed to number ranks the
  // same way twice, even given identical inputs, so reusing the exact communicator object
  // make_domain() already built is what keeps "who AMRMPICoupling thinks owns a node" and "who
  // actually holds its data" in agreement (docs/AMR_PLAN.md Phase 4 v1 "n>2 ranks" fix: an
  // earlier version assumed cart-rank == world-rank with a hand-rolled numbering formula,
  // which happened to hold for mpi_size==2 but silently didn't for mpi_size>=3).
  AMRMPICoupling(int patch_owner_rank, MPI_Comm world_comm, MPI_Comm cart_comm, const GridConfig& coarse_cfg,
                const onika::math::IJK& mpi_dims, LBMGrid& my_level0_grid, LBMGrid* shadow_grid,
                const onika::math::AABB& patch_bounds) {
    MPI_Comm_rank(cart_comm, &my_rank_);
    comm_ = cart_comm;

    // One-time, O(mpi_size) resolution of the owner's WORLD rank into ITS OWN cart rank: every
    // rank reports its own (world_rank -> cart_rank) pair via an Allgather over world_comm,
    // then every rank looks up patch_owner_rank's entry. Cheap (a handful of ints, once at
    // construction), and the only place this struct ever needs world_comm at all.
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);
    std::vector<int> cart_rank_by_world(world_size, -1);
    MPI_Allgather(&my_rank_, 1, MPI_INT, cart_rank_by_world.data(), 1, MPI_INT, world_comm);
    owner_rank_ = cart_rank_by_world[patch_owner_rank];

    is_owner_ = (my_rank_ == owner_rank_);

    amr_shadow_box_global_range(coarse_cfg, patch_bounds, shadow_lo_, shadow_hi_);

    // Group every node in the shadow footprint by owning rank, in (gx,gy,gz) ascending order
    // (the loop's own natural order) -- fixed/canonical so both sides of any exchange agree on
    // node ordering without ever exchanging the index list itself.
    std::map<int, std::vector<std::array<int, 3>>> by_rank;
    for (int gx = shadow_lo_.i; gx <= shadow_hi_.i; gx++) {
      for (int gy = shadow_lo_.j; gy <= shadow_hi_.j; gy++) {
        for (int gz = shadow_lo_.k; gz <= shadow_hi_.k; gz++) {
          const int owner = amr_rank_owning_global_node(coarse_cfg.dims_, mpi_dims, cart_comm, gx, gy, gz);
          by_rank[owner].push_back({gx, gy, gz});
        }
      }
    }

    for (auto& [rank, pts] : by_rank) {
      if (is_owner_) {
        PeerNodes pn;
        pn.rank_ = rank;
        pn.shadow_local_idx_.reserve(pts.size());
        if (rank == my_rank_) pn.local_coarse_idx_.reserve(pts.size());
        for (auto& p : pts) {
          const Point3D gp = {p[0], p[1], p[2]};
          // The shadow grid's own global index 0 is its OWN bounds_.bmin (shadow_lo_ in the
          // coarse domain's numbering), not the coarse domain's global index 0 -- `gp` must be
          // shifted into the shadow grid's own global-index space before convert<Area::Local>,
          // exactly as every other cross-grid lookup in this codebase goes through physical
          // position as the bridge (see amr_coupling.hpp's compute_c2f_entry) rather than
          // reusing one grid's raw index numbers on a different grid.
          const Point3D sg = {int(p[0] - shadow_lo_.i), int(p[1] - shadow_lo_.j), int(p[2] - shadow_lo_.k)};
          const Point3D sp = shadow_grid->template convert<Area::Local>(sg);
          pn.shadow_local_idx_.push_back((*shadow_grid)(sp[0], sp[1], sp[2]));
          if (rank == my_rank_) {
            const Point3D lp = my_level0_grid.template convert<Area::Local>(gp);
            pn.local_coarse_idx_.push_back(my_level0_grid(lp[0], lp[1], lp[2]));
          }
        }
        if (rank == my_rank_) {
          owner_local_ = std::move(pn);
        } else {
          owner_remote_.push_back(std::move(pn));
        }
      } else if (rank == my_rank_) {
        is_donor_ = true;
        PeerNodes pn;
        pn.rank_ = owner_rank_;
        pn.local_coarse_idx_.reserve(pts.size());
        for (auto& p : pts) {
          const Point3D gp = {p[0], p[1], p[2]};
          const Point3D lp = my_level0_grid.template convert<Area::Local>(gp);
          pn.local_coarse_idx_.push_back(my_level0_grid(lp[0], lp[1], lp[2]));
        }
        donor_self_ = std::move(pn);
      }
      // else: neither owner nor a donor for this peer's nodes -- nothing to do, ever.
    }
  }

  /** @brief Called on EVERY rank, every coarse step, right after this rank's own level 0 has
   * been stepped, compute_macro()'d (now), with its prev (rho,u) snapshot already captured --
   * a no-op unless this rank is a donor. Packs and ships this rank's contribution to the owner. */
  void donor_send_c2f(AMRLevel<Q>& my_level0, const double* rho_prev0, const FieldView<3>& u_prev0) {
    if (!is_donor_) return;
    const double* rho_now0 = my_level0.fields_.densities();
    FieldView<3> u_now0 = my_level0.fields_.flux();
    FieldView<Q> f_now0 = my_level0.fields_.distributions();

    const size_t n = donor_self_.local_coarse_idx_.size();
    std::vector<double> buf(n * kC2FDoubles);
    for (size_t k = 0; k < n; k++) {
      const int lidx = donor_self_.local_coarse_idx_[k];
      double* p = buf.data() + k * kC2FDoubles;
      p[0] = rho_prev0[lidx];
      p[1] = u_prev0(lidx, 0);
      p[2] = u_prev0(lidx, 1);
      p[3] = u_prev0(lidx, 2);
      p[4] = rho_now0[lidx];
      p[5] = u_now0(lidx, 0);
      p[6] = u_now0(lidx, 1);
      p[7] = u_now0(lidx, 2);
      for (int q = 0; q < Q; q++) p[8 + q] = f_now0(lidx, q);
    }
    MPI_Request req;
    MPI_Isend(buf.data(), int(buf.size()), MPI_DOUBLE, owner_rank_, kTagC2F, comm_, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);  // small payload (docs/AMR_PLAN.md Phase 4 v1: a few
                                        // hundred KB at most) -- simplest correct option for v1,
                                        // not pipelined/overlapped with compute.
  }

  /** @brief Owner-only (no-op elsewhere): fills `shadow`'s fields_ (rho_now/u_now/f_now) and
   * the caller-provided prev buffers from the owner's own local data plus one blocking receive
   * per remote donor (each donor_send_c2f() call above). Call once per coarse step, before any
   * apply_c2f() round -- the data is constant across both Phase-2 sub-cycling rounds, only
   * `alpha` differs between them, so this does not need to run twice. */
  void owner_receive_c2f(AMRLevel<Q>& my_level0, const double* rho_prev0, const FieldView<3>& u_prev0,
                        AMRLevel<Q>& shadow, double* shadow_rho_prev_out, FieldView<3> shadow_u_prev_out) {
    if (!is_owner_) return;
    const double* rho_now0 = my_level0.fields_.densities();
    FieldView<3> u_now0 = my_level0.fields_.flux();
    FieldView<Q> f_now0 = my_level0.fields_.distributions();
    double* shadow_rho = shadow.fields_.densities();
    FieldView<3> shadow_u = shadow.fields_.flux();
    FieldView<Q> shadow_f = shadow.fields_.distributions();

    auto write_one = [&](int sidx, const double* p) {
      shadow_rho_prev_out[sidx] = p[0];
      shadow_u_prev_out(sidx, 0) = p[1];
      shadow_u_prev_out(sidx, 1) = p[2];
      shadow_u_prev_out(sidx, 2) = p[3];
      shadow_rho[sidx] = p[4];
      shadow_u(sidx, 0) = p[5];
      shadow_u(sidx, 1) = p[6];
      shadow_u(sidx, 2) = p[7];
      for (int q = 0; q < Q; q++) shadow_f(sidx, q) = p[8 + q];
    };

    double local_buf[kC2FDoubles];
    for (size_t k = 0; k < owner_local_.shadow_local_idx_.size(); k++) {
      const int lidx = owner_local_.local_coarse_idx_[k];
      local_buf[0] = rho_prev0[lidx];
      local_buf[1] = u_prev0(lidx, 0);
      local_buf[2] = u_prev0(lidx, 1);
      local_buf[3] = u_prev0(lidx, 2);
      local_buf[4] = rho_now0[lidx];
      local_buf[5] = u_now0(lidx, 0);
      local_buf[6] = u_now0(lidx, 1);
      local_buf[7] = u_now0(lidx, 2);
      for (int q = 0; q < Q; q++) local_buf[8 + q] = f_now0(lidx, q);
      write_one(owner_local_.shadow_local_idx_[k], local_buf);
    }

    for (auto& peer : owner_remote_) {
      const size_t n = peer.shadow_local_idx_.size();
      std::vector<double> buf(n * kC2FDoubles);
      MPI_Request req;
      MPI_Irecv(buf.data(), int(buf.size()), MPI_DOUBLE, peer.rank_, kTagC2F, comm_, &req);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
      for (size_t k = 0; k < n; k++) write_one(peer.shadow_local_idx_[k], buf.data() + k * kC2FDoubles);
    }
  }

  /** @brief Owner-only (no-op elsewhere): after AMRCoupling<Q>::apply_f2c(shadow, fine) has
   * written the resolved f_ into `shadow`'s own fields_, scatters it back out -- local copy
   * into the owner's own level 0, plus one send per remote donor. Call once per coarse step,
   * after both Phase-2 sub-cycling rounds. */
  void owner_send_f2c(AMRLevel<Q>& shadow, AMRLevel<Q>& my_level0) {
    if (!is_owner_) return;
    FieldView<Q> shadow_f = shadow.fields_.distributions();
    FieldView<Q> f0 = my_level0.fields_.distributions();

    for (size_t k = 0; k < owner_local_.shadow_local_idx_.size(); k++) {
      const int sidx = owner_local_.shadow_local_idx_[k];
      const int lidx = owner_local_.local_coarse_idx_[k];
      for (int q = 0; q < Q; q++) f0(lidx, q) = shadow_f(sidx, q);
    }

    std::vector<MPI_Request> reqs(owner_remote_.size());
    std::vector<std::vector<double>> bufs(owner_remote_.size());
    for (size_t pi = 0; pi < owner_remote_.size(); pi++) {
      auto& peer = owner_remote_[pi];
      const size_t n = peer.shadow_local_idx_.size();
      bufs[pi].resize(n * kF2CDoubles);
      for (size_t k = 0; k < n; k++) {
        const int sidx = peer.shadow_local_idx_[k];
        double* p = bufs[pi].data() + k * kF2CDoubles;
        for (int q = 0; q < Q; q++) p[q] = shadow_f(sidx, q);
      }
      MPI_Isend(bufs[pi].data(), int(bufs[pi].size()), MPI_DOUBLE, peer.rank_, kTagF2C, comm_, &reqs[pi]);
    }
    for (auto& req : reqs) MPI_Wait(&req, MPI_STATUS_IGNORE);
  }

  /** @brief Called on EVERY rank, every coarse step, after owner_send_f2c() -- a no-op unless
   * this rank is a donor. Receives the F2C-resolved data for this rank's own contributed nodes
   * and applies it directly to this rank's own level 0 distributions, exactly like a local
   * AMRCoupling::apply_f2c would. */
  void donor_receive_f2c(AMRLevel<Q>& my_level0) {
    if (!is_donor_) return;
    FieldView<Q> f0 = my_level0.fields_.distributions();
    const size_t n = donor_self_.local_coarse_idx_.size();
    std::vector<double> buf(n * kF2CDoubles);
    MPI_Request req;
    MPI_Irecv(buf.data(), int(buf.size()), MPI_DOUBLE, owner_rank_, kTagF2C, comm_, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    for (size_t k = 0; k < n; k++) {
      const int lidx = donor_self_.local_coarse_idx_[k];
      const double* p = buf.data() + k * kF2CDoubles;
      for (int q = 0; q < Q; q++) f0(lidx, q) = p[q];
    }
  }
};

}  // namespace hippoLBM
