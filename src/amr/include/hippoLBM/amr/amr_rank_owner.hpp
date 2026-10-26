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

#include <onika/math/basic_types.h>

namespace hippoLBM {

/** @brief Phase 4 (docs/AMR_PLAN.md): given level 0's global node counts, its Cartesian
 * process grid dims, and the ACTUAL Cartesian communicator load_balancing()/make_domain()
 * built for it (`level0_cart_comm`, must be the very same communicator object level 0's own
 * domain decomposition uses -- see amr_n_level.cu's build_level() `out_subgrid` parameter),
 * returns which rank (numbered WITHIN that same cart communicator, not the world
 * communicator) owns a given global coarse node -- by inverting load_balancing()'s own
 * block-split formula (make_domain.hpp: `set_block_size`/`inf` computation) to get the
 * OWNING COORDINATE, then resolving that coordinate to a rank via a real MPI_Cart_rank call
 * on the real communicator, rather than assuming any particular rank-numbering convention.
 *
 * An earlier version of this function instead ASSUMED cart-rank == world-rank and derived the
 * rank from a hand-rolled row-major formula (matching LBMGhostManager's existing, pre-existing
 * assumption, grid/ghost_manager.hpp) -- empirically fine for mpi_size==2, but it produced a
 * silently wrong owner for some nodes at mpi_size==3 and mpi_size==4, causing exactly one
 * Isend/Irecv pair to never find its match: MPI buffers a few such stray messages via its
 * eager protocol without complaint, so the run kept making (slower and slower) progress for
 * a couple dozen iterations before actually deadlocking once that buffering was exhausted --
 * see docs/AMR_PLAN.md Phase 4 v1 "n>2 ranks" status entry for the full diagnosis. Querying
 * MPI_Cart_rank directly removes the assumption instead of trying to match it. */
inline int amr_rank_owning_global_node(const onika::math::IJK& global_dims, const onika::math::IJK& mpi_dims,
                                       MPI_Comm level0_cart_comm, int gx, int gy, int gz) {
  // Mirrors load_balancing()'s own per-axis split: all ranks but the last in a dimension own
  // exactly `global_dims/mpi_dims` nodes; the last one absorbs the remainder. Inverting: the
  // owning coordinate along one axis is floor(g / block), clamped to mpi_dims-1 so indices
  // inside the (possibly larger) last block still resolve to the last rank.
  auto coord1d = [](int g, int n, int d) -> int {
    if (d <= 1) return 0;
    const int block = n / d;
    int c = (block > 0) ? (g / block) : (d - 1);
    if (c > d - 1) c = d - 1;
    if (c < 0) c = 0;
    return c;
  };
  int coords[3] = {coord1d(gx, global_dims.i, mpi_dims.i), coord1d(gy, global_dims.j, mpi_dims.j),
                   coord1d(gz, global_dims.k, mpi_dims.k)};
  int rank = 0;
  MPI_Cart_rank(level0_cart_comm, coords, &rank);
  return rank;
}

}  // namespace hippoLBM
