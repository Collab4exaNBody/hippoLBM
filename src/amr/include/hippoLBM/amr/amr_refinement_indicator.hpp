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

#include <onika/parallel/parallel_for.h>

#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/grid.hpp>

namespace hippoLBM {

/** @brief Flow-driven refinement indicator (docs/AMR_PLAN.md Phase 6, sub-problem 1): the
 * classic Q-criterion, Q = 0.5*(|Omega|^2_F - |S|^2_F), where S and Omega are the symmetric
 * (strain-rate) and antisymmetric (rotation-rate) parts of the velocity gradient tensor
 * du_i/dx_j, computed by central finite differences on the already-macro_variables-filled
 * velocity field `u` (not re-derived from the distributions).
 *
 * Chosen over a simpler indicator (|u|, raw vorticity magnitude, or |grad u|) because it is
 * EXACTLY ZERO, analytically, in plane Couette flow (pure simple shear: for u_x = gamma*z, the
 * only nonzero gradient components are du_x/dz = -du_z/dx-equivalent, giving |S|^2 = |Omega|^2
 * = 0.5*gamma^2 identically) -- so on top of our already-validated Couette test case, any
 * nonzero Q unambiguously signals a genuine flow feature (an obstacle's wake, a vortex), not
 * just background shear that a simpler threshold-based indicator would also flag everywhere.
 * This is purely diagnostic for now (sub-problem 1 of Phase 6: compute the indicator) -- it
 * does not drive any remeshing decision (sub-problem 2, not implemented).
 *
 * Central differences require both neighbors along every axis to exist: callers should only
 * evaluate this over Traversal::Real (ghost_layer=2 guarantees both neighbors of a Real node
 * are defined, whether they're further Real nodes or the first ghost layer). */
template <int Q>
struct QCriterionFunc {
  LBMGrid g_;
  const FieldView<3> u_;    // velocity, already filled by macro_variables
  double* const q_out_;     // one entry per node, same indexing as g_(x,y,z)

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int x, int y, int z) const {
    const int idx = g_(x, y, z);
    const double inv_2dx = 1.0 / (2.0 * g_.dx_);

    const int ixp = g_(x + 1, y, z), ixm = g_(x - 1, y, z);
    const int iyp = g_(x, y + 1, z), iym = g_(x, y - 1, z);
    const int izp = g_(x, y, z + 1), izm = g_(x, y, z - 1);

    const double duxdx = (u_(ixp, 0) - u_(ixm, 0)) * inv_2dx;
    const double duydx = (u_(ixp, 1) - u_(ixm, 1)) * inv_2dx;
    const double duzdx = (u_(ixp, 2) - u_(ixm, 2)) * inv_2dx;

    const double duxdy = (u_(iyp, 0) - u_(iym, 0)) * inv_2dx;
    const double duydy = (u_(iyp, 1) - u_(iym, 1)) * inv_2dx;
    const double duzdy = (u_(iyp, 2) - u_(iym, 2)) * inv_2dx;

    const double duxdz = (u_(izp, 0) - u_(izm, 0)) * inv_2dx;
    const double duydz = (u_(izp, 1) - u_(izm, 1)) * inv_2dx;
    const double duzdz = (u_(izp, 2) - u_(izm, 2)) * inv_2dx;

    // symmetric strain-rate tensor S_ij = 0.5*(du_i/dx_j + du_j/dx_i)
    const double sxx = duxdx, syy = duydy, szz = duzdz;
    const double sxy = 0.5 * (duxdy + duydx);
    const double sxz = 0.5 * (duxdz + duzdx);
    const double syz = 0.5 * (duydz + duzdy);
    const double s_frob2 = sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);

    // antisymmetric rotation-rate tensor Omega_ij = 0.5*(du_i/dx_j - du_j/dx_i); diagonal is 0
    const double oxy = 0.5 * (duxdy - duydx);
    const double oxz = 0.5 * (duxdz - duzdx);
    const double oyz = 0.5 * (duydz - duzdy);
    const double omega_frob2 = 2.0 * (oxy * oxy + oxz * oxz + oyz * oyz);

    q_out_[idx] = 0.5 * (omega_frob2 - s_frob2);
  }
};

}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::QCriterionFunc<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
