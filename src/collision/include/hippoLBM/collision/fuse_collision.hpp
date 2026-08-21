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

#include <hippoLBM/collision/bgk.hpp>
#include <hippoLBM/collision/macro_variables.hpp>
#include <hippoLBM/collision/mrt.hpp>
#include <hippoLBM/grid/field_view.hpp>

namespace hippoLBM {

struct BGKCollisionModel {
  template <int Q>
  ONIKA_HOST_DEVICE_FUNC static inline void collide(int idx, bool update, const FieldView<Q>& f, double rho,
                                                     double ux, double uy, double uz, const onika::math::Vec3d& Fext,
                                                     double tau) {
    bgk_collide<Q>(idx, update, f, rho, ux, uy, uz, Fext, tau);
  }
};

struct MRTCollisionModel {
  template <int Q>
  ONIKA_HOST_DEVICE_FUNC static inline void collide(int idx, bool update, const FieldView<Q>& f, double rho,
                                                     double /*ux*/, double /*uy*/, double /*uz*/,
                                                     const onika::math::Vec3d& Fext, double tau) {
    mrt_collide<Q>(idx, update, f, rho, Fext, tau);
  }
};

template <int Q, Traversal TR, typename CollisionModel>
struct fuse_macro_collision {
  const int* __restrict__ levels_;  // traversal level (0 inside, 0 1 Real, 0 1 2 Extend, 0 1 2 3 All)
  const onika::math::Vec3d Fext_;   // external force term
  const FieldView<3> m1_;           // flux (velocity) field, written then read back for collision
  int* const __restrict__ obst_;    // obstacle field
  const FieldView<Q> f_;            // distribution functions
  double* const __restrict__ m0_;   // density field
  const double tau_;                // collision relaxation time

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    // macro_variables runs over Traversal::All unconditionally, like the standalone operator
    compute_macro_variables<Q>(idx, obst_, f_, m0_, m1_);
    const bool update = check_level<TR>(levels_[idx]) && (obst_[idx] == FLUIDE_);
    CollisionModel::template collide<Q>(idx, update, f_, m0_[idx], m1_(idx, 0), m1_(idx, 1), m1_(idx, 2), Fext_,
                                        tau_);
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal Tr, typename CollisionModel>
struct ParallelForFunctorTraits<hippoLBM::fuse_macro_collision<Q, Tr, CollisionModel>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
