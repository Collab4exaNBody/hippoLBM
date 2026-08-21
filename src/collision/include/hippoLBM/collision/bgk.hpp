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

#include <hippoLBM/grid/field_view.hpp>

// TODO: consider to move this file to a more general location, since it is used by both grid and domain
#define FLUIDE_ -1

namespace hippoLBM {

template <int Q>
ONIKA_HOST_DEVICE_FUNC inline void bgk_collide(int idx, bool update, const FieldView<Q>& f, double rho, double ux,
                                               double uy, double uz, const onika::math::Vec3d& Fext, double tau) {
  const double u_squ = (ux * ux + uy * uy + uz * uz);
  stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, Q>([&]<typename coeff>(int iLB) {
    double& fiLB = f(idx, iLB);
    double ef = coeff::ex * Fext.x + coeff::ey * Fext.y + coeff::ez * Fext.z;
    double eu = coeff::ex * ux + coeff::ey * uy + coeff::ez * uz;
    double feq = coeff::w * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
    fiLB += update * ((feq - fiLB) / tau + 3. * rho * coeff::w * ef);
  });
}

template <int Q, Traversal TR>
struct BGKLauncher {
  const int* __restrict__ levels_;  // It contains the traversal level (0 inside,
  // 0 1 Real,
  // 0 1 2 Extend,
  // and 0 1 2 3 All
  const onika::math::Vec3d Fext_;  // External force term, used in the computation of macroscopic variables.
  const FieldView<3> m1_;          // The field view for the first-order moments (momentum).
  int* const __restrict__ obst_;   // Pointer to the obstacle field.
  const FieldView<Q> f_;           // The field view for the distribution functions.
  double* const __restrict__ m0_;  // Pointer to the density field (zeroth-order moment).
  const double tau_;               // Relaxation time for the BGK collision model.

  /**
   * @brief Runs the BGK collision at a given index.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    const bool update = check_level<TR>(levels_[idx]) && (obst_[idx] == FLUIDE_);
    bgk_collide<Q>(idx, update, f_, m0_[idx], m1_(idx, 0), m1_(idx, 1), m1_(idx, 2), Fext_, tau_);
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal Tr>
struct ParallelForFunctorTraits<hippoLBM::BGKLauncher<Q, Tr>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
