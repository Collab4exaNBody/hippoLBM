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

#define FLUIDE_ -1

namespace hippoLBM {
template <int Q, Traversal TR>
struct fuse_macro_bgk {
  const int* __restrict__ levels_;  // traversal level (0 inside, 0 1 Real, 0 1 2 Extend, 0 1 2 3 All)
  const onika::math::Vec3d Fext_;   // external force term
  const FieldView<3> m1_;           // flux (velocity) field, written then read back for collision
  int* const __restrict__ obst_;    // obstacle field
  const FieldView<Q> f_;            // distribution functions
  double* const __restrict__ m0_;   // density field
  const double tau_;                // BGK relaxation time

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    // Cache f_(idx,*) in registers: read once here, reuse in the collision loop below
    // instead of re-fetching from (global) memory a second time.
    double fi[Q];

    // --- macro_variables ---
    double rho = 0.0;
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;
    if (obst_[idx] >= FLUIDE_) {
      stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, Q>([&]<typename coeff>(int iLB) {
        const double s = f_(idx, iLB);
        fi[iLB] = s;
        ux += s * coeff::ex;
        uy += s * coeff::ey;
        uz += s * coeff::ez;
        rho += s;
      });
      if (rho > 1.0e-14) {
        ux /= rho;
        uy /= rho;
        uz /= rho;
      }
      m0_[idx] = rho;
      m1_(idx, 0) = ux;
      m1_(idx, 1) = uy;
      m1_(idx, 2) = uz;
    } else {
      stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, Q>(
          [&]<typename coeff>(int iLB) { fi[iLB] = f_(idx, iLB); });
      m1_(idx, 0) = 0;
      m1_(idx, 1) = 0;
      m1_(idx, 2) = 0;
    }

    // --- bgk collision ---
    const bool update = check_level<TR>(levels_[idx]) && (obst_[idx] == FLUIDE_);
    const double u_squ = (ux * ux + uy * uy + uz * uz);
    stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, Q>([&]<typename coeff>(int iLB) {
      const double fiLB = fi[iLB];
      double ef = coeff::ex * Fext_.x + coeff::ey * Fext_.y + coeff::ez * Fext_.z;
      double eu = coeff::ex * ux + coeff::ey * uy + coeff::ez * uz;
      double feq = coeff::w * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
      f_(idx, iLB) = fiLB + update * ((feq - fiLB) / tau_ + 3. * rho * coeff::w * ef);
    });
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal Tr>
struct ParallelForFunctorTraits<hippoLBM::fuse_macro_bgk<Q, Tr>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
