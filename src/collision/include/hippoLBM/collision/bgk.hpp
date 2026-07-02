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
/**
 * @brief A functor for collision operations in the lattice Boltzmann method.
 */
template <int Q, Traversal TR>
struct bgk {
  const int* __restrict__ levels_;  // It contains the traversal level (0 inside,
  // 0 1 Real,
  // 0 1 2 Extend,
  // and 0 1 2 3 All
  const onika::math::Vec3d m_Fext_;  // External force term, used in the computation of macroscopic variables.
  const FieldView<3> m1_;            // The field view for the first-order moments (momentum).
  int* const __restrict__ obst_;     // Pointer to the obstacle field.
  const FieldView<Q> f_;             // The field view for the distribution functions.
  double* const __restrict__ m0_;    // Pointer to the density field (zeroth-order moment).
  const double tau_;                 // Relaxation time for the BGK collision model.

  /**
   * @brief Operator for performing collision operations at a given index.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    bool update = check_level<TR>(levels_[idx]) && (obst_[idx] == FLUIDE_);
    const double rho = m0_[idx];
    const double ux = m1_(idx, 0);
    const double uy = m1_(idx, 1);
    const double uz = m1_(idx, 2);
    const double u_squ = (ux * ux + uy * uy + uz * uz);

    stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, Q>([&]<typename coeff, int iLB> {
      double& fiLB = f_(idx, iLB);
      double ef = coeff::ex * m_Fext_.x + coeff::ey * m_Fext_.y + coeff::ez * m_Fext_.z;
      double eu = coeff::ex * ux + coeff::ey * uy + coeff::ez * uz;
      double feq = coeff::w * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
      fiLB += update * ((feq - fiLB) / tau_ + 3. * rho * coeff::w * ef);
    });
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal Tr>
struct ParallelForFunctorTraits<hippoLBM::bgk<Q, Tr>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
