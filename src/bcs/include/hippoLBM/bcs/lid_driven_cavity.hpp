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

#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/field_view.hpp>

namespace hippoLBM {

template <int Q>
struct LidDrivenCavityBCsFunctor;

/** @brief Cavity boundary condition */
template <>
struct LidDrivenCavityBCsFunctor<19> {
  static constexpr int Q = 19;  // number of discrete velocities
  onika::math::Vec3d U_;
  const FieldView<Q> F_;                // The field view for the distribution functions.
  int* const __restrict__ obst_;        // Pointer to the obstacle field.
  const int* const __restrict__ ex_;    // Pointer to an array of integers for X-direction.
  const int* const __restrict__ ey_;    // Pointer to an array of integers for Y-direction.
  const int* const __restrict__ ez_;    // Pointer to an array of integers for Z-direction.
  const double* const __restrict__ w_;  // Pointer to an array of doubles for the weights of the discrete velocity
                                        // directions.

  ONIKA_HOST_DEVICE_FUNC inline double compute_rho(int idx) const {
    double rho = 0.0;
    for (int iLB = 0; iLB < Q; iLB++) {
      rho += F_(idx, iLB);
    }
    return rho;
  }

  /** @brief Apply the cavity boundary condition
   * @param idx The index of the point.
   * @param obst The obstacle array.
   * @param fi The field view for the unknowns.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    if (obst_[idx] == FLUIDE_) {
      const double rho = compute_rho(idx);
      const double u_squ = onika::math::dot(U_, U_);

      for (int iLB = 0; iLB < Q; iLB++) {
        const int& exiLB = ex_[iLB];
        const int& eyiLB = ey_[iLB];
        const int& eziLB = ez_[iLB];
        const double& wiLB = w_[iLB];
        double eu = exiLB * U_.x + eyiLB * U_.y + eziLB * U_.z;
        F_(idx, iLB) = wiLB * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * u_squ);
      }
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::LidDrivenCavityBCsFunctor<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
