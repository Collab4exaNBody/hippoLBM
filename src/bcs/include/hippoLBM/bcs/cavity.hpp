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

/** @brief Coefficients for the cavity boundary condition according to the dimension and side. */
template <int DIM, Side S, int Q>
struct cavity {};
template <int dim, Side dir, int Q>
struct cavity_coeff {};

#define SPECIFIC_CAVITY_COEFF(DIM, S, Q, ...)          \
  template <>                                          \
  struct cavity_coeff<DIM, S, Q> {                     \
    std::integer_sequence<int, __VA_ARGS__> fid_ = {}; \
  };

SPECIFIC_CAVITY_COEFF(DIMZ, Side::Left, 19, 5, 14, 11, 18, 15)
SPECIFIC_CAVITY_COEFF(DIMZ, Side::Right, 19, 6, 13, 12, 17, 16)

/** @brief Cavity boundary condition */
template <int Dim, Side S>
struct cavity<Dim, S, 19> {
  static constexpr int Q = 19;  // number of discrete velocities
  static constexpr int Un = 5;  // number of unknowns per point in the cavity boundary condition
  double coeff_[Un];            // coefficients for the cavity boundary condition

  /** @brief Compute the coefficients for the cavity boundary condition
   * @param ux The x-component of the velocity at the boundary.
   * @param uy The y-component of the velocity at the boundary.
   * @param uz The z-component of the velocity at the boundary.
   * @param lx The local grid size in the x-dimension.
   * @param ly The local grid size in the y-dimension.
   * @param lz The local grid size in the z-dimension.
   */
  void compute_coeff(double ux, double uy, double uz, int lx, int ly, int lz) {
    const cavity_coeff<DIMZ, S, Q> c_coeff;
    double L = 0;
    if constexpr (Dim == DIMZ) L = lx;
    if constexpr (Dim == DIMY) L = ly;
    if constexpr (Dim == DIMZ) L = lz;
    const double uxx = ux * (1 + 0.5 / (L - 1));
    const double uyy = uy * (1 + 0.5 / (L - 1));
    const double uzz = uz * (1 + 0.5 / (L - 1));
    int idx = 0;
    stencil::for_specific_dirs_impl<typename LBMScheme<Q>::Coefficients>(
        [&]<typename coeff, int iLB> {
          coeff_[idx++] = 6. * coeff::w * (coeff::ex * uxx + coeff::ey * uyy + coeff::ez * uzz);
        },
        c_coeff.fid_);

    assert(idx == Un && "The number of computed coefficients does not match the expected number of unknowns.");
  }

  /** @brief Apply the cavity boundary condition
   * @param idx The index of the point.
   * @param obst The obstacle array.
   * @param fi The field view for the unknowns.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<Un>& fi) const {
    if (obst[idx] == FLUIDE_) {
      for (int i = 0; i < Un; i++) {
        fi(idx, i) += coeff_[i];
      }
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Dim, hippoLBM::Side S, int Q>
struct ParallelForFunctorTraits<hippoLBM::cavity<Dim, S, Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
