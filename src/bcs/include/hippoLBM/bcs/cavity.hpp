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

template <>
struct cavity_coeff<DIMZ, Side::Left, 19> {
  static constexpr std::integer_sequence<int, 5, 14, 11, 18, 15> fid_ = {};
};
template <>
struct cavity_coeff<DIMZ, Side::Right, 19> {
  static constexpr std::integer_sequence<int, 6, 13, 12, 17, 16> fid_ = {};
};

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
   * @param w The weights for the discrete velocities.
   * @param ex The x-components of the discrete velocities directions.
   * @param ey The y-components of the discrete velocities directions.
   * @param ez The z-components of the discrete velocities directions.
   * @param lx The local grid size in the x-dimension.
   * @param ly The local grid size in the y-dimension.
   * @param lz The local grid size in the z-dimension.
   */
  void compute_coeff(double ux, double uy, double uz, const double* const w, const int* ex, const int* ey,
                     const int* ez, int lx, int ly, int lz) {
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

    /*
  for (int i = 0; i < Un; i++) {
  const int fid = c_coeff.fid_[i];
  coeff_[i] = 6. * w[fid] * (ex[fid] * uxx + ey[fid] * uyy + ez[fid] * uzz);
  }*/
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
