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
#define FLUIDE_ -1  // TODO: move this to a more appropriate place (from LBMDEM3D code)

namespace hippoLBM {
namespace bcs {

/**
 * @brief Rho (density/pressure) boundary condition functor, specialized per boundary plane.
 * Undefined for any (Q, Traversal) pair without a matching specialization below.
 */
template <int Q, Traversal T>
struct Rho {};

/** @brief rho boundary condition at x=lx (Q=19). */
template <>
struct Rho<19, Traversal::Plan_yz_l> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double rho,
                                                const double ux, const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
      const double u =
          -1. + (f(idx, 3) + f(idx, 4) + f(idx, 5) + f(idx, 6) + f(idx, 15) + f(idx, 17) + f(idx, 18) + f(idx, 16) +
                 f(idx, 0) + 2. * (f(idx, 1) + f(idx, 7) + f(idx, 9) + f(idx, 11) + f(idx, 13))) /
                    rho;
      const double nyx = (1. / 2.) * (f(idx, 3) + f(idx, 15) + f(idx, 17) - (f(idx, 4) + f(idx, 18) + f(idx, 16))) -
                         (1. / 3.) * rho * uy;
      const double nzx = (1. / 2.) * (f(idx, 5) + f(idx, 18) + f(idx, 15) - (f(idx, 6) + f(idx, 17) + f(idx, 16))) -
                         (1. / 3.) * rho * uz;
      f(idx, 2) = f(idx, 1) - (1. / 3.) * rho * u;
      f(idx, 10) = f(idx, 9) + (1. / 6.) * rho * (-u + uy) - nyx;
      f(idx, 8) = f(idx, 7) + (1. / 6.) * rho * (-u - uy) + nyx;
      f(idx, 12) = f(idx, 11) + (1. / 6.) * rho * (-u - uz) + nzx;
      f(idx, 14) = f(idx, 13) + (1. / 6.) * rho * (-u + uz) - nzx;
    }
  }
};

/** @brief rho boundary condition at x=0 (Q=19). */
template <>
struct Rho<19, Traversal::Plan_yz_0> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double rho,
                                                const double ux, const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
      const double u =
          1. - (f(idx, 3) + f(idx, 4) + f(idx, 5) + f(idx, 6) + f(idx, 15) + f(idx, 17) + f(idx, 18) + f(idx, 16) +
                f(idx, 0) + 2. * (f(idx, 2) + f(idx, 10) + f(idx, 8) + f(idx, 14) + f(idx, 12))) /
                   rho;
      const double nyx = (1. / 2.) * (f(idx, 3) + f(idx, 15) + f(idx, 17) - (f(idx, 4) + f(idx, 18) + f(idx, 16))) -
                         (1. / 3.) * rho * uy;
      const double nzx = (1. / 2.) * (f(idx, 5) + f(idx, 18) + f(idx, 15) - (f(idx, 6) + f(idx, 17) + f(idx, 16))) -
                         (1. / 3.) * rho * uz;
      f(idx, 1) = f(idx, 2) + (1. / 3.) * rho * u;
      f(idx, 9) = f(idx, 10) + (1. / 6.) * rho * (u - uy) + nyx;
      f(idx, 7) = f(idx, 8) + (1. / 6.) * rho * (u + uy) - nyx;
      f(idx, 11) = f(idx, 12) + (1. / 6.) * rho * (u + uz) - nzx;
      f(idx, 13) = f(idx, 14) + (1. / 6.) * rho * (u - uz) + nzx;
    }
  }
};

/** @brief rho boundary condition at y=ly (Q=19). */
template <>
struct Rho<19, Traversal::Plan_xz_l> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double rho,
                                                const double ux, const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
      const double u =
          -1. + (f(idx, 1) + f(idx, 2) + f(idx, 5) + f(idx, 6) + f(idx, 11) + f(idx, 13) + f(idx, 14) + f(idx, 12) +
                 f(idx, 0) + 2. * (f(idx, 3) + f(idx, 7) + f(idx, 10) + f(idx, 15) + f(idx, 17))) /
                    rho;
      const double nxy = (1. / 2.) * (f(idx, 1) + f(idx, 11) + f(idx, 13) - (f(idx, 2) + f(idx, 14) + f(idx, 12))) -
                         (1. / 3.) * rho * ux;
      const double nzy = (1. / 2.) * (f(idx, 5) + f(idx, 11) + f(idx, 14) - (f(idx, 6) + f(idx, 13) + f(idx, 12))) -
                         (1. / 3.) * rho * uz;
      f(idx, 4) = f(idx, 3) - (1. / 3.) * rho * u;
      f(idx, 8) = f(idx, 7) + (1. / 6.) * rho * (-u - ux) + nxy;
      f(idx, 9) = f(idx, 10) + (1. / 6.) * rho * (-u + ux) - nxy;
      f(idx, 16) = f(idx, 15) + (1. / 6.) * rho * (-u - uz) + nzy;
      f(idx, 18) = f(idx, 17) + (1. / 6.) * rho * (-u + uz) - nzy;
    }
  }
};

/** @brief rho boundary condition at y=0 (Q=19). */
template <>
struct Rho<19, Traversal::Plan_xz_0> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double rho,
                                                const double ux, const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
      const double u =
          1. - (f(idx, 1) + f(idx, 2) + f(idx, 5) + f(idx, 6) + f(idx, 11) + f(idx, 13) + f(idx, 14) + f(idx, 12) +
                f(idx, 0) + 2. * (f(idx, 4) + f(idx, 9) + f(idx, 8) + f(idx, 18) + f(idx, 16))) /
                   rho;
      const double nxy = (1. / 2.) * (f(idx, 1) + f(idx, 11) + f(idx, 13) - (f(idx, 2) + f(idx, 14) + f(idx, 12))) -
                         (1. / 3.) * rho * ux;
      const double nzy = (1. / 2.) * (f(idx, 5) + f(idx, 11) + f(idx, 14) - (f(idx, 6) + f(idx, 13) + f(idx, 12))) -
                         (1. / 3.) * rho * uz;
      f(idx, 3) = f(idx, 4) + (1. / 3.) * rho * u;
      f(idx, 7) = f(idx, 8) + (1. / 6.) * rho * (u + ux) - nxy;
      f(idx, 10) = f(idx, 9) + (1. / 6.) * rho * (u - ux) + nxy;
      f(idx, 15) = f(idx, 16) + (1. / 6.) * rho * (u + uz) - nzy;
      f(idx, 17) = f(idx, 18) + (1. / 6.) * rho * (u - uz) + nzy;
    }
  }
};

/** @brief rho boundary condition at z=lz (Q=19). */
template <>
struct Rho<19, Traversal::Plan_xy_l> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double rho,
                                                const double ux, const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
      const double u =
          -1. + (f(idx, 1) + f(idx, 2) + f(idx, 3) + f(idx, 4) + f(idx, 7) + f(idx, 10) + f(idx, 8) + f(idx, 9) +
                 f(idx, 0) + 2. * (f(idx, 5) + f(idx, 11) + f(idx, 14) + f(idx, 15) + f(idx, 18))) /
                    rho;
      const double nxz =
          (1. / 2.) * (f(idx, 1) + f(idx, 7) + f(idx, 9) - (f(idx, 2) + f(idx, 10) + f(idx, 8))) - (1. / 3.) * rho * ux;
      const double nyz =
          (1. / 2.) * (f(idx, 3) + f(idx, 7) + f(idx, 10) - (f(idx, 4) + f(idx, 9) + f(idx, 8))) - (1. / 3.) * rho * uy;
      f(idx, 6) = f(idx, 5) - (1. / 3.) * rho * u;
      f(idx, 13) = f(idx, 14) + (1. / 6.) * rho * (-u + ux) - nxz;
      f(idx, 12) = f(idx, 11) + (1. / 6.) * rho * (-u - ux) + nxz;
      f(idx, 17) = f(idx, 18) + (1. / 6.) * rho * (-u + uy) - nyz;
      f(idx, 16) = f(idx, 15) + (1. / 6.) * rho * (-u - uy) + nyz;
    }
  }
};

/** @brief rho boundary condition at z=0 (Q=19). */
template <>
struct Rho<19, Traversal::Plan_xy_0> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double rho,
                                                const double ux, const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
      const double u =
          1. - (f(idx, 1) + f(idx, 2) + f(idx, 3) + f(idx, 4) + f(idx, 7) + f(idx, 9) + f(idx, 10) + f(idx, 8) +
                f(idx, 0) + 2. * (f(idx, 6) + f(idx, 13) + f(idx, 12) + f(idx, 17) + f(idx, 16))) /
                   rho;
      const double nxz =
          (1. / 2.) * (f(idx, 1) + f(idx, 7) + f(idx, 9) - (f(idx, 2) + f(idx, 10) + f(idx, 8))) - (1. / 3.) * rho * ux;
      const double nyz =
          (1. / 2.) * (f(idx, 3) + f(idx, 7) + f(idx, 10) - (f(idx, 4) + f(idx, 9) + f(idx, 8))) - (1. / 3.) * rho * uy;
      f(idx, 5) = f(idx, 6) + (1. / 3.) * rho * u;
      f(idx, 11) = f(idx, 12) + (1. / 6.) * rho * (u + ux) - nxz;
      f(idx, 14) = f(idx, 13) + (1. / 6.) * rho * (u - ux) + nxz;
      f(idx, 15) = f(idx, 16) + (1. / 6.) * rho * (u + uy) - nyz;
      f(idx, 18) = f(idx, 17) + (1. / 6.) * rho * (u - uy) + nyz;
    }
  }
};
}  // namespace bcs
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal T>
struct ParallelForFunctorTraits<hippoLBM::bcs::Rho<Q, T>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
