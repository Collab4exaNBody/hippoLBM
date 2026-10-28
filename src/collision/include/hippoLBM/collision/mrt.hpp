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
struct mrt {};

/**
 * @brief Relaxation rates of the non-hydrodynamic moments (D'Humieres et al. 2002).
 * The rates of the stress moments (3pxx, pww, pxy, pyz, pxz) are not listed here: they are set to 1/tau, which fixes
 * the viscosity nu = (tau - 1/2) / 3.
 */
struct MRTRates {
  double s_e = 1.19;    // energy
  double s_eps = 1.4;   // energy square
  double s_q = 1.2;     // energy flux
  double s_pi = 1.4;    // stress square (3pixx, piww)
  double s_m = 1.98;    // third order (mx, my, mz)
};
/**
 * @brief A functor for collision operations in the lattice Boltzmann method.
 */
template <>
struct mrt<19> {
  const onika::math::Vec3d Fext_;  // External force term, used in the computation of macroscopic variables.
  const MRTRates rates_;           // Relaxation rates of the non-hydrodynamic moments.

  /** @brief Computes the MRT collision operator for a given lattice node.
   * @param f The field view for the distribution functions.
   * @param idx The index of the lattice node.
   * @param tau The relaxation time.
   */
  ONIKA_HOST_DEVICE_FUNC void mrt_core(const FieldView<19>& f, const size_t idx, double tau) const {
    // Stress moments relax with 1/tau (viscosity), the others with the user-defined rates.
    const double s9 = 1 / tau, s13 = s9;
    const double s1 = rates_.s_e, s2 = rates_.s_eps, s4 = rates_.s_q, s10 = rates_.s_pi, s16 = rates_.s_m;
    // The equilibrium moments are the moments of the BGK equilibrium (rho-dependent j.j / rho term). Hence the rest
    // state and the BGK equilibrium are fixed points of the collision, and MRT with all rates = 1/tau is exactly BGK.
    const double weps = 3, wepsj = -11. / 2, wxx = -1. / 2;
    double jj, jx2, jy2, jz2;
    double rho, e, eps, jx, qx, jy, qy, jz, qz, pxx3, pixx3, pww, piww, pxy, pyz, pxz, mx, my, mz;
    double eO, epsO, qxO, qyO, qzO, pxx3O, pixx3O, pwwO, piwwO, pxyO, pyzO, pxzO, mxO, myO, mzO;
    double e_eq, eps_eq, pxx3_eq, pww_eq, pxy_eq, pyz_eq, pxz_eq;

    // Compute macroscopic quantities
    rho = +f(idx, 0) + f(idx, 1) + f(idx, 2) + f(idx, 3) + f(idx, 4) + f(idx, 5) + f(idx, 6) + f(idx, 7) + f(idx, 8) +
          f(idx, 9) + f(idx, 10) + f(idx, 11) + f(idx, 12) + f(idx, 13) + f(idx, 14) + f(idx, 15) + f(idx, 16) +
          f(idx, 17) + f(idx, 18);
    e = -30 * f(idx, 0) - 11 * f(idx, 1) - 11 * f(idx, 2) - 11 * f(idx, 3) - 11 * f(idx, 4) - 11 * f(idx, 5) -
        11 * f(idx, 6) + 8 * f(idx, 7) + 8 * f(idx, 8) + 8 * f(idx, 9) + 8 * f(idx, 10) + 8 * f(idx, 11) +
        8 * f(idx, 12) + 8 * f(idx, 13) + 8 * f(idx, 14) + 8 * f(idx, 15) + 8 * f(idx, 16) + 8 * f(idx, 17) +
        8 * f(idx, 18);
    eps = +12 * f(idx, 0) - 4 * f(idx, 1) - 4 * f(idx, 2) - 4 * f(idx, 3) - 4 * f(idx, 4) - 4 * f(idx, 5) -
          4 * f(idx, 6) + f(idx, 7) + f(idx, 8) + f(idx, 9) + f(idx, 10) + f(idx, 11) + f(idx, 12) + f(idx, 13) +
          f(idx, 14) + f(idx, 15) + f(idx, 16) + f(idx, 17) + f(idx, 18);
    jx = +f(idx, 1) - f(idx, 2) + f(idx, 7) - f(idx, 8) + f(idx, 9) - f(idx, 10) + f(idx, 11) - f(idx, 12) +
         f(idx, 13) - f(idx, 14);
    qx = -4 * f(idx, 1) + 4 * f(idx, 2) + f(idx, 7) - f(idx, 8) + f(idx, 9) - f(idx, 10) + f(idx, 11) - f(idx, 12) +
         f(idx, 13) - f(idx, 14);
    jy = +f(idx, 3) - f(idx, 4) + f(idx, 7) - f(idx, 8) - f(idx, 9) + f(idx, 10) + f(idx, 15) - f(idx, 16) +
         f(idx, 17) - f(idx, 18);
    qy = -4 * f(idx, 3) + 4 * f(idx, 4) + f(idx, 7) - f(idx, 8) - f(idx, 9) + f(idx, 10) + f(idx, 15) - f(idx, 16) +
         f(idx, 17) - f(idx, 18);
    jz = +f(idx, 5) - f(idx, 6) + f(idx, 11) - f(idx, 12) - f(idx, 13) + f(idx, 14) + f(idx, 15) - f(idx, 16) -
         f(idx, 17) + f(idx, 18);
    qz = -4 * f(idx, 5) + 4 * f(idx, 6) + f(idx, 11) - f(idx, 12) - f(idx, 13) + f(idx, 14) + f(idx, 15) - f(idx, 16) -
         f(idx, 17) + f(idx, 18);
    pxx3 = +2 * f(idx, 1) + 2 * f(idx, 2) - f(idx, 3) - f(idx, 4) - f(idx, 5) - f(idx, 6) + f(idx, 7) + f(idx, 8) +
           f(idx, 9) + f(idx, 10) + f(idx, 11) + f(idx, 12) + f(idx, 13) + f(idx, 14) - 2 * f(idx, 15) -
           2 * f(idx, 16) - 2 * f(idx, 17) - 2 * f(idx, 18);
    pixx3 = -4 * f(idx, 1) - 4 * f(idx, 2) + 2 * f(idx, 3) + 2 * f(idx, 4) + 2 * f(idx, 5) + 2 * f(idx, 6) + f(idx, 7) +
            f(idx, 8) + f(idx, 9) + f(idx, 10) + f(idx, 11) + f(idx, 12) + f(idx, 13) + f(idx, 14) - 2 * f(idx, 15) -
            2 * f(idx, 16) - 2 * f(idx, 17) - 2 * f(idx, 18);
    pww = +f(idx, 3) + f(idx, 4) - f(idx, 5) - f(idx, 6) + f(idx, 7) + f(idx, 8) + f(idx, 9) + f(idx, 10) - f(idx, 11) -
          f(idx, 12) - f(idx, 13) - f(idx, 14);
    piww = -2 * f(idx, 3) - 2 * f(idx, 4) + 2 * f(idx, 5) + 2 * f(idx, 6) + f(idx, 7) + f(idx, 8) + f(idx, 9) +
           f(idx, 10) - f(idx, 11) - f(idx, 12) - f(idx, 13) - f(idx, 14);
    pxy = +f(idx, 7) + f(idx, 8) - f(idx, 9) - f(idx, 10);
    pyz = +f(idx, 15) + f(idx, 16) - f(idx, 17) - f(idx, 18);
    pxz = +f(idx, 11) + f(idx, 12) - f(idx, 13) - f(idx, 14);
    mx = +f(idx, 7) - f(idx, 8) + f(idx, 9) - f(idx, 10) - f(idx, 11) + f(idx, 12) - f(idx, 13) + f(idx, 14);
    my = -f(idx, 7) + f(idx, 8) + f(idx, 9) - f(idx, 10) + f(idx, 15) - f(idx, 16) + f(idx, 17) - f(idx, 18);
    mz = +f(idx, 11) - f(idx, 12) - f(idx, 13) + f(idx, 14) - f(idx, 15) + f(idx, 16) + f(idx, 17) - f(idx, 18);

    jx2 = jx * jx;
    jy2 = jy * jy;
    jz2 = jz * jz;
    jj = (jx2 + jy2 + jz2);

    // Calculate MRT collision parameters
    e_eq = -11 * rho + 19 * jj / rho;
    eO = e - s1 * (e - e_eq);
    eps_eq = weps * rho + wepsj * jj / rho;
    epsO = eps - s2 * (eps - eps_eq);

    qxO = qx - s4 * (qx + 2. / 3. * jx);
    qyO = qy - s4 * (qy + 2. / 3. * jy);
    qzO = qz - s4 * (qz + 2. / 3. * jz);
    pxx3_eq = 1 / rho * (2 * jx2 - (jy2 + jz2));
    pxx3O = pxx3 - s9 * (pxx3 - pxx3_eq);
    pixx3O = pixx3 - s10 * (pixx3 - wxx * pxx3_eq);

    pww_eq = 1 / rho * (jy2 - jz2);
    pwwO = pww - s9 * (pww - pww_eq);
    piwwO = piww - s10 * (piww - wxx * pww_eq);
    pxy_eq = 1 / rho * jx * jy;
    pxyO = pxy - s13 * (pxy - pxy_eq);
    pyz_eq = 1 / rho * jy * jz;
    pyzO = pyz - s13 * (pyz - pyz_eq);
    pxz_eq = 1 / rho * jx * jz;
    pxzO = pxz - s13 * (pxz - pxz_eq);
    mxO = mx - s16 * mx;
    myO = my - s16 * my;
    mzO = mz - s16 * mz;

    // Update distribution functions after collision
    f(idx, 0) = +1. / 19 * rho - 5. / 399 * eO + 1. / 21 * epsO;
    f(idx, 1) = +1. / 19 * rho - 11. / 2394 * eO - 1. / 63 * epsO + 1. / 10 * jx - 1. / 10 * qxO + 1. / 18 * pxx3O -
                1. / 18 * pixx3O;
    f(idx, 2) = +1. / 19 * rho - 11. / 2394 * eO - 1. / 63 * epsO - 1. / 10 * jx + 1. / 10 * qxO + 1. / 18 * pxx3O -
                1. / 18 * pixx3O;
    f(idx, 3) = +1. / 19 * rho - 11. / 2394 * eO - 1. / 63 * epsO + 1. / 10 * jy - 1. / 10 * qyO - 1. / 36 * pxx3O +
                1. / 36 * pixx3O + 1. / 12 * pwwO - 1. / 12 * piwwO;
    f(idx, 4) = +1. / 19 * rho - 11. / 2394 * eO - 1. / 63 * epsO - 1. / 10 * jy + 1. / 10 * qyO - 1. / 36 * pxx3O +
                1. / 36 * pixx3O + 1. / 12 * pwwO - 1. / 12 * piwwO;
    f(idx, 5) = +1. / 19 * rho - 11. / 2394 * eO - 1. / 63 * epsO + 1. / 10 * jz - 1. / 10 * qzO - 1. / 36 * pxx3O +
                1. / 36 * pixx3O - 1. / 12 * pwwO + 1. / 12 * piwwO;
    f(idx, 6) = +1. / 19 * rho - 11. / 2394 * eO - 1. / 63 * epsO - 1. / 10 * jz + 1. / 10 * qzO - 1. / 36 * pxx3O +
                1. / 36 * pixx3O - 1. / 12 * pwwO + 1. / 12 * piwwO;
    f(idx, 7) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO + 1. / 10 * jx + 1. / 40 * qxO + 1. / 10 * jy +
                1. / 40 * qyO + 1. / 36 * pxx3O + 1. / 72 * pixx3O + 1. / 12 * pwwO + 1. / 24 * piwwO + 1. / 4 * pxyO +
                1. / 8 * mxO - 1. / 8 * myO;
    f(idx, 8) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO - 1. / 10 * jx - 1. / 40 * qxO - 1. / 10 * jy -
                1. / 40 * qyO + 1. / 36 * pxx3O + 1. / 72 * pixx3O + 1. / 12 * pwwO + 1. / 24 * piwwO + 1. / 4 * pxyO -
                1. / 8 * mxO + 1. / 8 * myO;
    f(idx, 9) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO + 1. / 10 * jx + 1. / 40 * qxO - 1. / 10 * jy -
                1. / 40 * qyO + 1. / 36 * pxx3O + 1. / 72 * pixx3O + 1. / 12 * pwwO + 1. / 24 * piwwO - 1. / 4 * pxyO +
                1. / 8 * mxO + 1. / 8 * myO;
    f(idx, 10) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO - 1. / 10 * jx - 1. / 40 * qxO + 1. / 10 * jy +
                 1. / 40 * qyO + 1. / 36 * pxx3O + 1. / 72 * pixx3O + 1. / 12 * pwwO + 1. / 24 * piwwO - 1. / 4 * pxyO -
                 1. / 8 * mxO - 1. / 8 * myO;
    f(idx, 11) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO + 1. / 10 * jx + 1. / 40 * qxO + 1. / 10 * jz +
                 1. / 40 * qzO + 1. / 36 * pxx3O + 1. / 72 * pixx3O - 1. / 12 * pwwO - 1. / 24 * piwwO + 1. / 4 * pxzO -
                 1. / 8 * mxO + 1. / 8 * mzO;
    f(idx, 12) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO - 1. / 10 * jx - 1. / 40 * qxO - 1. / 10 * jz -
                 1. / 40 * qzO + 1. / 36 * pxx3O + 1. / 72 * pixx3O - 1. / 12 * pwwO - 1. / 24 * piwwO + 1. / 4 * pxzO +
                 1. / 8 * mxO - 1. / 8 * mzO;
    f(idx, 13) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO + 1. / 10 * jx + 1. / 40 * qxO - 1. / 10 * jz -
                 1. / 40 * qzO + 1. / 36 * pxx3O + 1. / 72 * pixx3O - 1. / 12 * pwwO - 1. / 24 * piwwO - 1. / 4 * pxzO -
                 1. / 8 * mxO - 1. / 8 * mzO;
    f(idx, 14) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO - 1. / 10 * jx - 1. / 40 * qxO + 1. / 10 * jz +
                 1. / 40 * qzO + 1. / 36 * pxx3O + 1. / 72 * pixx3O - 1. / 12 * pwwO - 1. / 24 * piwwO - 1. / 4 * pxzO +
                 1. / 8 * mxO + 1. / 8 * mzO;
    f(idx, 15) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO + 1. / 10 * jy + 1. / 40 * qyO + 1. / 10 * jz +
                 1. / 40 * qzO - 1. / 18 * pxx3O - 1. / 36 * pixx3O + 1. / 4 * pyzO + 1. / 8 * myO - 1. / 8 * mzO;
    f(idx, 16) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO - 1. / 10 * jy - 1. / 40 * qyO - 1. / 10 * jz -
                 1. / 40 * qzO - 1. / 18 * pxx3O - 1. / 36 * pixx3O + 1. / 4 * pyzO - 1. / 8 * myO + 1. / 8 * mzO;
    f(idx, 17) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO + 1. / 10 * jy + 1. / 40 * qyO - 1. / 10 * jz -
                 1. / 40 * qzO - 1. / 18 * pxx3O - 1. / 36 * pixx3O - 1. / 4 * pyzO + 1. / 8 * myO + 1. / 8 * mzO;
    f(idx, 18) = +1. / 19 * rho + 4. / 1197 * eO + 1. / 252 * epsO - 1. / 10 * jy - 1. / 40 * qyO + 1. / 10 * jz +
                 1. / 40 * qzO - 1. / 18 * pxx3O - 1. / 36 * pixx3O - 1. / 4 * pyzO - 1. / 8 * myO - 1. / 8 * mzO;
  }

  /**
   * @brief Operator for performing collision operations at a given index.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const __restrict__ obst, const FieldView<19>& f,
                                                double* const __restrict__ m0, const double tau) const {
    if (obst[idx] == FLUID_CELL) {
      const double rho = m0[idx];

      // step 1, fill f[iLB]
      mrt_core(f, idx, tau);
      // step 2, adjust with Fext
      stencil::for_each<typename LBMScheme<19>::Coefficients>([&]<typename coeff>(int iLB) {
        const double ef = coeff::ex * Fext_.x + coeff::ey * Fext_.y + coeff::ez * Fext_.z;
        double& fiLB = f(idx, iLB);
        fiLB += 3. * rho * coeff::w * ef;
      });
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::mrt<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
