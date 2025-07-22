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

#include <grid/field_view.hpp>
#define FLUIDE_ -1

namespace hippoLBM
{
  using namespace onika::math;
  template<int Q> struct mrt {};
  /**
   * @brief A functor for collision operations in the lattice Boltzmann method.
   */
  template<>
    struct mrt<19>
    {
      const Vec3d m_Fext;

      ONIKA_HOST_DEVICE_FUNC void mrt_core(const FieldView<19>& f, const size_t idx, double tau) const 
      {
        const double s9 = 1/tau, s13 = s9;
        // D'humiéres et al parametrization
        const double s1 = 1.19, s2 = 1.4, s4 = 1.2, s10 = s2, s16 = 1.98;
        const double weps = 0, wepsj = -475./63., wxx = 0;
        // LBGK parametrization
        // const double s1 = s9, s2 = s9, s4 = s9, s10 = s2, s16 = s9;
        // const double weps = 3., wepsj = -11./2, wxx = -1./2;
        double rho0, jj, jx2, jy2, jz2;
        double rho, e, eps, jx, qx, jy, qy, jz, qz, pxx3, pixx3, pww, piww,
               pxy, pyz, pxz, mx, my, mz;
        double eO, epsO, qxO, qyO, qzO, pxx3O, pixx3O, pwwO, piwwO,
               pxyO, pyzO, pxzO, mxO, myO, mzO;
        double e_eq, eps_eq, pxx3_eq, pww_eq, pxy_eq, pyz_eq, pxz_eq;

        // Compute macroscopic quantities
        rho =  + f(idx,0) + f(idx,1) + f(idx,2) + f(idx,3) + f(idx,4) + f(idx,5) + f(idx,6) + f(idx,7) + f(idx,8) + f(idx,9) + f(idx,10) + f(idx,11) + f(idx,12) + f(idx,13) + f(idx,14) + f(idx,15) + f(idx,16) + f(idx,17) + f(idx,18);
        e =  - 30 * f(idx,0) - 11 * f(idx,1) - 11 * f(idx,2) - 11 * f(idx,3) - 11 * f(idx,4) - 11 * f(idx,5) - 11 * f(idx,6) + 8 * f(idx,7) + 8 * f(idx,8) + 8 * f(idx,9) + 8 * f(idx,10) + 8 * f(idx,11) + 8 * f(idx,12) + 8 * f(idx,13) + 8 * f(idx,14) + 8 * f(idx,15) + 8 * f(idx,16) + 8 * f(idx,17) + 8 * f(idx,18);
        eps =  + 12 * f(idx,0) - 4 * f(idx,1) - 4 * f(idx,2) - 4 * f(idx,3) - 4 * f(idx,4) - 4 * f(idx,5) - 4 * f(idx,6) + f(idx,7) + f(idx,8) + f(idx,9) + f(idx,10) + f(idx,11) + f(idx,12) + f(idx,13) + f(idx,14) + f(idx,15) + f(idx,16) + f(idx,17) + f(idx,18);
        jx =  + f(idx,1) - f(idx,2) + f(idx,7) - f(idx,8) + f(idx,9) - f(idx,10) + f(idx,11) - f(idx,12) + f(idx,13) - f(idx,14);
        qx =  - 4 * f(idx,1) + 4 * f(idx,2) + f(idx,7) - f(idx,8) + f(idx,9) - f(idx,10) + f(idx,11) - f(idx,12) + f(idx,13) - f(idx,14);
        jy =  + f(idx,3) - f(idx,4) + f(idx,7) - f(idx,8) - f(idx,9) + f(idx,10) + f(idx,15) - f(idx,16) + f(idx,17) - f(idx,18);
        qy =  - 4 * f(idx,3) + 4 * f(idx,4) + f(idx,7) - f(idx,8) - f(idx,9) + f(idx,10) + f(idx,15) - f(idx,16) + f(idx,17) - f(idx,18);
        jz =  + f(idx,5) - f(idx,6) + f(idx,11) - f(idx,12) - f(idx,13) + f(idx,14) + f(idx,15) - f(idx,16) - f(idx,17) + f(idx,18);
        qz =  - 4 * f(idx,5) + 4 * f(idx,6) + f(idx,11) - f(idx,12) - f(idx,13) + f(idx,14) + f(idx,15) - f(idx,16) - f(idx,17) + f(idx,18);
        pxx3 =  + 2 * f(idx,1) + 2 * f(idx,2) - f(idx,3) - f(idx,4) - f(idx,5) - f(idx,6) + f(idx,7) + f(idx,8) + f(idx,9) + f(idx,10) + f(idx,11) + f(idx,12) + f(idx,13) + f(idx,14) - 2 * f(idx,15) - 2 * f(idx,16) - 2 * f(idx,17) - 2 * f(idx,18);
        pixx3 =  - 4 * f(idx,1) - 4 * f(idx,2) + 2 * f(idx,3) + 2 * f(idx,4) + 2 * f(idx,5) + 2 * f(idx,6) + f(idx,7) + f(idx,8) + f(idx,9) + f(idx,10) + f(idx,11) + f(idx,12) + f(idx,13) + f(idx,14) - 2 * f(idx,15) - 2 * f(idx,16) - 2 * f(idx,17) - 2 * f(idx,18);
        pww =  + f(idx,3) + f(idx,4) - f(idx,5) - f(idx,6) + f(idx,7) + f(idx,8) + f(idx,9) + f(idx,10) - f(idx,11) - f(idx,12) - f(idx,13) - f(idx,14);
        piww =  - 2 * f(idx,3) - 2 * f(idx,4) + 2 * f(idx,5) + 2 * f(idx,6) + f(idx,7) + f(idx,8) + f(idx,9) + f(idx,10) - f(idx,11) - f(idx,12) - f(idx,13) - f(idx,14);
        pxy =  + f(idx,7) + f(idx,8) - f(idx,9) - f(idx,10);
        pyz =  + f(idx,15) + f(idx,16) - f(idx,17) - f(idx,18);
        pxz =  + f(idx,11) + f(idx,12) - f(idx,13) - f(idx,14);
        mx =  + f(idx,7) - f(idx,8) + f(idx,9) - f(idx,10) - f(idx,11) + f(idx,12) - f(idx,13) + f(idx,14);
        my =  - f(idx,7) + f(idx,8) + f(idx,9) - f(idx,10) + f(idx,15) - f(idx,16) + f(idx,17) - f(idx,18);
        mz =  + f(idx,11) - f(idx,12) - f(idx,13) + f(idx,14) - f(idx,15) + f(idx,16) + f(idx,17) - f(idx,18);

        rho0 = 1.;
        jx2 = jx*jx;
        jy2 = jy*jy;
        jz2 = jz*jz;
        jj = (jx2 + jy2 + jz2);

        // Calculate MRT collision parameters
        e_eq = -11 * rho + 19/rho0 * jj;
        eO = e - s1 * (e - e_eq);
        eps_eq = weps * rho + wepsj/rho0 * jj;
        epsO = eps - s2 * (eps - eps_eq);

        qxO = qx - s4 * (qx + 2./3. * jx);
        qyO = qy - s4 * (qy + 2./3. * jy);
        qzO = qz - s4 * (qz + 2./3. * jz);
        pxx3_eq = 1/(rho0) * (2*jx2 - (jy2 + jz2));
        pxx3O = pxx3 - s9 * (pxx3 - pxx3_eq);
        pixx3O = pixx3 - s10 * (pixx3 - wxx * pxx3_eq);

        pww_eq = 1/rho0 * (jy2 - jz2);
        pwwO = pww - s9 * (pww - pww_eq);
        piwwO = piww - s10 * (piww - wxx * pww_eq);
        pxy_eq = 1/rho0 * jx * jy;
        pxyO = pxy - s13 * (pxy - pxy_eq);
        pyz_eq = 1/rho0 * jy * jz;
        pyzO = pyz - s13 * (pyz - pyz_eq);
        pxz_eq = 1/rho0 * jx * jz;
        pxzO = pxz - s13 * (pxz - pxz_eq);
        mxO = mx - s16 * mx;
        myO = my - s16 * my;
        mzO = mz - s16 * mz;

        // Update distribution functions after collision
        f(idx,0) =  + 1./19 * rho - 5./399 * eO + 1./21 * epsO;
        f(idx,1) =  + 1./19 * rho - 11./2394 * eO - 1./63 * epsO + 1./10 * jx - 1./10 * qxO + 1./18 * pxx3O - 1./18 * pixx3O;
        f(idx,2) =  + 1./19 * rho - 11./2394 * eO - 1./63 * epsO - 1./10 * jx + 1./10 * qxO + 1./18 * pxx3O - 1./18 * pixx3O;
        f(idx,3) =  + 1./19 * rho - 11./2394 * eO - 1./63 * epsO + 1./10 * jy - 1./10 * qyO - 1./36 * pxx3O + 1./36 * pixx3O + 1./12 * pwwO - 1./12 * piwwO;
        f(idx,4) =  + 1./19 * rho - 11./2394 * eO - 1./63 * epsO - 1./10 * jy + 1./10 * qyO - 1./36 * pxx3O + 1./36 * pixx3O + 1./12 * pwwO - 1./12 * piwwO;
        f(idx,5) =  + 1./19 * rho - 11./2394 * eO - 1./63 * epsO + 1./10 * jz - 1./10 * qzO - 1./36 * pxx3O + 1./36 * pixx3O - 1./12 * pwwO + 1./12 * piwwO;
        f(idx,6) =  + 1./19 * rho - 11./2394 * eO - 1./63 * epsO - 1./10 * jz + 1./10 * qzO - 1./36 * pxx3O + 1./36 * pixx3O - 1./12 * pwwO + 1./12 * piwwO;
        f(idx,7) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO + 1./10 * jx + 1./40 * qxO + 1./10 * jy + 1./40 * qyO + 1./36 * pxx3O + 1./72 * pixx3O + 1./12 * pwwO + 1./24 * piwwO + 1./4 * pxyO + 1./8 * mxO - 1./8 * myO;
        f(idx,8) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO - 1./10 * jx - 1./40 * qxO - 1./10 * jy - 1./40 * qyO + 1./36 * pxx3O + 1./72 * pixx3O + 1./12 * pwwO + 1./24 * piwwO + 1./4 * pxyO - 1./8 * mxO + 1./8 * myO;
        f(idx,9) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO + 1./10 * jx + 1./40 * qxO - 1./10 * jy - 1./40 * qyO + 1./36 * pxx3O + 1./72 * pixx3O + 1./12 * pwwO + 1./24 * piwwO - 1./4 * pxyO + 1./8 * mxO + 1./8 * myO;
        f(idx,10) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO - 1./10 * jx - 1./40 * qxO + 1./10 * jy + 1./40 * qyO + 1./36 * pxx3O + 1./72 * pixx3O + 1./12 * pwwO + 1./24 * piwwO - 1./4 * pxyO - 1./8 * mxO - 1./8 * myO;
        f(idx,11) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO + 1./10 * jx + 1./40 * qxO + 1./10 * jz + 1./40 * qzO + 1./36 * pxx3O + 1./72 * pixx3O - 1./12 * pwwO - 1./24 * piwwO + 1./4 * pxzO - 1./8 * mxO + 1./8 * mzO;
        f(idx,12) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO - 1./10 * jx - 1./40 * qxO - 1./10 * jz - 1./40 * qzO + 1./36 * pxx3O + 1./72 * pixx3O - 1./12 * pwwO - 1./24 * piwwO + 1./4 * pxzO + 1./8 * mxO - 1./8 * mzO;
        f(idx,13) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO + 1./10 * jx + 1./40 * qxO - 1./10 * jz - 1./40 * qzO + 1./36 * pxx3O + 1./72 * pixx3O - 1./12 * pwwO - 1./24 * piwwO - 1./4 * pxzO - 1./8 * mxO - 1./8 * mzO;
        f(idx,14) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO - 1./10 * jx - 1./40 * qxO + 1./10 * jz + 1./40 * qzO + 1./36 * pxx3O + 1./72 * pixx3O - 1./12 * pwwO - 1./24 * piwwO - 1./4 * pxzO + 1./8 * mxO + 1./8 * mzO;
        f(idx,15) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO + 1./10 * jy + 1./40 * qyO + 1./10 * jz + 1./40 * qzO - 1./18 * pxx3O - 1./36 * pixx3O + 1./4 * pyzO + 1./8 * myO - 1./8 * mzO;
        f(idx,16) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO - 1./10 * jy - 1./40 * qyO - 1./10 * jz - 1./40 * qzO - 1./18 * pxx3O - 1./36 * pixx3O + 1./4 * pyzO - 1./8 * myO + 1./8 * mzO;
        f(idx,17) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO + 1./10 * jy + 1./40 * qyO - 1./10 * jz - 1./40 * qzO - 1./18 * pxx3O - 1./36 * pixx3O - 1./4 * pyzO + 1./8 * myO + 1./8 * mzO;
        f(idx,18) =  + 1./19 * rho + 4./1197 * eO + 1./252 * epsO - 1./10 * jy - 1./40 * qyO + 1./10 * jz + 1./40 * qzO - 1./18 * pxx3O - 1./36 * pixx3O - 1./4 * pyzO - 1./8 * myO - 1./8 * mzO;
      }


      /**
       * @brief Operator for performing collision operations at a given index.
       */
      ONIKA_HOST_DEVICE_FUNC inline void operator()(
          int idx, 
          int * const __restrict__ obst, 
          const FieldView<19>& f,
          double * const __restrict__ m0, 
          const int* __restrict__ ex, 
          const int* __restrict__ ey,
          const int* __restrict__ ez, 
          const double* const __restrict__ w, 
          const double tau) const
      {
        if (obst[idx] == FLUIDE_) 
        {
          const double rho = m0[idx];

          // step 1, fill f[iLB]
          mrt_core(f, idx, tau);
          // step 2, adjust with Fext
          for (int iLB = 0; iLB < 19; iLB++) 
          {
            const double ef = ex[iLB] * m_Fext.x + ey[iLB] * m_Fext.y + ez[iLB] * m_Fext.z;
            double& fiLB = f(idx,iLB);
            fiLB += 3. * rho * w[iLB] * ef;            
          }
        }
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Q> struct ParallelForFunctorTraits<hippoLBM::mrt<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
