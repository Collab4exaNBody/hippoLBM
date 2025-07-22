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

#include <grid/enum.hpp>
#include <grid/field_view.hpp>

namespace hippoLBM
{
  template<int DIM, Side S, int Q> struct cavity{};
  template<int dim, Side dir, int Q> struct cavity_coeff{};

  template<> struct cavity_coeff<DIMZ, Side::Left, 19> { int fid[5] = {5,14,11,18,15};};
  template<> struct cavity_coeff<DIMZ, Side::Right, 19> { int fid[5] = {6,13,12,17,16};};

  template<int Dim, Side S>
    struct cavity<Dim, S, 19>
    {
      static constexpr int Q = 19;
      static constexpr int Un = 5;
      double coeff[Un];

      void compute_coeff(
          double ux, double uy, double uz,
          const double * const w,
          const int* ex, const int* ey, const int* ez,
          int lx, int ly, int lz)
      {
        const cavity_coeff<DIMZ, S, Q> c_coeff;
        double L = 0;
        if constexpr (Dim == DIMZ) L = lx;
        if constexpr (Dim == DIMY) L = ly;
        if constexpr (Dim == DIMZ) L = lz;
        const double uxx = ux * ( 1 + 0.5/ ( L - 1 ));
        const double uyy = uy * ( 1 + 0.5/ ( L - 1 ));
        const double uzz = uz * ( 1 + 0.5/ ( L - 1 ));
        for(int i = 0 ; i < Un ; i++)
        {
          const int fid = c_coeff.fid[i];
          coeff[i] = 6. * w[fid] * (ex[fid] * uxx + ey[fid] * uyy + ez[fid] * uzz);
        }
      }

      ONIKA_HOST_DEVICE_FUNC inline void operator()(
          int idx, 
          int * const obst, 
          const FieldView<Un>& fi) const
      {
        if (obst[idx] == FLUIDE_) {
          for(int i = 0 ; i < Un ; i++)
          {
            fi(idx,i) += coeff[i];
          }
        }
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Dim, hippoLBM::Side S, int Q> struct ParallelForFunctorTraits<hippoLBM::cavity<Dim, S, Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
