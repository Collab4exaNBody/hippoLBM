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

namespace hippoLBM
{
  using namespace onika::math;
  /**
   * @brief A functor for computing macroscopic variables for lattice Boltzmann method.
   */
  template<int Q>
    struct macro_variables
    {
      const Vec3d Fext_2;
      ONIKA_HOST_DEVICE_FUNC inline void operator()(
          const int idx, 
          const FieldView<3>& pm1, 
          int * const pobst, 
          const FieldView<Q>& pf,
          double * const pm0,
          const int* pex, 
          const int* pey, 
          const int* pez) const
      {
        if (pobst[idx] >= FLUIDE_) 
        {
          double rho = 0.0;
          double ux  = 0.0;
          double uy  = 0.0;
          double uz  = 0.0;
          for (int iLB = 0; iLB < Q; iLB++) 
          {
            const double s = pf(idx,iLB);
            ux += s * pex[iLB];
            uy += s * pey[iLB];
            uz += s * pez[iLB];
            rho += s;
          }

          if (rho > 1.0e-14) 
          {
            ux /= rho;
            uy /= rho;
            uz /= rho;
          }

          pm0[idx] = rho;
          pm1(idx, 0) = ux;
          pm1(idx, 1) = uy;
          pm1(idx, 2) = uz;
        }
        else
        {
          pm1(idx, 0) = 0;
          pm1(idx, 1) = 0;
          pm1(idx, 2) = 0;
        }
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Q> struct ParallelForFunctorTraits<hippoLBM::macro_variables<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}

