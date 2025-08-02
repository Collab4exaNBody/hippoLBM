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

#include<hippoLBM/grid/field_view.hpp>
#include<hippoLBM/grid/grid_helper.hpp>

namespace hippoLBM
{
  /**
   * @brief Initializes the distributions in a lattice Boltzmann model.
   */
  template<int Q>
    struct init_distributions
    {
      double coeff;
      GridIJKtoIdx ijk_to_idx;
      /**
       * @brief Operator to initialize distributions at a given index.
       *
       * @param idx The index to initialize distributions.
       * @param f Pointer to the distribution function.
       * @param w Pointer to the weight coefficients.
       */
      ONIKA_HOST_DEVICE_FUNC inline void operator()(const int idx, const FieldView<Q>& f, const double* const w) const
      {
        for (int iLB = 0; iLB < Q; iLB++)
        {
          f(idx,iLB) = coeff * w[iLB];
        }
      };

      ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k, const FieldView<Q>& f, const double* const w) const
      {
        this->operator()(ijk_to_idx(i,j,k), f, w);
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template <int Q> struct ParallelForFunctorTraits<hippoLBM::init_distributions<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true
        ;
    };
  }
}
