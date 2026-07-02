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

// TODO: This file is a temporary place for the macro_variables functor. It should be moved to a more appropriate
// location in the future, such as a compute or utility directory, to better organize the codebase.
#define FLUIDE_ -1

namespace hippoLBM {
/**
 * @brief A functor for computing macroscopic variables for lattice Boltzmann method.
 * @tparam Q The number of discrete velocity directions in the LBM scheme.
 */
template <int Q>
struct macro_variables {
  const onika::math::Vec3d
      Fext_2_;  // External force term divided by 2, used in the computation of macroscopic variables.

  /** @brief Computes the macroscopic variables for a given lattice node.
   * @param idx The index of the lattice node.
   * @param pm1 The field view for the macroscopic variables.
   * @param pobst The obstacle field.
   * @param pf The distribution functions.
   * @param pm0 The density field.
   * @param pex The x-components of the discrete velocity vectors.
   * @param pey The y-components of the discrete velocity vectors.
   * @param pez The z-components of the discrete velocity vectors.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(const int idx, const FieldView<3>& pm1, int* const pobst,
                                                const FieldView<Q>& pf, double* const pm0) const {
    if (pobst[idx] >= FLUIDE_) {
      double rho = 0.0;
      double ux = 0.0;
      double uy = 0.0;
      double uz = 0.0;

      stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, Q>([&]<typename coeff>(int iLB) {
        const double s = pf(idx, iLB);
        ux += s * coeff::ex;
        uy += s * coeff::ey;
        uz += s * coeff::ez;
        rho += s;
      });

      if (rho > 1.0e-14) {
        ux /= rho;
        uy /= rho;
        uz /= rho;
      }

      pm0[idx] = rho;
      pm1(idx, 0) = ux;
      pm1(idx, 1) = uy;
      pm1(idx, 2) = uz;
    } else {
      pm1(idx, 0) = 0;
      pm1(idx, 1) = 0;
      pm1(idx, 2) = 0;
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::macro_variables<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
