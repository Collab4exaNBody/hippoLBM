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

namespace hippoLBM {
struct MacroState {
  double rho, ux, uy, uz;
};

template <int Q>
ONIKA_HOST_DEVICE_FUNC inline MacroState compute_macro(int idx, const FieldView<Q>& f) {
  double rho = 0.0;
  double ux = 0.0;
  double uy = 0.0;
  double uz = 0.0;

  stencil::for_each<typename LBMScheme<Q>::Coefficients, 0, Q>([&]<typename coeff>(int iLB) {
    const double s = f(idx, iLB);
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
  return {rho, ux, uy, uz};
}

template <int Q>
ONIKA_HOST_DEVICE_FUNC inline void compute_macro_variables(int idx, int* const __restrict__ obst,
                                                            const FieldView<Q>& f, double* const __restrict__ m0,
                                                            const FieldView<3>& m1) {
  if (obst[idx] >= FLUIDE_) {
    const MacroState mv = compute_macro<Q>(idx, f);
    m0[idx] = mv.rho;
    m1(idx, 0) = mv.ux;
    m1(idx, 1) = mv.uy;
    m1(idx, 2) = mv.uz;
  } else {
    m1(idx, 0) = 0;
    m1(idx, 1) = 0;
    m1(idx, 2) = 0;
  }
}

template <int Q, Traversal TR>
struct MacroVariablesLauncher {
  const int* __restrict__ levels_;  // traversal level (0 inside, 0 1 Real, 0 1 2 Extend, 0 1 2 3 All)
  int* const __restrict__ obst_;    // Pointer to the obstacle field.
  const FieldView<Q> f_;            // The field view for the distribution functions.
  double* const __restrict__ m0_;   // Pointer to the density field (zeroth-order moment), written.
  const FieldView<3> m1_;           // The field view for the first-order moments (velocity), written.

  /**
   * @brief Computes the macroscopic variables at a given index.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    if (check_level<TR>(levels_[idx])) compute_macro_variables<Q>(idx, obst_, f_, m0_, m1_);
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal Tr>
struct ParallelForFunctorTraits<hippoLBM::MacroVariablesLauncher<Q, Tr>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
