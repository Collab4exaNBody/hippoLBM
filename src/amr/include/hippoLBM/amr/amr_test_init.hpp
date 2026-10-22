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

#include <cmath>

#include <onika/parallel/parallel_for.h>

#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/grid.hpp>
#include <hippoLBM/grid/scheme.hpp>
#include <hippoLBM/grid/stencil.hpp>

namespace hippoLBM {

/** @brief Equilibrium initialization at a localized Gaussian density bump (u=0 everywhere),
 * evaluated from compute_position() -- i.e. in physical space, so the SAME field is
 * reproduced regardless of a grid's resolution or offset. Used by both AMRTwoLevel's own
 * level setup and the standalone amr_init_gaussian_bump operator (for a plain, non-AMR
 * uniform-fine reference run used in quantitative convergence validation,
 * docs/AMR_PLAN.md): a uniform field can't exercise the C2F/F2C interpolation at all, and a
 * reference run needs the exact same initial condition to be comparable. */
template <int Q>
struct GaussianDensityInit {
  onika::math::Vec3d center_;
  double sigma_;
  double amplitude_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k, const FieldView<Q>& f, const LBMGrid& grid) const {
    const int idx = grid(i, j, k);
    const onika::math::Vec3d pos = grid.template compute_position<Area::Global>(i, j, k);
    const onika::math::Vec3d d = pos - center_;
    const double r2 = d.x * d.x + d.y * d.y + d.z * d.z;
    const double rho = 1.0 + amplitude_ * std::exp(-r2 / (2.0 * sigma_ * sigma_));
    stencil::for_each<typename LBMScheme<Q>::Coefficients>([&]<typename coeff>(int iLB) { f(idx, iLB) = rho * coeff::w; });
  }
};

}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::GaussianDensityInit<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
