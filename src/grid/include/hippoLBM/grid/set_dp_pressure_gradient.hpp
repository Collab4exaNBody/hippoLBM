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

#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid.hpp>
#include <hippoLBM/grid/scheme.hpp>

namespace hippoLBM {
/**
 * @brief Sets the distributions at equilibrium (zero velocity) with a density varying linearly along one dimension:
 * rho = rho_0 at x_0 and rho_1 at x_1, clamped outside [x_0, x_1].
 * @tparam Dim dimension of the gradient (DIMX, DIMY or DIMZ)
 */
template <int Q, int Dim>
struct SetDPGradientFunc {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k, const FieldView<Q>& f, const LBMGrid& grid,
                                                const double x_0, const double x_1, const double rho_0,
                                                const double rho_1) const {
    const onika::math::Vec3d pos = grid.compute_position<Area::Global>(i, j, k);
    double x;
    if constexpr (Dim == DIMX) x = pos.x;
    if constexpr (Dim == DIMY) x = pos.y;
    if constexpr (Dim == DIMZ) x = pos.z;
    double t = (x - x_0) / (x_1 - x_0);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    const double rho = rho_0 + t * (rho_1 - rho_0);
    const int idx = grid(i, j, k);
    stencil::for_each<typename LBMScheme<Q>::Coefficients>([&]<typename coeff>(int iLB) { f(idx, iLB) = rho * coeff::w; });
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, int Dim>
struct ParallelForFunctorTraits<hippoLBM::SetDPGradientFunc<Q, Dim>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
