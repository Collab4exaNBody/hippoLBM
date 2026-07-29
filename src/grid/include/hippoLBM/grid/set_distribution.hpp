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

#include <onika/math/basic_types.h>
#include <onika/math/matrix4d.h>

#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/grid_helper.hpp>

namespace hippoLBM {

template <int Q>
struct SetDistributionFunc {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k, const FieldView<Q>& f, const double value,
                                                const LBMGrid& grid) const {
    const int idx = grid(i, j, k);
    stencil::for_each<typename LBMScheme<Q>::Coefficients>(
        [&]<typename coeff>(int iLB) { f(idx, iLB) = value * coeff::w; });
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k, const FieldView<Q>& f, const double value,
                                                const LBMGrid& grid, const onika::math::Mat4d& quadric) const {
    onika::math::Vec3d pos = grid.compute_position<Area::Global>(i, j, k);
    if (onika::math::quadric_eval(quadric, pos) <= 0.0) {
      const int idx = grid(i, j, k);
      stencil::for_each<typename LBMScheme<Q>::Coefficients>(
          [&]<typename coeff>(int iLB) { f(idx, iLB) = value * coeff::w; });
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::SetDistributionFunc<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
