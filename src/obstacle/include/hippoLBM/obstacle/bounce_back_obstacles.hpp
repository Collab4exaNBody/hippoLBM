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
#include <hippoLBM/grid/grid.hpp>

namespace hippoLBM {
namespace bcs {
//////////////////////// Wall streaming ///////////////////////////////

template <int Q>
struct wall_bounce_back {};

template <>
struct wall_bounce_back<19> {
  LBMGrid g_;
  const int* const obst_;
  const FieldView<19> f_;
  static constexpr int Q = 19;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const {
    const int idx = g_(coord.x, coord.y, coord.z);
    if (obst_[idx] == WALL_) {
      stencil::for_each<typename LBMScheme<19>::Coefficients, 1, Q>([&]<typename coeff>(int iLB) {
        const int next_x = coord.x + coeff::ex;
        const int next_y = coord.y + coeff::ey;
        const int next_z = coord.z + coeff::ez;
        if (g_.is_defined(next_x, next_y, next_z)) {
          const int idx_next = g_(next_x, next_y, next_z);
          if (obst_[idx_next] != WALL_)
            f_(idx, iLB) = f_(idx_next, coeff::iopp);  // call this function before the stream step
        }
      });
    }
  }
};
}  // namespace bcs
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::bcs::wall_bounce_back<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
