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

#include <onika/memory/allocator.h>

#include <hippoLBM/grid/field_view.hpp>

namespace hippoLBM {

struct LBMScratchStreamingBuffer {
  onika::memory::CudaMMVector<double> f_;

  inline void resize(size_t grid_size, int Q) { f_.resize(grid_size * Q); }

  template <int Q>
  inline FieldView<Q> view(size_t grid_size) {
    return FieldView<Q>{onika::cuda::vector_data(f_), grid_size};
  }
};

template <int Q, Traversal Tr>
struct pull_streaming_step {
  const int* __restrict__ levels_;  // It contains the traversal level (0 inside, 0 1 Real, 0 1 2 Extend, and 0 1 2 3
                                    // All
  LBMGrid g_;                       // The LBM grid containing the lattice structure and related information.
  const FieldView<Q> src_;          // The live, pre-streaming distributions (post-collision), read-only.
  const FieldView<Q> dst_;          // The scratch buffer receiving the post-streaming distributions.

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    if (check_level<Tr>(levels_[idx])) {
      const auto [x, y, z] = g_(idx);
      dst_(idx, 0) = src_(idx, 0);  // the rest population does not move
      stencil::for_each<typename LBMScheme<Q>::Coefficients, 1, Q, 1>([&]<typename coeff>(int iLB) {
        const int prev_x = x - coeff::ex;
        const int prev_y = y - coeff::ey;
        const int prev_z = z - coeff::ez;

        if (g_.is_defined(prev_x, prev_y, prev_z)) {
          const int prev_idx = g_(prev_x, prev_y, prev_z);
          dst_(idx, iLB) = src_(prev_idx, iLB);
        } else {
          // no upstream neighbor (physical domain boundary): left for a later bcs operator
          dst_(idx, iLB) = src_(idx, iLB);
        }
      });
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal Tr>
struct ParallelForFunctorTraits<hippoLBM::pull_streaming_step<Q, Tr>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
