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

namespace hippoLBM {
/**
 * @brief Operator for performing the first step of streaming at a given index.
 * @param idx The index.
 * @param f Pointer to an array of doubles representing distribution functions.
 */
template <int Q, Traversal Tr>
struct streaming_step1 {
  const int* __restrict__ levels;  // It contains the traversal level (0 inside, 0 1 Real, 0 1 2 Extend, and 0 1 2 3 All
  const FieldView<Q> f;            // The field view for the distribution functions.
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    if (check_level<Tr>(levels[idx])) {
      for (int iLB = 1; iLB < Q; iLB += 2) {
        std::swap(f(idx, iLB), f(idx, iLB + 1));
      }
    }
  }
};

/**
 * @brief A functor for the second step of streaming in the lattice Boltzmann method for XYZ directions.
 */
template <int Q, Traversal Tr>
struct streaming_step2 {
  const int* __restrict__ levels;  // It contains the traversal level (0 inside, 0 1 Real, 0 1 2 Extend, and 0 1 2 3 All
  LBMGrid g;                       // The LBM grid containing the lattice structure and related information.
  const FieldView<Q> f;            // The field view for the distribution functions.
  const int* __restrict__ const ex;  // Pointer to an array of integers for X-direction.
  const int* __restrict__ const ey;  // Pointer to an array of integers for Y-direction.
  const int* __restrict__ const ez;  // Pointer to an array of integers for Z-direction.
  /**
   * @brief Operator for performing the second step of streaming at given coordinates (x, y, z).
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const {
    const int idx = g(coord.x, coord.y, coord.z);
    for (int iLB = 1; iLB < Q; iLB += 2) {
      const int next_x = coord.x + ex[iLB];
      const int next_y = coord.y + ey[iLB];
      const int next_z = coord.z + ez[iLB];

      if (g.is_defined(next_x, next_y, next_z)) {
        const int next_idx = g(next_x, next_y, next_z);
        std::swap(f(idx, iLB + 1), f(next_idx, iLB));
      }
    }
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    if (check_level<Tr>(levels[idx])) {
      auto [x, y, z] = g(idx);
      for (int iLB = 1; iLB < Q; iLB += 2) {
        const int next_x = x + ex[iLB];
        const int next_y = y + ey[iLB];
        const int next_z = z + ez[iLB];

        if (g.is_defined(next_x, next_y, next_z)) {
          const int next_idx = g(next_x, next_y, next_z);
          std::swap(f(idx, iLB + 1), f(next_idx, iLB));
        }
      }
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q, hippoLBM::Traversal Tr>
struct ParallelForFunctorTraits<hippoLBM::streaming_step1<Q, Tr>> {
  static inline constexpr bool RequiresBlockSynchronousCall = true;
  static inline constexpr bool CudaCompatible = true;
};

template <int Q, hippoLBM::Traversal Tr>
struct ParallelForFunctorTraits<hippoLBM::streaming_step2<Q, Tr>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
