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

#include <onika/parallel/parallel_for.h>

#include <hippoLBM/grid/field_view.hpp>

namespace hippoLBM {
/** @brief A packer for transferring data between fields.
 * @tparam Components The number of components in the field.
 */
template <int Components>
struct packer {
  FieldView<Components> dst_;  // Pointer to the destination field.
  FieldView<Components> src_;  // Pointer to the source field.
  Box3D dst_box_;              // The box defining the region in the destination field to be packed.
  Box3D mesh_box_;             // The box defining the region in the source field to be packed.

  /** @brief The operator for packing data.
   * @param coord The coordinate of the cell to be packed.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const {
    const auto& inf = dst_box_.inf_;
    const int dst_idx = compute_idx(dst_box_, coord.x - inf[0], coord.y - inf[1], coord.z - inf[2]);
    const int src_idx = compute_idx(mesh_box_, coord.x, coord.y, coord.z);
    copyTo<Components>(dst_, dst_idx, src_, src_idx, 1);
  }
};

/** @brief An unpacker for transferring data between fields.
 * @tparam Components The number of components in the field.
 */
template <int Components>
struct unpacker {
  FieldView<Components> dst_;  // Pointer to the destination field.
  FieldView<Components> src_;  // Pointer to the source field.
  Box3D src_box_;              // The box defining the region in the source field to be unpacked.
  Box3D mesh_box_;             // The box defining the region in the destination field to be unpacked.

  /** @brief The operator for unpacking data.
   * @param coord The coordinate of the cell to be unpacked.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const {
    const auto& inf = src_box_.inf_;
    const int dst_idx = compute_idx(mesh_box_, coord.x, coord.y, coord.z);
    const int src_idx = compute_idx(src_box_, coord.x - inf[0], coord.y - inf[1], coord.z - inf[2]);
    copyTo<Components>(dst_, dst_idx, src_, src_idx, 1);
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int C>
struct ParallelForFunctorTraits<hippoLBM::packer<C> > {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <int C>
struct ParallelForFunctorTraits<hippoLBM::unpacker<C> > {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
