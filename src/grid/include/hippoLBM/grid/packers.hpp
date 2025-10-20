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

namespace hippoLBM
{
  template<int Components>
    struct packer
    {
      FieldView<Components> dst;
      FieldView<Components> src; 
      Box3D dst_box; 
      Box3D mesh_box;

      ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const
      {
        const auto& inf = dst_box.inf;
        const int dst_idx = compute_idx(dst_box, coord.x - inf[0], coord.y-inf[1], coord.z-inf[2]);
        const int src_idx = compute_idx(mesh_box, coord.x, coord.y, coord.z);
        copyTo<Components>(dst, dst_idx, src, src_idx, 1);
      }
    };

  template<int Components>
    struct unpacker
    {
      FieldView<Components> dst;
      FieldView<Components> src;
      Box3D src_box;
      Box3D mesh_box;

      ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const
      {
        const auto& inf = src_box.inf;
        const int dst_idx = compute_idx(mesh_box, coord.x , coord.y , coord.z);
        const int src_idx = compute_idx(src_box, coord.x - inf[0], coord.y - inf[1], coord.z - inf[2]);
        copyTo<Components>(dst, dst_idx, src, src_idx, 1);
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int C> struct ParallelForFunctorTraits< hippoLBM::packer<C> >
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };

    template<int C> struct ParallelForFunctorTraits< hippoLBM::unpacker<C> >
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}

