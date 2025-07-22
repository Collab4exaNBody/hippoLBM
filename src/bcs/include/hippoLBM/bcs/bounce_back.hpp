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

#include <grid/enum.hpp>

namespace hippoLBM
{
  constexpr Side LEFT = Side::Left;
  constexpr Side RIGHT = Side::Right;

  template<int dim, Side dir>
    inline constexpr Traversal get_traversal();

  template<> inline constexpr Traversal get_traversal<DIMX,  LEFT>() { return Traversal::Plan_yz_0; }
  template<> inline constexpr Traversal get_traversal<DIMX, RIGHT>() { return Traversal::Plan_yz_l; }
  template<> inline constexpr Traversal get_traversal<DIMY,  LEFT>() { return Traversal::Plan_xz_0; }
  template<> inline constexpr Traversal get_traversal<DIMY, RIGHT>() { return Traversal::Plan_xz_l; }
  template<> inline constexpr Traversal get_traversal<DIMZ,  LEFT>() { return Traversal::Plan_xy_0; }
  template<> inline constexpr Traversal get_traversal<DIMZ, RIGHT>() { return Traversal::Plan_xy_l; }

  ////////////////////// Pre streaming ///////////////////////////

  template<int dim, Side dir, int Q>
    struct pre_bounce_back {};

  template<int dim, Side dir, int Q> struct pre_bounce_back_coeff{};
  template<> struct pre_bounce_back_coeff<DIMX, LEFT, 19> { int fid[5] = {2,10,8,12,14};};
  template<> struct pre_bounce_back_coeff<DIMX, RIGHT,19> { int fid[5] = {1,9,7,11,13};  };

  template<> struct pre_bounce_back_coeff<DIMY, LEFT, 19> { int fid[5] = {4,8,9,16,18};  };
  template<> struct pre_bounce_back_coeff<DIMY, RIGHT,19> { int fid[5] = {3,7,10,15,17};};

  template<> struct pre_bounce_back_coeff<DIMZ, LEFT, 19> { int fid[5] = {6,13,12,17,16};};
  template<> struct pre_bounce_back_coeff<DIMZ, RIGHT,19> { int fid[5] = {5,14,11,18,15};};


  template<int Dim, Side S> struct pre_bounce_back<Dim, S, 19>
  {
    const int * const traversal; 
    static constexpr int Un = 5;
    pre_bounce_back_coeff<Dim, S, 19> coeff;
    ONIKA_HOST_DEVICE_FUNC inline void operator()(
        int idx, 
        const FieldView<19>& f, // data could be modified, but the ptr inside FieldView can't be modified
        const FieldView<Un>& fi) const
    {
      const int fidx = traversal[idx];
      for(int i = 0 ; i < 5 ; i++)
      {
        fi(idx, i) = f(fidx, coeff.fid[i]);
      }
    }
  };

  ////////////////////// Post streaming ///////////////////////////

  template<int Dim, Side S, int Q>
    struct post_bounce_back {};

  template<int Dim, Side S, int Q> struct post_bounce_back_coeff{};
  template<> struct post_bounce_back_coeff<DIMX, LEFT, 19> { int fid[5] = {1,9,7,11,13};};
  template<> struct post_bounce_back_coeff<DIMX, RIGHT,19> { int fid[5] = {2,10,8,12,14};  };

  template<> struct post_bounce_back_coeff<DIMY, LEFT, 19> { int fid[5] = {3,7,10,15,17};  };
  template<> struct post_bounce_back_coeff<DIMY, RIGHT,19> { int fid[5] = {4,8,9,16,18};};

  template<> struct post_bounce_back_coeff<DIMZ, LEFT, 19> { int fid[5] = {5,14,11,18,15};};
  template<> struct post_bounce_back_coeff<DIMZ, RIGHT,19> { int fid[5] = {6,13,12,17,16};};

  template<int Dim, Side S> struct post_bounce_back<Dim, S, 19>
  {
    const int * const traversal; 
    static constexpr int Un = 5;
    post_bounce_back_coeff<Dim, S, 19> coeff;
    ONIKA_HOST_DEVICE_FUNC inline void operator()(
        int idx, 
        const FieldView<19>& f, // data could be modified, but the ptr inside FieldView can't be modified
        const FieldView<Un>& fi) const
    {
      const int fidx = traversal[idx];
      for(int i = 0 ; i < 5 ; i++)
      {
        f(fidx, coeff.fid[i]) = fi(idx, i);
      }
    }
  };


  //////////////////////// Wall streaming ///////////////////////////////

  template<int Q> struct wall_bounce_back {};

  template<>
    struct wall_bounce_back<19>
    {
      grid<3> g;
      const int * const obst;
      const FieldView<19> f;
      const int* ex;
      const int* ey; 
      const int* ez;
      static constexpr int Q = 19;
      const int iopp[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

      ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const
      {
        const int idx = g(coord.x, coord.y, coord.z);
        if(obst[idx] == WALL_)
        {
          for(int iLB = 1 ; iLB < Q ; iLB++)
          {
            const int next_x = coord.x + ex[iLB];
            const int next_y = coord.y + ey[iLB];
            const int next_z = coord.z + ez[iLB];
            if(g.is_defined(next_x, next_y, next_z))
            {
              const int idx_next = g(next_x, next_y, next_z);
              if(obst[idx_next] != WALL_) f(idx, iLB) = f(idx_next, iopp[iLB]);
            }
          }
        }
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int Dim, hippoLBM::Side S, int Q> struct ParallelForFunctorTraits<hippoLBM::pre_bounce_back<Dim, S, Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };

    template<int Dim, hippoLBM::Side S, int Q> struct ParallelForFunctorTraits<hippoLBM::post_bounce_back<Dim, S, Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };

    template<int Q> struct ParallelForFunctorTraits<hippoLBM::wall_bounce_back<Q>>
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}
