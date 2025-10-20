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

#include <onika/math/basic_types_def.h>
#include <hippoLBM/grid/grid.hpp>

#define LEVEL_EXTEND 2
#define LEVEL_REAL 1
#define LEVEL_INSIDE 0


namespace hippoLBM
{
  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
  using ::onika::math::IJK;
  using namespace onika;
  using namespace onika::cuda;

  struct traversal_data
  {
    const int * const ptr;
    const size_t size;
  };


  template<Traversal Type>
    ONIKA_HOST_DEVICE_FUNC inline bool check_level(int level)
    {
      static_assert(Type != Traversal::Edge);
      static_assert(Type != Traversal::Ghost_Edge);
      static_assert(Type != Traversal::Plan_xy_0);
      static_assert(Type != Traversal::Plan_xy_l);
      static_assert(Type != Traversal::Plan_xz_0);
      static_assert(Type != Traversal::Plan_xz_l);
      static_assert(Type != Traversal::Plan_yz_0);
      static_assert(Type != Traversal::Plan_yz_l);

      if constexpr( Traversal::All )    return true;
      if constexpr( Traversal::Extend ) return level <= LEVEL_EXTEND;
      if constexpr( Traversal::Real )   return level <= LEVEL_REAL;
      if constexpr( Traversal::Inside ) return level == LEVEL_INSIDE;

    }

  struct LBMGridRegion
  {
    vector_t<int> level; // 0 inside, 1 real, 2 extend ,3 All
    vector_t<int> ghost_edge;
    vector_t<int> inside;
    vector_t<int> real;
    vector_t<int> all;
    vector_t<int> edge;
    vector_t<int> extend;
    vector_t<int> plane_xy_0, plane_xy_l;
    vector_t<int> plane_xz_0, plane_xz_l;
    vector_t<int> plane_yz_0, plane_yz_l;

    LBMGridRegion() {};

    template<Traversal Tr> traversal_data get_data();
    template<Traversal Tr> const traversal_data get_data() const;

    inline traversal_data get_levels()    
    { 
      return{ vector_data(level), vector_size(level)}; 
    }  

    void build_traversal(LBMGrid& G, const IJK MPI_coord, const IJK MPI_grid)
    {
      constexpr Area L = Area::Local;
      constexpr Traversal A = Traversal::All;
      constexpr Traversal R = Traversal::Real;
      constexpr Traversal I = Traversal::Inside;
      constexpr Traversal E = Traversal::Extend;
      auto ba = G.build_box<L, A>();
      auto br = G.build_box<L, R>();
      auto bi = G.build_box<L, I>();
      auto ex = G.build_box<L, E>();
      all.resize(ba.number_of_points());
      level.resize(ba.number_of_points());
      real.resize(br.number_of_points());
      inside.resize(bi.number_of_points());
      ghost_edge.resize(all.size() - inside.size());
      extend.resize(ex.number_of_points());

      size_t shift_a(0), shift_r(0), shift_i(0), shift_ge(0), shift_ex(0);
      for (int z = ba.start(2); z <= ba.end(2); z++) {
        for (int y = ba.start(1); y <= ba.end(1); y++) {
          for (int x = ba.start(0); x <= ba.end(0); x++) {
            Point3D p = {x, y, z};
            int idx = G(x, y, z);
            all[shift_a++] = idx;
            level[idx] = 3; // ALL
            if (ex.contains(p))
            {
              extend[shift_ex++] = idx;
              level[idx] = LEVEL_EXTEND;
              if (br.contains(p)) {
                real[shift_r++] = idx;
                level[idx] = LEVEL_REAL;
                if (bi.contains(p)) {
                  inside[shift_i++] = idx;
                  level[idx] = LEVEL_INSIDE;
                }
              }
            }

            if (!bi.contains(p)) {
              ghost_edge[shift_ge++] = idx;
            }
          }
        }
      }

      assert(shift_ex == extend.size());
      assert(shift_i == inside.size());
      assert(shift_r == real.size());
      assert(shift_a == all.size());
      assert(shift_ge == ghost_edge.size());

      // used by bcs functors
      int plane_size_xy = ba.get_length(0) * ba.get_length(1);
      int plane_size_xz = ba.get_length(0) * ba.get_length(2);
      int plane_size_yz = ba.get_length(1) * ba.get_length(2);
      int idx_xy0(0), idx_xyl(0);
      int idx_xz0(0), idx_xzl(0);
      int idx_yz0(0), idx_yzl(0);

      int plane_0x = br.start(0);
      int plane_0y = br.start(1);
      int plane_0z = br.start(2);
      int plane_lx = br.end(0);
      int plane_ly = br.end(1);
      int plane_lz = br.end(2);


      if (MPI_coord.i == 0)
        plane_yz_0.resize(plane_size_yz);
      if (MPI_coord.i == MPI_grid.i - 1)
        plane_yz_l.resize(plane_size_yz);

      if (MPI_coord.j == 0)
        plane_xz_0.resize(plane_size_xz);
      if (MPI_coord.j == MPI_grid.j - 1)
        plane_xz_l.resize(plane_size_xz);

      if (MPI_coord.k == 0)
        plane_xy_0.resize(plane_size_xy);
      if (MPI_coord.k == MPI_grid.k - 1)
        plane_xy_l.resize(plane_size_xy);

      // Plan XY
      for (int y = ba.start(1); y <= ba.end(1); y++) {
        for (int x = ba.start(0); x <= ba.end(0); x++) {
          if (MPI_coord.k == 0)
          {
            plane_xy_0[idx_xy0++] = G(x, y, plane_0z);
          }
          if (MPI_coord.k == MPI_grid.k - 1)
          {
            //onika::lout << "( "<<x << " , " << y << " , " << plane_lz << " )" << std::endl; 
            plane_xy_l[idx_xyl++] = G(x, y, plane_lz);
          }
        }
      }

      // debug: onika::lout << "Last idx_xyl= " << idx_xyl << " Plane size xyl: " << plane_size_xy << std::endl;

      // Plan XZ
      for (int z = ba.start(2); z <= ba.end(2); z++) {
        for (int x = ba.start(0); x <= ba.end(0); x++) {
          if (MPI_coord.j == 0)
            plane_xz_0[idx_xz0++] = G(x, plane_0y, z);
          if (MPI_coord.j == MPI_grid.j - 1)
            plane_xz_l[idx_xzl++] = G(x, plane_ly, z);
        }
      }

      // Plane YZ
      for (int z = ba.start(2); z <= ba.end(2); z++) {
        for (int y = ba.start(1); y <= ba.end(1); y++) {
          if (MPI_coord.i == 0)
            plane_yz_0[idx_yz0++] = G(plane_0x, y, z);
          if (MPI_coord.i == MPI_grid.i - 1)
            plane_yz_l[idx_yzl++] = G(plane_lx, y, z);
        }
      }

      if (MPI_coord.k == 0)
        assert(idx_xy0 == plane_size_xy);
      if (MPI_coord.k == MPI_grid.k - 1)
        assert(idx_xyl == plane_size_xy);
    }
  };

  template<> inline traversal_data LBMGridRegion::get_data<Traversal::All>()        { return{ vector_data(all), vector_size(all)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Real>()       { return{ vector_data(real), vector_size(real)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Extend>()     { return{ vector_data(extend), vector_size(extend)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Inside>()     { return{ vector_data(inside), vector_size(inside)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Edge>()       { return{ vector_data(edge), vector_size(edge)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Ghost_Edge>() { return{ vector_data(ghost_edge), vector_size(ghost_edge)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_0>()  { return{ vector_data(plane_xy_0), vector_size(plane_xy_0)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_l>()  { return{ vector_data(plane_xy_l), vector_size(plane_xy_l)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_0>()  { return{ vector_data(plane_xz_0), vector_size(plane_xz_0)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_l>()  { return{ vector_data(plane_xz_l), vector_size(plane_xz_l)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_0>()  { return{ vector_data(plane_yz_0), vector_size(plane_yz_0)}; }  
  template<> inline traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_l>()  { return{ vector_data(plane_yz_l), vector_size(plane_yz_l)}; }  
  template<> const inline traversal_data LBMGridRegion::get_data<Traversal::All>() const        { return{ vector_data(all), vector_size(all)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Real>() const       { return{ vector_data(real), vector_size(real)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Extend>() const     { return{ vector_data(extend), vector_size(extend)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Inside>() const     { return{ vector_data(inside), vector_size(inside)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Edge>() const       { return{ vector_data(edge), vector_size(edge)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Ghost_Edge>() const { return{ vector_data(ghost_edge), vector_size(ghost_edge)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_0>() const  { return{ vector_data(plane_xy_0), vector_size(plane_xy_0)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_l>() const  { return{ vector_data(plane_xy_l), vector_size(plane_xy_l)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_0>() const  { return{ vector_data(plane_xz_0), vector_size(plane_xz_0)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_l>() const  { return{ vector_data(plane_xz_l), vector_size(plane_xz_l)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_0>() const  { return{ vector_data(plane_yz_0), vector_size(plane_yz_0)}; }  
  template<> inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_l>() const  { return{ vector_data(plane_yz_l), vector_size(plane_yz_l)}; }  

  using traversal_getter_t = traversal_data (*)(LBMGridRegion&);

  constexpr std::array<traversal_getter_t, 12> traversal_table = {
    +[](LBMGridRegion& r) { return r.get_data<Traversal::All>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Real>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Extend>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Inside>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Edge>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Ghost_Edge>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_0>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_l>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_0>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_l>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_0>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_l>(); }
  };

  using const_traversal_getter_t = traversal_data (*)(const LBMGridRegion&);

  constexpr std::array<const_traversal_getter_t, 12> const_traversal_table = {
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::All>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Real>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Extend>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Inside>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Edge>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Ghost_Edge>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_0>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_l>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_0>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_l>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_0>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_l>(); }
  };

  inline traversal_data get_traversal(LBMGridRegion& region, Traversal Tr)
  {
    const std::size_t idx = static_cast<std::size_t>(Tr);
    if (idx >= traversal_table.size())
    {
      throw std::out_of_range("Invalid traversal type");
    }
    return traversal_table[idx](region);
  }

  inline traversal_data get_traversal(const LBMGridRegion& region, Traversal Tr)
  {
    const std::size_t idx = static_cast<std::size_t>(Tr);
    if (idx >= const_traversal_table.size())
    {
      throw std::out_of_range("Invalid traversal type");
    }
    return const_traversal_table[idx](region);
  }
};
