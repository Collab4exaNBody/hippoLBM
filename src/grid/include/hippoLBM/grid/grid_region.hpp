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
#include <string>
#include <vector>

#define LEVEL_EXTEND 2
#define LEVEL_REAL 1
#define LEVEL_INSIDE 0

namespace hippoLBM {
using ::onika::math::IJK;
using namespace onika;
using namespace onika::cuda;

/** @brief Data structure for storing traversal information. */
struct traversal_data {
  const int* const ptr_;  // Pointer to the traversal data.
  const size_t size_;     // Size of the traversal data.
};

/** @brief Check if a level is within the specified traversal type.
 * @param level The level to check.
 * @tparam Type The traversal type to check against.
 */
template <Traversal Type>
ONIKA_HOST_DEVICE_FUNC inline bool check_level(int level) {
  static_assert(Type != Traversal::Edge);
  static_assert(Type != Traversal::Ghost_Edge);
  static_assert(Type != Traversal::Plan_xy_0);
  static_assert(Type != Traversal::Plan_xy_l);
  static_assert(Type != Traversal::Plan_xz_0);
  static_assert(Type != Traversal::Plan_xz_l);
  static_assert(Type != Traversal::Plan_yz_0);
  static_assert(Type != Traversal::Plan_yz_l);

  if constexpr (Type == Traversal::All) return true;
  if constexpr (Type == Traversal::Extend) return level <= LEVEL_EXTEND;
  if constexpr (Type == Traversal::Real) return level <= LEVEL_REAL;
  if constexpr (Type == Traversal::Inside) return level == LEVEL_INSIDE;
}

/** @brief Data structure for storing grid region information. */
struct LBMGridRegion {
  template <typename T>
  using vector_t = onika::memory::CudaMMVector<T>;
  vector_t<int> level_;  // 0 inside_, 1 real_, 2 extend_ ,3 All
  vector_t<int> ghost_edge_;
  vector_t<int> inside_;
  vector_t<int> real_;
  vector_t<int> all_;
  vector_t<int> edge_;
  vector_t<int> extend_;
  vector_t<int> plane_xy_0_, plane_xy_l_;
  vector_t<int> plane_xz_0_, plane_xz_l_;
  vector_t<int> plane_yz_0_, plane_yz_l_;

  LBMGridRegion() {};

  template <Traversal Tr>
  traversal_data get_data();

  template <Traversal Tr>
  const traversal_data get_data() const;

  inline traversal_data get_levels() { return {vector_data(level_), vector_size(level_)}; }

  void build_traversal(LBMGrid& G, const onika::math::IJK MPI_coord, const onika::math::IJK MPI_grid) {
    constexpr Area L = Area::Local;
    constexpr Traversal A = Traversal::All;
    constexpr Traversal R = Traversal::Real;
    constexpr Traversal I = Traversal::Inside;
    constexpr Traversal E = Traversal::Extend;
    auto ba = G.build_box<L, A>();
    auto br = G.build_box<L, R>();
    auto bi = G.build_box<L, I>();
    auto ex = G.build_box<L, E>();
    all_.resize(ba.number_of_points());
    level_.resize(ba.number_of_points());
    real_.resize(br.number_of_points());
    inside_.resize(bi.number_of_points());
    ghost_edge_.resize(all_.size() - inside_.size());
    extend_.resize(ex.number_of_points());

    size_t shift_a(0), shift_r(0), shift_i(0), shift_ge(0), shift_ex(0);
    for (int z = ba.start(2); z <= ba.end(2); z++) {
      for (int y = ba.start(1); y <= ba.end(1); y++) {
        for (int x = ba.start(0); x <= ba.end(0); x++) {
          Point3D p = {x, y, z};
          int idx = G(x, y, z);
          all_[shift_a++] = idx;
          level_[idx] = 3;  // ALL
          if (ex.contains(p)) {
            extend_[shift_ex++] = idx;
            level_[idx] = LEVEL_EXTEND;
            if (br.contains(p)) {
              real_[shift_r++] = idx;
              level_[idx] = LEVEL_REAL;
              if (bi.contains(p)) {
                inside_[shift_i++] = idx;
                level_[idx] = LEVEL_INSIDE;
              }
            }
          }

          if (!bi.contains(p)) {
            ghost_edge_[shift_ge++] = idx;
          }
        }
      }
    }

    assert(shift_ex == extend_.size());
    assert(shift_i == inside_.size());
    assert(shift_r == real_.size());
    assert(shift_a == all_.size());
    assert(shift_ge == ghost_edge_.size());

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

    if (MPI_coord.i == 0) plane_yz_0_.resize(plane_size_yz);
    if (MPI_coord.i == MPI_grid.i - 1) plane_yz_l_.resize(plane_size_yz);

    if (MPI_coord.j == 0) plane_xz_0_.resize(plane_size_xz);
    if (MPI_coord.j == MPI_grid.j - 1) plane_xz_l_.resize(plane_size_xz);

    if (MPI_coord.k == 0) plane_xy_0_.resize(plane_size_xy);
    if (MPI_coord.k == MPI_grid.k - 1) plane_xy_l_.resize(plane_size_xy);

    // Plan XY
    for (int y = ba.start(1); y <= ba.end(1); y++) {
      for (int x = ba.start(0); x <= ba.end(0); x++) {
        if (MPI_coord.k == 0) {
          plane_xy_0_[idx_xy0++] = G(x, y, plane_0z);
        }
        if (MPI_coord.k == MPI_grid.k - 1) {
          // onika::lout << "( "<<x << " , " << y << " , " << plane_lz << " )" << std::endl;
          plane_xy_l_[idx_xyl++] = G(x, y, plane_lz);
        }
      }
    }

    // debug: onika::lout << "Last idx_xyl= " << idx_xyl << " Plane size xyl: " << plane_size_xy << std::endl;

    // Plan XZ
    for (int z = ba.start(2); z <= ba.end(2); z++) {
      for (int x = ba.start(0); x <= ba.end(0); x++) {
        if (MPI_coord.j == 0) plane_xz_0_[idx_xz0++] = G(x, plane_0y, z);
        if (MPI_coord.j == MPI_grid.j - 1) plane_xz_l_[idx_xzl++] = G(x, plane_ly, z);
      }
    }

    // Plane YZ
    for (int z = ba.start(2); z <= ba.end(2); z++) {
      for (int y = ba.start(1); y <= ba.end(1); y++) {
        if (MPI_coord.i == 0) plane_yz_0_[idx_yz0++] = G(plane_0x, y, z);
        if (MPI_coord.i == MPI_grid.i - 1) plane_yz_l_[idx_yzl++] = G(plane_lx, y, z);
      }
    }

    if (MPI_coord.k == 0) assert(idx_xy0 == plane_size_xy);
    if (MPI_coord.k == MPI_grid.k - 1) assert(idx_xyl == plane_size_xy);
  }
};

template <>
inline traversal_data LBMGridRegion::get_data<Traversal::All>() {
  return {vector_data(all_), vector_size(all_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Real>() {
  return {vector_data(real_), vector_size(real_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Extend>() {
  return {vector_data(extend_), vector_size(extend_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Inside>() {
  return {vector_data(inside_), vector_size(inside_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Edge>() {
  return {vector_data(edge_), vector_size(edge_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Ghost_Edge>() {
  return {vector_data(ghost_edge_), vector_size(ghost_edge_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_0>() {
  return {vector_data(plane_xy_0_), vector_size(plane_xy_0_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_l>() {
  return {vector_data(plane_xy_l_), vector_size(plane_xy_l_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_0>() {
  return {vector_data(plane_xz_0_), vector_size(plane_xz_0_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_l>() {
  return {vector_data(plane_xz_l_), vector_size(plane_xz_l_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_0>() {
  return {vector_data(plane_yz_0_), vector_size(plane_yz_0_)};
}
template <>
inline traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_l>() {
  return {vector_data(plane_yz_l_), vector_size(plane_yz_l_)};
}
template <>
const inline traversal_data LBMGridRegion::get_data<Traversal::All>() const {
  return {vector_data(all_), vector_size(all_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Real>() const {
  return {vector_data(real_), vector_size(real_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Extend>() const {
  return {vector_data(extend_), vector_size(extend_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Inside>() const {
  return {vector_data(inside_), vector_size(inside_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Edge>() const {
  return {vector_data(edge_), vector_size(edge_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Ghost_Edge>() const {
  return {vector_data(ghost_edge_), vector_size(ghost_edge_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_0>() const {
  return {vector_data(plane_xy_0_), vector_size(plane_xy_0_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xy_l>() const {
  return {vector_data(plane_xy_l_), vector_size(plane_xy_l_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_0>() const {
  return {vector_data(plane_xz_0_), vector_size(plane_xz_0_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_xz_l>() const {
  return {vector_data(plane_xz_l_), vector_size(plane_xz_l_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_0>() const {
  return {vector_data(plane_yz_0_), vector_size(plane_yz_0_)};
}
template <>
inline const traversal_data LBMGridRegion::get_data<Traversal::Plan_yz_l>() const {
  return {vector_data(plane_yz_l_), vector_size(plane_yz_l_)};
}

using traversal_getter_t = traversal_data (*)(LBMGridRegion&);

constexpr std::array<traversal_getter_t, 12> traversal_table = {
    +[](LBMGridRegion& r) { return r.get_data<Traversal::All>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Real>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Inside>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Edge>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Ghost_Edge>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_0>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_l>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_0>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_l>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_0>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_l>(); },
    +[](LBMGridRegion& r) { return r.get_data<Traversal::Extend>(); }};

using const_traversal_getter_t = traversal_data (*)(const LBMGridRegion&);

constexpr std::array<const_traversal_getter_t, 12> const_traversal_table = {
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::All>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Real>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Inside>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Edge>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Ghost_Edge>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_0>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xy_l>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_0>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_xz_l>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_0>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Plan_yz_l>(); },
    +[](const LBMGridRegion& r) { return r.get_data<Traversal::Extend>(); }};

/** @brief Get traversal data for a mutable grid region.
 * @param region The grid region to retrieve traversal data from.
 * @param Tr The traversal type to retrieve.
 * @return The traversal data for the specified type.
 */
inline traversal_data get_traversal(LBMGridRegion& region, Traversal Tr) {
  const std::size_t idx = static_cast<std::size_t>(Tr);
  if (idx >= traversal_table.size()) {
    throw std::out_of_range("Invalid traversal type");
  }
  return traversal_table[idx](region);
}

/** @brief Get traversal data for a constant grid region.
 * @param region The grid region to retrieve traversal data from.
 * @param Tr The traversal type to retrieve.
 * @return The traversal data for the specified type.
 */
inline traversal_data get_traversal(const LBMGridRegion& region, Traversal Tr) {
  const std::size_t idx = static_cast<std::size_t>(Tr);
  if (idx >= const_traversal_table.size()) {
    throw std::out_of_range("Invalid traversal type");
  }
  return const_traversal_table[idx](region);
}

/** @brief Convert a traversal name (e.g. "real", "ghost_edge", "plan_xy_l") to a Traversal value. */
inline Traversal traversal_from_string(const std::string& name) {
  if (name == "all") return Traversal::All;
  if (name == "real") return Traversal::Real;
  if (name == "inside") return Traversal::Inside;
  if (name == "edge") return Traversal::Edge;
  if (name == "ghost" || name == "ghost_edge") return Traversal::Ghost_Edge;
  if (name == "plan_xy_0") return Traversal::Plan_xy_0;
  if (name == "plan_xy_l") return Traversal::Plan_xy_l;
  if (name == "plan_xz_0") return Traversal::Plan_xz_0;
  if (name == "plan_xz_l") return Traversal::Plan_xz_l;
  if (name == "plan_yz_0") return Traversal::Plan_yz_0;
  if (name == "plan_yz_l") return Traversal::Plan_yz_l;
  if (name == "extend") return Traversal::Extend;
  throw std::out_of_range("Unknown traversal name: " + name);
}

/** @brief Get traversal data for several traversal names at once (e.g. {"real", "ghost", "plan_xy_l"}). */
inline std::vector<traversal_data> get_traversal(LBMGridRegion& region, const std::vector<std::string>& names) {
  std::vector<traversal_data> result;
  result.reserve(names.size());
  for (const auto& name : names) {
    result.push_back(get_traversal(region, traversal_from_string(name)));
  }
  return result;
}

/** @brief Get traversal data for several traversal names at once (e.g. {"real", "ghost", "plan_xy_l"}). */
inline std::vector<traversal_data> get_traversal(const LBMGridRegion& region, const std::vector<std::string>& names) {
  std::vector<traversal_data> result;
  result.reserve(names.size());
  for (const auto& name : names) {
    result.push_back(get_traversal(region, traversal_from_string(name)));
  }
  return result;
}
};  // namespace hippoLBM
