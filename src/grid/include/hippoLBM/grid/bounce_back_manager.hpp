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
#include <onika/cuda/stl_adaptors.h>

#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/grid.hpp>
#include <stdexcept>
#include <string>

namespace hippoLBM {

/** @brief Helper function to calculate the index for a given dimension and side. */
template <int dim, Side dir>
inline constexpr int helper_dim_idx() {
  static_assert(dim < DIM_MAX);
  if constexpr (dir == Side::Right)
    return dim * 2 + 1;
  else
    return dim * 2;
}

/** @brief Maps a named bounce-back plane (e.g. "plan_xy_0") to its index in the
 * 2*DIM_MAX-sized plane arrays, matching helper_dim_idx's numbering. */
inline int bounce_back_plane_index(const std::string& name) {
  if (name == "plan_yz_0") return 0;
  if (name == "plan_yz_l") return 1;
  if (name == "plan_xz_0") return 2;
  if (name == "plan_xz_l") return 3;
  if (name == "plan_xy_0") return 4;
  if (name == "plan_xy_l") return 5;
  throw std::out_of_range("Unknown bounce-back plane name: " + name);
}

/** @brief Dimension (DIMX/DIMY/DIMZ) a bounce-back plane index belongs to. */
inline int bounce_back_plane_dim(int index) { return index / 2; }

/** @brief A manager for handling bounce-back boundary conditions. */
template <int Q>
struct bounce_back_manager {};

/** @brief Specialization of the bounce_back_manager for 19 discrete velocities. */
template <>
struct bounce_back_manager<19> {
  static constexpr int Un = 5;  ///<! The number of unknowns per point in the bounce-back manager.
  // data[0] : x left
  // data[1] : x right
  // data[2] : y left
  // data[3] : y right
  // data[4] : z bottom
  // data[5] : z top
  std::array<onika::memory::CudaMMVector<double>, 2 * DIM_MAX> _data_;

  /** @brief Get the data for a given dimension and side.
   * @param dim The dimension for which to get the data.
   * @tparam Un The number of unknowns per point.
   */
  FieldView<Un> get_data(int i) {
    assert(onika::cuda::vector_size(_data_[i]) % Un == 0);
    uint64_t size = onika::cuda::vector_size(_data_[i]) / Un;
    double* ptr = onika::cuda::vector_data(_data_[i]);
    return FieldView<Un>{ptr, size};
  }

  /** @brief Get the size of the data for a given dimension.
   * @param lgs The local grid size.
   * @tparam dim The dimension for which to get the size.
   */
  template <int dim>
  uint64_t get_size(const onika::math::IJK lgs) {
    if constexpr (dim == DIMX) return lgs.j * lgs.k;
    if constexpr (dim == DIMY) return lgs.i * lgs.k;
    if constexpr (dim == DIMZ) return lgs.i * lgs.j;
  }

  /** @brief Resize the data for a given dimension and side.
   * @param lgs The local grid size.
   * @tparam Dim The dimension for which to resize the data.
   * @tparam S The side for which to resize the data.
   */
  template <int Dim, Side S>
  void resize_data(const onika::math::IJK& lgs) {
    const uint64_t size_dim = get_size<Dim>(lgs) * Un;
    int i = helper_dim_idx<Dim, S>();
    auto& data = _data_[i];
    if (size_dim != onika::cuda::vector_size(data)) {
      data.resize(size_dim);
    }
  }

  /** @brief Resize the data for every plane that was explicitly requested and that this
   * rank actually sits at the boundary of.
   * @param active_planes Which of the 2*DIM_MAX named planes (indexed via
   *        bounce_back_plane_index / helper_dim_idx) should get a bounce-back buffer.
   * @param lgs The local grid size.
   * @param MPI_coord The MPI coordinates.
   * @param MPI_grid_size The MPI grid size.
   */
  void resize_data(const std::array<bool, 2 * DIM_MAX>& active_planes, const onika::math::IJK& lgs,
                   const onika::math::IJK& MPI_coord, const onika::math::IJK& MPI_grid_size) {
    if (active_planes[helper_dim_idx<DIMX, Left>()] && MPI_coord.i == 0) resize_data<DIMX, Left>(lgs);
    if (active_planes[helper_dim_idx<DIMX, Right>()] && MPI_coord.i == MPI_grid_size.i - 1)
      resize_data<DIMX, Right>(lgs);

    if (active_planes[helper_dim_idx<DIMY, Left>()] && MPI_coord.j == 0) resize_data<DIMY, Left>(lgs);
    if (active_planes[helper_dim_idx<DIMY, Right>()] && MPI_coord.j == MPI_grid_size.j - 1)
      resize_data<DIMY, Right>(lgs);

    if (active_planes[helper_dim_idx<DIMZ, Left>()] && MPI_coord.k == 0) resize_data<DIMZ, Left>(lgs);
    if (active_planes[helper_dim_idx<DIMZ, Right>()] && MPI_coord.k == MPI_grid_size.k - 1)
      resize_data<DIMZ, Right>(lgs);
  }
};
}  // namespace hippoLBM
