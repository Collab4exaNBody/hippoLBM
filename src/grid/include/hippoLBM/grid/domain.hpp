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

#include <hippoLBM/core/box3d.hpp>
#include <hippoLBM/grid/ghost_manager.hpp>
#include <hippoLBM/grid/grid.hpp>

namespace hippoLBM {
template <int Q>
struct LBMDomain {
  LBMGhostManager<Q> m_ghost_manager_;  //< The ghost manager for handling ghost cell communication.
  Box3D m_box_;                        //< The computational box representing the local domain, including ghost layers.
  LBMGrid m_grid_;                     //< The LBM grid containing the distribution functions and macroscopic variables.
  onika::math::AABB bounds_;           //< The axis-aligned bounding box representing the physical domain boundaries.
  int3d domain_size_;               //< The size of the local domain in terms of the number of nodes in each dimension.
  onika::math::IJK MPI_coord_;      //< The MPI coordinates of the current process in the Cartesian communicator.
  onika::math::IJK MPI_grid_size_;  //< The size of the MPI grid in terms of the number of processes in each dimension.
  LBMDomain() {};
  LBMDomain(LBMGhostManager<Q>& g, Box3D& b, LBMGrid& gr, onika::math::AABB& bd, int3d& ds, onika::math::IJK& mc,
            onika::math::IJK& mgs)
      : m_ghost_manager_(g), m_box_(b), m_grid_(gr), bounds_(bd), domain_size_(ds), MPI_coord_(mc), MPI_grid_size_(mgs) {}

  /** @brief Get the size of the local domain.
   * @return The size of the local domain in terms of the number of nodes in each dimension.
   */
  int3d size() { return domain_size_; }  // return the number of nodes in each dimension, without ghost_layers

  /** @brief Get the grid spacing.
   * @return The grid spacing.
   */
  double dx() { return m_grid_.dx_; }

  /** @brief Get the LBM grid.
   * @return Reference to the LBM grid.
   */
  LBMGrid& grid() { return m_grid_; }
};
};  // namespace hippoLBM
