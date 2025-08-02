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

#include <grid/box.hpp>
#include <grid/grid.hpp>

namespace hippoLBM
{
  /**
   * @brief Build communication coordinates based on relative positions, domain size, and periodic boundary conditions.
   *
   * @tparam DIM The dimension of the coordinate space.
   * @param relative_pos The relative position.
   * @param coord The current coordinate.
   * @param domain_size The size of the domain.
   * @param periods An array indicating whether each dimension has periodic boundary conditions (1 for periodic, 0 for non-periodic).
   * @param ndims An array of the number of dimensions in each dimension.
   * @return A tuple containing a boolean flag indicating whether the communication coordinate is valid and the computed communication coordinate.
   */
  template<int DIM>
    std::tuple<bool, int3d> build_comm(const int3d& relative_pos, const int* coord, const int3d& domain_size, const int* periods, const int* ndims)
    {
      int3d coord_neig;
      for (int dim = 0; dim < DIM; dim++) {
        coord_neig[dim] = coord[dim] + relative_pos[dim];
        if (coord_neig[dim] == -1) {
          if (periods[dim] == 1) {
            coord_neig[dim] += ndims[dim];
          } else { // does not exist
            return {false, coord_neig};   
          }
        } else if (coord_neig[dim] == ndims[dim]) {
          if (periods[dim] == 1) {
            coord_neig[dim] -= ndims[dim];
          } else { // does not exist
            return {false, coord_neig};
          }
        }
      }
      return {true, coord_neig};
    }

  /**
   * @brief Build send and receive boxes based on a shift and local size in a multi-dimensional space.
   *
   * @tparam DIM The dimension of the coordinate space.
   * @param shift The shift vector indicating the direction of communication.
   * @param local_size The size of the local box in each dimension.
   * @return A tuple containing the send and receive boxes.
   */
  template<int DIM>
    std::tuple<box<DIM>, box<DIM>> build_boxes(const int3d& shift, grid<3>& g)
    {
      auto real = g.build_box<Local, Real>();
      const int ghost_layer = g.ghost_layer;
      box<DIM> send = real;
      box<DIM> recv = real;
      auto lower = real.lower();
      auto upper = real.upper();
      static_assert(DIM>=1);
      for(int dim = 0; dim < DIM ; dim++)
      {
        if(shift[dim] == -1) 
        {
          send.sup.set_val(dim, lower[dim] + ghost_layer - 1);
          recv.inf.set_val(dim, lower[dim] - ghost_layer);
          recv.sup.set_val(dim, lower[dim] - ghost_layer + 1);
          assert(recv.get_length(dim) == ghost_layer);
          assert(send.get_length(dim) == ghost_layer);
        }
        if(shift[dim] == 1) 
        {
          send.inf.set_val(dim, upper[dim] - ghost_layer + 1);
          recv.inf.set_val(dim, upper[dim] + ghost_layer - 1);
          recv.sup.set_val(dim, upper[dim] + ghost_layer);
          assert(recv.get_length(dim) == ghost_layer);
          assert(send.get_length(dim) == ghost_layer);
        }
      }
      return {send,recv};
    }

  template<int DIM>
    box<DIM> fix_box_with_periodicity(const int3d& shift, grid<3>& g)
    {
      auto real = g.build_box<Local, Real>();
      const int ghost_layer = g.ghost_layer;
      box<DIM> send = real;
      auto lower = real.lower();
      auto upper = real.upper();
      static_assert(DIM>=1);
      for(int dim = 0; dim < DIM ; dim++)
      {
        if(shift[dim] == -1) 
        {
          send.inf.set_val(dim, upper[dim] - ghost_layer + 1);
          assert(send.get_length(dim) == ghost_layer);
        }
        if(shift[dim] == 1) 
        {
          send.sup.set_val(dim, lower[dim] + ghost_layer - 1);
          assert(send.get_length(dim) == ghost_layer);
        }
      }
      return send;
    }
}
