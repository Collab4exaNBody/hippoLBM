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

#include <hippoLBM/core/box3d.hpp>

namespace hippoLBM {
template <typename T>
using vector_t = onika::memory::CudaMMVector<T>;

/**
 * @brief A communication container for sending and receiving data between processes.
 * @tparam Components The number of data elements per point.
 */
template <int Components>
struct LBMComm {
  int dest_;  ///< The destination process ID.
  int tag_;   ///< The MPI communication tag.
  Box3D box_;
  vector_t<double> data_;  ///< The communication buffer.

  // used for debuging
  void debug_print_comm() {
    onika::lout << "Dest: " << dest_ << " Tag: " << tag_ << " Data Size: " << data_.size() << std::endl;
    onika::lout << "Box: " << std::endl;
    box_.print();
  }

  /**
   * @brief Constructor for the comm struct.
   *
   * @param dest The destination process ID.
   * @param tag The MPI communication tag.
   * @param b The communication box.
   */
  LBMComm(const int dest, const int tag, const Box3D& b) : dest_(dest), tag_(tag), box_(b), data_() {
    int size = b.number_of_points();
    allocate(size);
  }

  // default
  LBMComm() {}

  /**
   * @brief Get the size of the data buffer.
   * @return The size of the data buffer.
   */
  int get_size() { return onika::cuda::vector_size(data_); }

  /**
   * @brief Get the destination process ID.
   * @return The destination process ID.
   */
  int get_dest() { return dest_; }

  /**
   * @brief Get the MPI communication tag.
   * @return The MPI communication tag.
   */
  int get_tag() { return tag_; }

  /**
   * @brief Get the communication box.
   * @return Reference to the communication box.
   */
  Box3D& get_box() { return box_; }

  /**
   * @brief Get a pointer to the data buffer.
   * @return Pointer to the data buffer.
   */
  double* get_data() { return onika::cuda::vector_data(data_); }

  /**
   * @brief Allocate memory for the data buffer.
   * @param size The size of the data buffer.
   */
  void allocate(int size) { data_.resize(size * Components); }
};

/**
 * @brief A container for ghost cell communication consisting of send and receive communications.
 * @tparam Components The number of data elements per point.
 * @tparam DIM The dimension of the communication box.
 */
template <int Components>
struct LBMGhostComm {
  LBMComm<Components> send_;  ///< The send communication.
  LBMComm<Components> recv_;  ///< The receive communication.

  LBMGhostComm() {}
  /**
   * @brief Constructor for the LBMGhostComm struct.
   * @param s The send communication.
   * @param r The receive communication.
   */
  LBMGhostComm(LBMComm<Components>& s, LBMComm<Components>& r) : send_(s), recv_(r) {}

  // used for debuging
  void debug_print_comm() {
    onika::lout << " Ghost Comm[Send]" << std::endl;
    send_.debug_print_comm();
    onika::lout << " Ghost Comm[Recv]" << std::endl;
    recv_.debug_print_comm();
  }
};
}  // namespace hippoLBM
