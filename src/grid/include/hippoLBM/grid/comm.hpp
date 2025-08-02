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

#include <hippoLBM/grid/box3d.hpp>
#include <onika/cuda/stl_adaptors.h>

namespace hippoLBM
{
  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

  /**
   * @brief A communication container for sending and receiving data between processes.
   *
   * @tparam Components The number of data elements per point.
   */
  template<int Components>
    struct LBMComm
    {
      int m_dest; ///< The destination process ID.
      int m_tag; ///< The MPI communication tag.
      Box3D m_box;
      vector_t<double> m_data; ///< The communication buffer.

      // used for debuging
      void debug_print_comm()
      {
        onika::lout << "Dest: " << m_dest << " Tag: " << m_tag << " Data Size: " << m_data.size() << std::endl;
        onika::lout << "Box: " << std::endl; 
        m_box.print();
      }

      /**
       * @brief Constructor for the comm struct.
       *
       * @param dest The destination process ID.
       * @param tag The MPI communication tag.
       * @param b The communication box.
       */
      LBMComm(const int dest, const int tag, const Box3D& b) : m_dest(dest), m_tag(tag), m_box(b), m_data()
      {
        int size = b.number_of_points();
        allocate(size);
      }

      // default
      LBMComm() {}

      /**
       * @brief Get the size of the data buffer.
       *
       * @return The size of the data buffer.
       */
      int get_size() { return onika::cuda::vector_size(m_data); }

      /**
       * @brief Get the destination process ID.
       *
       * @return The destination process ID.
       */
      int get_dest() { return m_dest; }

      /**
       * @brief Get the MPI communication tag.
       *
       * @return The MPI communication tag.
       */
      int get_tag() { return m_tag; }

      /**
       * @brief Get the communication box.
       *
       * @return Reference to the communication box.
       */
      Box3D& get_box() { return m_box; }

      /**
       * @brief Get a pointer to the data buffer.
       *
       * @return Pointer to the data buffer.
       */
      double* get_data() { return onika::cuda::vector_data(m_data); }

      /**
       * @brief Allocate memory for the data buffer.
       *
       * @param size The size of the data buffer.
       */
      void allocate(int size)
      {
        m_data.resize(size * Components);
      }
    };

  /**
   * @brief A container for ghost cell communication consisting of send and receive communications.
   *
   * @tparam Components The number of data elements per point.
   * @tparam DIM The dimension of the communication box.
   */
  template<int Components>
    struct LBMGhostComm
    {
      LBMComm<Components> send; ///< The send communication.
      LBMComm<Components> recv; ///< The receive communication.

      LBMGhostComm() {}
      /**
       * @brief Constructor for the LBMGhostComm struct.
       *
       * @param s The send communication.
       * @param r The receive communication.
       */
      LBMGhostComm(LBMComm<Components>& s, LBMComm<Components>& r) : send(s), recv(r) {}

      // used for debuging
      void debug_print_comm()
      {
        onika::lout << " Ghost Comm[Send]" << std::endl;
        send.debug_print_comm();
        onika::lout << " Ghost Comm[Recv]" << std::endl;
        recv.debug_print_comm();
      }

    };
}
