//
//  GhostManagercpp
//  
//
//  Created by prat on 19/09/2023.
//  Copyright © 2019 amarsid. All rights reserved.
//

#pragma once
#include <cstring>
#include <grid_lbm/point.hpp>
#include <grid_lbm/box.hpp>
#include <grid_lbm/comm.hpp>
#include <grid_lbm/operator_ghost_manager.hpp>
#include <grid_lbm/packers.hpp>

namespace hippoLBM
{

  /**
   * @brief A manager for ghost cell communication between processes.
   *
   * @tparam N The number of data elements per point.
   * @tparam DIM The dimension of the communication box.
   */
  template<int N, int DIM>
    struct ghost_manager
    {
      std::vector<ghost_comm<N, DIM>> m_data; ///< Vector of ghost communications.
      std::vector<MPI_Request> m_request; ///< Vector of MPI requests.


      void debug_print_comm()
      {
	onika::lout << "Debug Print Comms, number of comms" << m_data.size() << " N: " << N << " DIM: " << DIM << std::endl;
	for(auto it: m_data) it.debug_print_comm();
      }

      /**
       * @brief Get the number of ghost communications.
       *
       * @return The number of ghost communications.
       */
      int get_size() { return m_data.size(); }

      /**
       * @brief Add a send and receive communication pair to the manager.
       *
       * @param s The send communication.
       * @param r The receive communication.
       */
      void add_comm(comm<N, DIM>& s, comm<N, DIM>& r)
      {
	m_data.push_back(ghost_comm(s, r));
      }

      void reset()
      {
	m_data.resize(0);
	resize_request();
      }

      /**
       * @brief Resize the MPI request vector based on the number of ghost communications.
       */
      void resize_request()
      {
	const int nb_request = this->get_size() * 2;
	m_request.resize(nb_request);
      }

      /**
       * @brief Wait for all MPI requests to complete.
       */
      void wait_all()
      {
	MPI_Waitall(m_request.size(), m_request.data(), MPI_STATUSES_IGNORE);
      }

      /**
       * @brief Initiate non-blocking receives for ghost cell data.
       */
      void do_recv()
      {
	int acc = 0;
#ifdef PRINT_DEBUG_MPI
	std::cout << "Number of messages " << this->m_data.size() << std::endl;
#endif
	for (auto& it : this->m_data)
	{
	  auto& recv = it.recv;
	  int nb_bytes = recv.get_size() * sizeof(double);
#ifdef PRINT_DEBUG_MPI
	  std::cout << "I recv " << nb_bytes << " bytes from " << recv.get_dest() << " with tag " << recv.get_tag() << std::endl;
#endif
	  MPI_Irecv(recv.get_data(), nb_bytes, MPI_CHAR, recv.get_dest(), recv.get_tag(), MPI_COMM_WORLD, &(this->m_request[acc++]));
	}
      }

      /**
       * @brief Unpack received ghost cell data into the mesh.
       *
       * @param mesh Pointer to the mesh data.
       * @param mesh_box The box representing the mesh.
       */
      void do_unpack(double* mesh, box<DIM>& mesh_box)
      {
	unpacker<N, DIM> unpack;
	for (auto& it : this->m_data)
	{
	  auto& recv = it.recv;
	  unpack(mesh, recv.get_data(), recv.get_box(), mesh_box);
	}
      }
      void do_unpack(WrapperF<N>& mesh, box<DIM>& mesh_box)
      {
	unpacker<N, DIM> unpack;
	for (auto& it : this->m_data)
	{
	  auto& recv = it.recv;
          WrapperF<N> wrecv = {recv.get_data() , recv.get_size() / N};
#ifdef ONIKA_CUDA_VERSION
          cuda_parallel_for_box(recv.get_box(), unpack, mesh, wrecv, recv.get_box(), mesh_box);
#else
	  unpack(mesh, wrecv, recv.get_box(), mesh_box);
#endif
	}
        ONIKA_CU_DEVICE_SYNCHRONIZE();
      }

      /**
       * @brief Pack and send ghost cell data from the mesh.
       *
       * @param mesh Pointer to the mesh data.
       * @param mesh_box The box representing the mesh.
       */
      void do_pack_send(double* mesh, box<DIM>& mesh_box)
      {
	packer<N, DIM> pack;
	const int size = this->get_size();
	int acc = size;
	for (auto& it : this->m_data)
	{
	  auto& send = it.send;
	  pack(send.get_data(), mesh, send.get_box(), mesh_box);
	  int nb_bytes = send.get_size() * sizeof(double);
#ifdef PRINT_DEBUG_MPI
	  std::cout << "I send " << nb_bytes << " bytes to " << send.get_dest() << " with tag " << send.get_tag() << std::endl;
#endif
	  MPI_Isend(send.get_data(), nb_bytes, MPI_CHAR, send.get_dest(), send.get_tag(), MPI_COMM_WORLD, &(this->m_request[acc++]));
	}
      }

      void do_pack_send(WrapperF<N>& mesh, box<DIM>& mesh_box)
      {
	packer<N, DIM> pack;
	const int size = this->get_size();
	int acc = size;
	for (auto& it : this->m_data)
	{
	  auto& send = it.send;
          WrapperF<N> wsend = {send.get_data() , send.get_size() / N};
#ifdef ONIKA_CUDA_VERSION
          cuda_parallel_for_box(send.get_box(), pack, wsend, mesh, send.get_box(), mesh_box);
#else
	  pack(wsend, mesh, send.get_box(), mesh_box);
#endif
        }
        ONIKA_CU_DEVICE_SYNCHRONIZE();
        for (auto& it : this->m_data)
        {
	  auto& send = it.send;
	  int nb_bytes = send.get_size() * sizeof(double);
	  MPI_Isend(send.get_data(), nb_bytes, MPI_CHAR, send.get_dest(), send.get_tag(), MPI_COMM_WORLD, &(this->m_request[acc++]));
	}
      }
    };
}
