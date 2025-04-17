#pragma once
#include <mpi.h>
#include <cstring>
#include <onika/cuda/cuda_context.h>
#include <grid/point.hpp>
#include <grid/box.hpp>
#include <grid/comm.hpp>
#include <grid/operator_ghost_manager.hpp>
#include <grid/packers.hpp>

namespace hippoLBM
{

  /**
   * @brief A manager for ghost cell communication between processes.
   *
   * @tparam N The number of data elements per point.
   * @tparam DIM The dimension of the communication box.
   */
  template<int Components, int DIM>
    struct ghost_manager
    {
      using ParExecSpace3d = onika::parallel::ParallelExecutionSpace<3>;
      std::vector<ghost_comm<Components, DIM>> m_data; ///< Vector of ghost communications.
      std::vector<MPI_Request> m_request; ///< Vector of MPI requests.


      void debug_print_comm()
      {
        onika::lout << "Debug Print Comms, number of comms" << m_data.size() << " Components: " << 
 Components << " DIM: " << DIM << std::endl;
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
      void add_comm(comm<Components, DIM>& s, comm<Components, DIM>& r)
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
        //MPI_Waitall(m_request.size(), m_request.data(), MPI_STATUSES_IGNORE);
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
          auto& send = it.recv;
          auto& recv = it.recv;
          int nb_bytes = recv.get_size() * sizeof(double);
#ifdef PRINT_DEBUG_MPI
          std::cout << "I recv " << nb_bytes << " bytes from " << recv.get_dest() << " with tag " << recv.get_tag() << std::endl;
#endif
          bool do_recv = !((send.get_tag() == recv.get_tag()) && (send.get_dest() == recv.get_dest()));
          if(do_recv) // NOT (periodic case && himself)
          {
           MPI_Irecv(recv.get_data(), nb_bytes, MPI_CHAR, recv.get_dest(), recv.get_tag(), MPI_COMM_WORLD, &(this->m_request[acc++]));
          }
        }
      }

      /**
       * @brief Unpack received ghost cell data into the mesh.
       *
       * @param mesh Pointer to the mesh data.
       * @param mesh_box The box representing the mesh.
       */
      template<typename ParExecCtxFunc>
	void do_unpack(
	    FieldView<Components>& mesh, 
	    box<DIM>& mesh_box, 
	    ParExecCtxFunc& par_exec_ctx)
	{
	  for (auto& it : this->m_data)
	  {
	    auto& recv = it.recv;
	    // Wrap data
	    FieldView<Components> wrecv = {recv.get_data() , recv.get_size() / Components};
	    // Define kernel
	    unpacker<Components, DIM> unpack = {mesh, wrecv, recv.get_box(), mesh_box};
	    // Define cuda/omp grid
	    ParExecSpace3d parallel_range = set(recv.get_box());        
	    // Run kernel
	    parallel_for(parallel_range, unpack, par_exec_ctx("unpack"));
	  }
	  ONIKA_CU_DEVICE_SYNCHRONIZE();
	}

      /**
       * @brief Pack and send ghost cell data from the mesh.
       *
       * @param mesh Pointer to the mesh data.
       * @param mesh_box The box representing the mesh.
       */
      template<typename ParExecCtxFunc>
	void do_pack_send(
	    FieldView<Components>& mesh, 
	    box<DIM>& mesh_box,
	    ParExecCtxFunc& par_exec_ctx)
	{
	  const int size = this->get_size();
	  int acc = size;
	  for (auto& it : this->m_data)
	  {
	    auto& send = it.send;
	    // Wrap data
	    FieldView<Components> wsend = {send.get_data() , send.get_size() / Components};
	    // Define kernel
	    packer<Components, DIM> pack = {wsend, mesh, send.get_box(), mesh_box};
	    // Define cuda/omp grid
	    ParExecSpace3d parallel_range = set(send.get_box());
	    // Run kernel
	    parallel_for(parallel_range, pack, par_exec_ctx("pack"));
	  }
	  ONIKA_CU_DEVICE_SYNCHRONIZE();

	  for (auto& it : this->m_data)
	  {
	    auto& send = it.send;
      auto& recv = it.recv; 
	    int nb_bytes = send.get_size() * sizeof(double);
      if((send.get_tag() == recv.get_tag()) && (send.get_dest() == recv.get_dest())) // periodic case && himself
      {
        ONIKA_CU_MEMCPY(recv.get_data(), send.get_data(), nb_bytes); // cudaMemcpyDefault, 0 /** default stream */);
      }
      else
      {
	      MPI_Isend(send.get_data(), nb_bytes, MPI_CHAR, send.get_dest(), send.get_tag(), MPI_COMM_WORLD, &(this->m_request[acc++]));
      }
	  }
	}
    };
}
