#pragma once

#include <grid_lbm/box.hpp>

namespace hippoLBM
{
  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

  /**
   * @brief A communication container for sending and receiving data between processes.
   *
   * @tparam N The number of data elements per point.
   * @tparam DIM The dimension of the communication box.
   */
  template<int N, int DIM>
    struct comm
    {
      int m_dest; ///< The destination process ID.
      int m_tag; ///< The MPI communication tag.
      box<DIM> m_box;
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
      comm(const int dest, const int tag, const box<DIM>& b) : m_dest(dest), m_tag(tag), m_box(b), m_data()
      {
	int size = b.number_of_points();
	allocate(size);
      }

      // default
      comm() {}

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
      box<DIM>& get_box() { return m_box; }

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
	m_data.resize(size * N);
      }
    };

  /**
   * @brief A container for ghost cell communication consisting of send and receive communications.
   *
   * @tparam N The number of data elements per point.
   * @tparam DIM The dimension of the communication box.
   */
  template<int N, int DIM>
    struct ghost_comm
    {
      comm<N, DIM> send; ///< The send communication.
      comm<N, DIM> recv; ///< The receive communication.

      ghost_comm() {}
      /**
       * @brief Constructor for the ghost_comm struct.
       *
       * @param s The send communication.
       * @param r The receive communication.
       */
      ghost_comm(comm<N, DIM>& s, comm<N, DIM>& r) : send(s), recv(r) {}

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
