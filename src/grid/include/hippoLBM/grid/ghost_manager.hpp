#pragma once
#include <mpi.h>
#include <onika/cuda/cuda_context.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <hippoLBM/core/box3d.hpp>
#include <hippoLBM/core/point3d.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/operator_ghost_manager.hpp>
#include <hippoLBM/grid/packers.hpp>

namespace hippoLBM {

/**
 * @brief A manager for ghost cell communication between processes.
 * @tparam N The number of data elements per point.
 * @tparam DIM The dimension of the communication box.
 */
template <int Components>
struct LBMGhostManager {
  using ParExecSpace3d = onika::parallel::ParallelExecutionSpace<3>;
  std::vector<LBMGhostComm<Components>> data_;  ///< Vector of ghost communications.
  std::vector<MPI_Request> request_;            ///< Vector of MPI requests.
  int count_request_ = 0;                       ///< count number of request

  void debug_print_comm() {
    onika::lout << "Debug Print Comms, number of comms" << data_.size() << " Components: " << Components << std::endl;
    for (auto it : data_) it.debug_print_comm();
  }

  /**
   * @brief Get the number of ghost communications.
   * @return The number of ghost communications.
   */
  uint64_t get_size() { return data_.size(); }

  /**
   * @brief Add a send and receive communication pair to the manager.
   * @param s The send communication.
   * @param r The receive communication.
   */
  void add_comm(LBMComm<Components>& s, LBMComm<Components>& r) { data_.push_back(LBMGhostComm(s, r)); }

  /** @brief Reset the ghost manager. */
  void reset() {
    data_.resize(0);
    resize_request();
  }

  /**
   * @brief Resize the MPI request vector based on the number of ghost communications.
   */
  void resize_request() {
    const uint64_t nb_request = this->get_size() * 2;
    request_.resize(nb_request);
  }

  /**
   * @brief Wait for all MPI requests to complete.
   */
  void wait_all() {
    MPI_Waitall(count_request_ /* request_.size() */, request_.data(), MPI_STATUSES_IGNORE);
    count_request_ = 0;
  }

  /**
   * @brief Initiate non-blocking receives for ghost cell data.
   */
  void do_recv() {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    count_request_ = 0;
#ifdef PRINT_DEBUG_MPI
    std::cout << "Number of messages " << this->data_.size() << std::endl;
#endif
    for (auto& it : this->data_) {
      auto& recv = it.recv_;
      uint64_t nb_bytes = recv.get_size() * sizeof(double);
#ifdef PRINT_DEBUG_MPI
      std::cout << "I recv " << nb_bytes << " bytes from " << recv.get_dest() << " with tag " << recv.get_tag()
                << std::endl;
#endif
      // bool do_recv = !((send.get_tag() == recv.get_tag()) && (send.get_dest() == recv.get_dest()));
      bool do_recv = !(recv.get_dest() == mpi_rank);
      if (do_recv)  // NOT (periodic case && himself)
      {
        MPI_Irecv(recv.get_data(), nb_bytes, MPI_CHAR, recv.get_dest(), recv.get_tag(), MPI_COMM_WORLD,
                  &(this->request_[count_request_++]));
      }
    }
  }

  /**
   * @brief Unpack received ghost cell data into the mesh.
   * @param mesh Pointer to the mesh data.
   * @param mesh_box The box representing the mesh.
   */
  template <typename ParExecCtxFunc>
  void do_unpack(FieldView<Components>& mesh, Box3D& mesh_box, ParExecCtxFunc& par_exec_ctx) {
    for (auto& it : this->data_) {
      auto& recv = it.recv_;
      // Wrap data
      FieldView<Components> wrecv = {recv.get_data(), uint64_t(recv.get_size() / Components)};
      // Define kernel
      unpacker<Components> unpack = {mesh, wrecv, recv.get_box(), mesh_box};
      // Define cuda/omp grid
      ParExecSpace3d parallel_range = set(recv.get_box());
      // Run kernel
      parallel_for(parallel_range, unpack, par_exec_ctx("unpack"));
    }
    ONIKA_CU_DEVICE_SYNCHRONIZE();
  }

  /**
   * @brief Pack and send ghost cell data from the mesh.
   * @param mesh Pointer to the mesh data.
   * @param mesh_box The box representing the mesh.
   */
  template <typename ParExecCtxFunc>
  void do_pack_send(FieldView<Components>& mesh, Box3D& mesh_box, ParExecCtxFunc& par_exec_ctx) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    for (auto& it : this->data_) {
      auto& send = it.send_;
      // Wrap data
      FieldView<Components> wsend = {send.get_data(), uint64_t(send.get_size() / Components)};
      // Define kernel
      packer<Components> pack = {wsend, mesh, send.get_box(), mesh_box};
      // Define cuda/omp grid
      ParExecSpace3d parallel_range = set(send.get_box());
      // Run kernel
      parallel_for(parallel_range, pack, par_exec_ctx("pack"));
    }
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    for (auto& it : this->data_) {
      auto& send = it.send_;
      auto& recv = it.recv_;
      uint64_t nb_bytes = send.get_size() * sizeof(double);
      // if((send.get_tag() == recv.get_tag()) && (send.get_dest() == recv.get_dest())) // periodic case && himself
      if (mpi_rank == recv.get_dest())  // periodic case && himself
      {
        ONIKA_CU_MEMCPY(recv.get_data(), send.get_data(), nb_bytes);  // cudaMemcpyDefault, 0 /** default stream */);
      } else {
        MPI_Isend(send.get_data(), nb_bytes, MPI_CHAR, send.get_dest(), send.get_tag(), MPI_COMM_WORLD,
                  &(this->request_[count_request_++]));
      }
    }
  }
};

/** @brief Write communication data to files for debugging purposes.
 * @param ghost_manager The ghost manager containing the communication data to be written.
 * @tparam Components The number of data elements per point in the ghost communications.
 */
template <int Components>
void write_comm(LBMGhostManager<Components>& ghost_manager) {
  auto& comms = ghost_manager.data_;
  const size_t number_of_comms = comms.size();
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  std::string basename = "HippoLBMDebugCommManager";
  if (mpi_rank == 0) {
    std::filesystem::create_directories(basename);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::stringstream ss;

  ss << "Rank " << std::to_string(mpi_rank) << std::endl;

  for (size_t comm_id = 0; comm_id < number_of_comms; comm_id++) {
    auto& [send, recv] = comms[comm_id];
    ss << "[Send] to:   " << send.get_dest() << " tag: " << send.get_tag() << " size: " << send.get_size() << std::endl;
    ss << "[Recv] from: " << recv.get_dest() << " tag: " << recv.get_tag() << " size: " << recv.get_size() << std::endl;
  }

  std::string filename = basename + "/proc_" + std::to_string(mpi_rank) + ".txt";
  std::ofstream file(filename);
  file << ss.rdbuf();
}
}  // namespace hippoLBM
