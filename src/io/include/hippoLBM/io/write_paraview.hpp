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

#include <onika/string_utils.h>

#include <filesystem>
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/io/writer.hpp>

namespace hippoLBM {
/** Currently, it works for double */
struct ExternalParaviewField {
  std::string field_name_;    // The name of the field to be written.
  double* const input_data_;  // Pointer to the input data array that contains the values to be written.
  int number_of_components_;  // The number of components in the data (e.g., 1 for scalar fields, 3 for vector fields).
  uint64_t number_of_elements_;  // The total number of elements in the data array (e.g., the number of grid points).
};

/** @brief Structure to hold external Paraview fields */
struct ExternalParaviewFields {
  std::vector<ExternalParaviewField> fields_;  // A vector to store multiple external Paraview fields.

  /** @brief Register a new field to be written
   * @param field_name The name of the field to be written.
   * @param data Pointer to the input data array that contains the values to be written.
   * @param components The number of components in the data (e.g., 1 for scalar fields, 3 for vector fields).
   * @param elements The total number of elements in the data array (e.g., the number of grid points).
   */
  void register_field(std::string field_name, double* const data, int components, uint64_t elements) {
    fields_.push_back(ExternalParaviewField{field_name, data, components, elements});
  }

  /** @brief Write the Paraview XML header for the external fields
   * @param outFile The output file stream.
   */
  inline void write_pvtr(std::ofstream& outFile) const {
    for (auto& field : fields_) {
      outFile << "       <PDataArray"
              << " Name=\"" << field.field_name_ << "\""
              << " type=\"Float32\""
              << " NumberOfComponents=\"" << field.number_of_components_ << "\"/>" << std::endl;
    }
  }

  /** @brief Write the Paraview XML data for the external fields
   * @param grid The LBM grid containing the simulation domain information.
   * @param outFile The output file stream.
   */
  inline void write_vtr(const LBMGrid& grid, std::ofstream& outFile) const {
    for (auto& field : fields_) {
      WriterExternalData writer_external_data = {field.number_of_components_, field.number_of_elements_};
      outFile << "          <DataArray type=\"Float32\""
              << " Name=\"" << field.field_name_ << "\""
              << " format=\"ascii\""
              << " NumberOfComponents=\"" << field.number_of_components_ << "\">" << std::endl;
      std::stringstream paraview_stream_buffer;
      for_all<Area::Local, Traversal::All>(grid, writer_external_data, paraview_stream_buffer, field.input_data_);
      outFile << paraview_stream_buffer.rdbuf();
      outFile << std::endl;
      outFile << "          </DataArray>" << std::endl;
    }
  }
};

struct ExternalParaviewFieldsNullOp {
  // inline void write_pvtr(std::ofstream& outFile) const {}
  // template <Area A, Traversal Tr> inline void write_vtr(const LBMGrid& grid, std::ofstream& outFile) const {}
};

struct ParaviewBuffers {
  /** Buffers */
  onika::memory::CudaMMVector<float> u_;   // velocity, scaled to float for paraview
  onika::memory::CudaMMVector<float> p_;   // pressure
  onika::memory::CudaMMVector<int> obst_;  // obstacle flag

  /** streams */
  std::stringstream i_;  // x-coordinates of the grid points
  std::stringstream j_;  // y-coordinates of the grid points
  std::stringstream k_;  // z-coordinates of the grid points

  /** @brief Resize the buffers
   * @param size The new size of the buffers
   */
  void resize(const int size) {
    u_.resize(3 * size);  // Vec3d
    p_.resize(size);
    obst_.resize(size);
  }

  /** @brief Convert simulation data to stream
   * @param Box The bounding box for the simulation domain
   * @param dx The grid spacing
   */
  void sim_data_to_stream(Box3D& Box, double dx) {
    // todo
  }

  /** @brief Convert simulation header data to stream
   * @param Box The bounding box for the simulation domain
   * @param dx The grid spacing
   */
  void sim_header_to_stream(Box3D& Box, double dx) {
    for (int x = Box.start(0); x <= Box.end(0); x++) i_ << (double)(x * dx) << " ";
    for (int y = Box.start(1); y <= Box.end(1); y++) j_ << (double)(y * dx) << " ";
    for (int z = Box.start(2); z <= Box.end(2); z++) k_ << (double)(z * dx) << " ";
  }
};

/** @brief Write the Paraview XML header for the external fields
 * @param basedir The base directory for the output files
 * @param basename The base name for the output files
 * @param number_of_files The number of files to write
 * @param domain The LBM domain containing the simulation data
 * @param external_paraview_fields The external Paraview fields to write
 */
template <typename LBMDomain, typename EPF>
inline void write_pvtr(std::string basedir, std::string basename, size_t number_of_files, LBMDomain& domain,
                       const EPF& external_paraview_fields = ExternalParaviewFieldsNullOp{}) {
  const LBMGrid& grid = domain.m_grid_;
  auto [lx, ly, lz] = domain.domain_size_;
  // I could be smarter here
  int box_size = sizeof(Box3D);
  auto global = grid.build_box<Area::Global, Traversal::All>();
  std::vector<Box3D> recv;
  recv.resize(number_of_files);
  MPI_Gather(&global, box_size, MPI_CHAR, recv.data(), box_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::string name = basedir + "/" + basename + ".pvtr";
    std::ofstream outFile(name);
    if (!outFile) {
      std::cerr << "Erreur : impossible de créer le fichier de sortie suivant: " << name << std::endl;
      return;
    }

    outFile << " <VTKFile type=\"PRectilinearGrid\"> " << std::endl;
    outFile << "   <PRectilinearGrid WholeExtent=\"0 " << lx - 1 << " 0 " << ly - 1 << " 0 " << lz - 1 << "\""
            << std::endl;
    ;
    outFile << "                     GhostLevel=\"#\">" << std::endl;
    // outFile << " GhostLevel=\"#\">" << std::endl;
    //   outFile << "      <Piece Extent=\"0 " << lx << " 0 " << ly << " 0 " << lz<< "\"" << std::endl;
    for (size_t i = 0; i < number_of_files; i++) {
      std::string subfile = basename + "/%06d.vtr";
      subfile = onika::format_string(subfile, i);
      outFile << "     <Piece Extent=\" " << recv[i].start(0) << " " << recv[i].end(0) << " " << recv[i].start(1) << " "
              << recv[i].end(1) << " " << recv[i].start(2) << " " << recv[i].end(2) << "\" Source=\"" << subfile
              << "\"/>" << std::endl;
    }
    outFile << "    <PCoordinates>" << std::endl;
    outFile << "      <PDataArray type=\"Float32\" Name=\"X\"/>" << std::endl;
    outFile << "      <PDataArray type=\"Float32\" Name=\"Y\"/>" << std::endl;
    outFile << "      <PDataArray type=\"Float32\" Name=\"Z\"/>" << std::endl;
    outFile << "    </PCoordinates>" << std::endl;
    outFile << "     <PPointData Scalars=\"P OBST\"  Vectors=\"U\" >" << std::endl;
    outFile << "       <PDataArray Name=\"P\" type=\"Float32\" NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "       <PDataArray Name=\"OBST\" type=\"Float32\" NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "       <PDataArray Name=\"U\" type=\"Float32\" NumberOfComponents=\"3\"/>" << std::endl;
    // define your fields in external_paraview_fields
    external_paraview_fields.write_pvtr(outFile);
    outFile << "     </PPointData> " << std::endl;
    outFile << "   </PRectilinearGrid>" << std::endl;
    outFile << " </VTKFile>" << std::endl;
  }
}

/** @brief Write the Paraview XML file for a single process
 * @param name The name of the output file
 * @param domain The LBM domain containing the simulation data
 * @param data The simulation data to write
 * @param traversals The grid traversals for the simulation
 * @param params The LBM parameters
 * @param external_paraview_fields The external Paraview fields to write
 */
template <typename LBMDomain, typename LBMFieds, typename EPF>
inline void write_vtr(std::string name, const LBMDomain& domain, LBMFieds& data, const LBMGridRegion& traversals,
                      const LBMParameters& params,
                      const EPF& external_paraview_fields = ExternalParaviewFieldsNullOp{}) {
  const LBMGrid& grid = domain.m_grid_;
  auto [lx, ly, lz] = domain.domain_size_;
  const double dx = grid.dx_;
  name = name + ".vtr";
  std::ofstream outFile(name);
  if (!outFile) {
    std::cerr << "Erreur : impossible de créer le fichier de sortie suivant: " << name << std::endl;
    return;
  }
  // only real point
  constexpr Area L = Area::Local;
  constexpr Area G = Area::Global;
  auto local = grid.build_box<L, Traversal::All>();
  auto global = grid.build_box<G, Traversal::All>();

  auto [traversal_ptr, traversal_size] = traversals.get_data<Traversal::All>();

  const int* const obst = data.obstacles();

  NullFuncWriter nullop;
  write_file writer_obst = {nullop};

  double ratio_dx_dtLB = dx / params.dtLB_;
  UWriter u = {obst, ratio_dx_dtLB};
  WriteVec3d writer_vec3d = {u, local};
  // WriteVec3d writer_vec3d = {nullop, local};

  double c_c_avg_rho_div_three = 1. / 3. * params.celerity_ * params.celerity_ * params.avg_rho_;
  PressionWriter pression = {obst, c_c_avg_rho_div_three};
  write_file writer_double = {pression};
  // write_file writer_double = {nullop};

  assert(local.get_length(0) == global.get_length(0));
  assert(local.get_length(1) == global.get_length(1));
  assert(local.get_length(2) == global.get_length(2));

  ParaviewBuffers paraview_streams;
  paraview_streams.sim_header_to_stream(global, dx);

  outFile << "<VTKFile type=\"RectilinearGrid\">" << std::endl;
  outFile << " <RectilinearGrid WholeExtent=\" 0 " << lx - 1 << " 0 " << ly - 1 << " 0 " << lz - 1 << "\">"
          << std::endl;
  outFile << "      <Piece Extent=\"" << global.start(0) << " " << global.end(0) << " " << global.start(1) << " "
          << global.end(1) << " " << global.start(2) << " " << global.end(2) << " \">" << std::endl;
  outFile << "      <Coordinates>" << std::endl;
  outFile << "          <DataArray type=\"Float32\" Name=\"X\" format=\"ascii\">" << std::endl;
  outFile << paraview_streams.i_.rdbuf();
  outFile << std::endl;
  outFile << "          </DataArray>" << std::endl;
  outFile << "          <DataArray type=\"Float32\" Name=\"Y\" format=\"ascii\">" << std::endl;
  outFile << paraview_streams.j_.rdbuf();
  outFile << std::endl;
  outFile << "          </DataArray>" << std::endl;
  outFile << "          <DataArray type=\"Float32\" Name=\"Z\" format=\"ascii\">" << std::endl;
  outFile << paraview_streams.k_.rdbuf();
  outFile << std::endl;
  outFile << "          </DataArray>" << std::endl;
  outFile << "      </Coordinates>" << std::endl;
  outFile << "      <PointData>" << std::endl;
  outFile << "          <DataArray type=\"Float32\" Name=\"P\" format=\"ascii\">" << std::endl;
  {
    std::stringstream paraview_stream_buffer;
    for_all(traversal_ptr, traversal_size, writer_double, paraview_stream_buffer, onika::cuda::vector_data(data.m0_));
    outFile << paraview_stream_buffer.rdbuf();
  }
  outFile << std::endl;
  outFile << "          </DataArray>" << std::endl;
  outFile << "          <DataArray type=\"Float32\" Name=\"U\" format=\"ascii\" NumberOfComponents=\"3\">" << std::endl;
  {
    std::stringstream paraview_stream_buffer;
    for_all<L, Traversal::All>(grid, writer_vec3d, paraview_stream_buffer, data.flux());
    outFile << paraview_stream_buffer.rdbuf();
  }
  outFile << std::endl;
  outFile << "          </DataArray>" << std::endl;
  outFile << "          <DataArray type=\"Float32\" Name=\"OBST\" format=\"ascii\">" << std::endl;
  {
    std::stringstream paraview_stream_buffer;
    for_all(traversal_ptr, traversal_size, writer_obst, paraview_stream_buffer, onika::cuda::vector_data(data.obst_));
    outFile << paraview_stream_buffer.rdbuf();
  }
  outFile << std::endl;
  outFile << "          </DataArray>" << std::endl;

  // define your fields in external_paraview_fields
  external_paraview_fields.write_vtr(grid, outFile);
  // external_paraview_fields.write_vtr<L, Traversal::All>( grid, outFile);

  // end file
  outFile << "      </PointData>" << std::endl;
  outFile << "      </Piece>" << std::endl;
  outFile << " </RectilinearGrid>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

template <int Q>
void write_paraview(MPI_Comm comm, std::string filename, std::string basedir, long timestep, LBMFields<Q>& fields,
                    const LBMParameters& parameters, const LBMGridRegion& traversals, const LBMDomain<Q>& domain,
                    const ExternalParaviewFields& external_paraview_fields, bool display_filename = false) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::string file_name = filename;
  file_name = onika::format_string(file_name, timestep);
  std::string fullname = basedir + file_name;

  if (rank == 0) {
    std::filesystem::create_directories(fullname);
  }

  if (display_filename) {
    lout << "writing paraview file: " << fullname << std::endl;
  }

  fullname += "/%06d";
  fullname = onika::format_string(fullname, rank);

  MPI_Barrier(comm);
  write_pvtr(basedir, file_name, size, domain, external_paraview_fields);
  write_vtr(fullname, domain, fields, traversals, parameters, external_paraview_fields);
}
}  // namespace hippoLBM
