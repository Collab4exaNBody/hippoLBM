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

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/io/dump_compression.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace hippoLBM {

/** @brief On-disk element type of a dumped field. */
enum class DumpFieldType : int32_t { INT32 = 0, FLOAT64 = 1 };

/** @brief Describes one stored grid field: its name, its number of components per
 * grid point, and the type of each component. */
struct DumpFieldDescriptor {
  char name_[16] = {};
  int32_t components_ = 0;
  int32_t datatype_ = 0;  ///< DumpFieldType
};

struct DumpHeader {
  // comment: used to sanity-check the file on read
  static constexpr int32_t MAGIC = 0x4d424c48;  ///< TODO I'm not sure of what it did , I keep it ...
  static constexpr int32_t VERSION = 100;       ///< represent the current version of the dump file format, 1.0.0 = 100
  static constexpr int32_t MAX_FIELDS = 8;      ///< Warning: arbitrary limit, must be kept

  int32_t magic_ = MAGIC;
  int32_t version_ = VERSION;
  int32_t Q_ = 0;           ///< number of LBM discrete directions (e.g. 19 for D3Q19)
  int32_t num_fields_ = 0;  ///< number of valid entries in fields_
  int32_t num_ranks_ = 0;   ///< number of entries in the index table (ranks at write time, not at read time)
  int32_t global_size_[3] = {0, 0, 0};  ///< global grid size (nodes), not the local subdomain
  double dx_ = 0.0;                     ///< grid spacing
  int64_t timestep_ = 0;
  LBMParameters params_;
  DumpFieldDescriptor fields_[MAX_FIELDS];
};

struct DumpRankIndex {
  int32_t global_inf_[3] = {0, 0, 0};  ///< inclusive lower corner of this rank's real subdomain, global coordinates
  int32_t global_sup_[3] = {0, 0, 0};  ///< inclusive upper corner of this rank's real subdomain, global coordinates
  uint64_t offset_[DumpHeader::MAX_FIELDS] = {};             ///< byte offset of each field's block
  uint64_t compressed_size_[DumpHeader::MAX_FIELDS] = {};    ///< size on disk, in bytes (compressed or raw)
  uint64_t uncompressed_size_[DumpHeader::MAX_FIELDS] = {};  ///< raw size, in bytes
  uint8_t is_compressed_[DumpHeader::MAX_FIELDS] = {};       ///< 1 if the block on disk is zlib-compressed, 0 if raw
};

struct DumpFieldSource {
  std::string name_;            // name of the field, must match one of the DumpHeader::fields_ entries
  const void* data_ = nullptr;  // pointer to the field data for this rank's real subdomain (ghost-free)
  int32_t components_ = 1;  // number of components per grid point (e.g. 1 for density, 3 for flux, Q for distributions)
  DumpFieldType datatype_ = DumpFieldType::FLOAT64;  // type of each component (int32 or float64)
};

// For USERs: use this function to get the list of fields to dump from an LBMFields object
// Example usage: For ExaCoLD, add interface fields to the list of fields to dump.
template <int Q>
inline std::vector<DumpFieldSource> hippolbm_dump_fields(LBMFields<Q>& fields) {
  return {
      DumpFieldSource{"obstacle", fields.obstacles(), 1, DumpFieldType::INT32},
      DumpFieldSource{"f", fields.distributions().data_, Q, DumpFieldType::FLOAT64},
  };
}

/** @brief Creates the checkpoint file and writes its DumpHeader (rank 0 only).
 * @param comm MPI communicator; every rank must call this function.
 * @param filename Path to the checkpoint file to create.
 * @param domain The LBM domain, used for its global grid size and dx.
 * @param params The current LBM parameters.
 * @param timestep The current simulation timestep.
 * @param sources The fields that will be dumped for this file (name, components, type).
 * @return The DumpHeader that was written, to be passed on to write_dump_fields().
 */
template <int Q>
inline DumpHeader write_dump_header(MPI_Comm comm, const std::string& filename, const LBMDomain<Q>& domain,
                                    const LBMParameters& params, long timestep,
                                    const std::vector<DumpFieldSource>& sources) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  DumpHeader header;
  header.Q_ = Q;
  header.num_ranks_ = size;
  header.timestep_ = timestep;
  header.dx_ = domain.grid().dx_;
  header.global_size_[0] = domain.domain_size_[0];
  header.global_size_[1] = domain.domain_size_[1];
  header.global_size_[2] = domain.domain_size_[2];
  header.params_ = params;

  if (sources.size() > static_cast<size_t>(DumpHeader::MAX_FIELDS)) {
    std::cerr << "[dump_fields] Error: " << sources.size() << " fields requested, but only " << DumpHeader::MAX_FIELDS
              << " are supported (DumpHeader::MAX_FIELDS)." << std::endl;
  }
  header.num_fields_ = std::min<int32_t>(static_cast<int32_t>(sources.size()), DumpHeader::MAX_FIELDS);
  for (int32_t i = 0; i < header.num_fields_; i++) {
    std::snprintf(header.fields_[i].name_, sizeof(header.fields_[i].name_), "%s", sources[i].name_.c_str());
    header.fields_[i].components_ = sources[i].components_;
    header.fields_[i].datatype_ = static_cast<int32_t>(sources[i].datatype_);
  }

  MPI_File file;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  MPI_File_set_size(file, 0);  // truncate: discard any leftover data from a previous, larger dump
  if (rank == 0) {
    MPI_File_write_at(file, 0, &header, sizeof(DumpHeader), MPI_BYTE, MPI_STATUS_IGNORE);
  }
  MPI_File_close(&file);
  return header;
}

/** @brief Reads back the DumpHeader written by write_dump_header().
 * @param comm MPI communicator to open the file with.
 * @param filename Path to the checkpoint file.
 * @return The DumpHeader stored at the beginning of the file.
 */
inline DumpHeader read_dump_header(MPI_Comm comm, const std::string& filename) {
  DumpHeader header;
  MPI_File file;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  MPI_File_read_at(file, 0, &header, sizeof(DumpHeader), MPI_BYTE, MPI_STATUS_IGNORE);
  MPI_File_close(&file);

  if (header.magic_ != DumpHeader::MAGIC) {
    std::cerr << "[dump_fields] Warning: '" << filename << "' does not look like a hippoLBM checkpoint file."
              << std::endl;
  }
  return header;
}

constexpr uint64_t DUMP_MAX_IO_OPERATION_SIZE = 512ull * 1024ull * 1024ull;  // maximum bytes per I/O operation

inline void mpi_write_at_bytes(MPI_File file, uint64_t offset, const void* data, uint64_t n) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
  uint64_t written = 0;
  while (written < n) {
    const uint64_t chunk = std::min(n - written, DUMP_MAX_IO_OPERATION_SIZE);
    MPI_File_write_at(file, static_cast<MPI_Offset>(offset + written), p + written, static_cast<int>(chunk), MPI_BYTE,
                      MPI_STATUS_IGNORE);
    written += chunk;
  }
}

inline std::vector<uint8_t> pack_field(const DumpFieldSource& source, const Box3D& local_real, const LBMGrid& grid,
                                       uint64_t num_local_points, uint64_t num_points) {
  const int32_t C = source.components_;
  const size_t elem_size = (source.datatype_ == DumpFieldType::INT32) ? sizeof(int32_t) : sizeof(double);
  std::vector<uint8_t> buffer(num_points * C * elem_size);

  // We don't use FieldView here because we want to pack the field in the same way as hippoLBM's own fields, which is
  // not necessarily the same as FieldView's layout.
  auto src_offset = [&](uint64_t idx, int32_t c) -> uint64_t {
#ifdef WFAOS
    return idx * C + c;
#else
    return num_local_points * uint64_t(c) + idx;
#endif
  };

  uint64_t p = 0;
  for (int z = local_real.start(2); z <= local_real.end(2); z++) {
    for (int y = local_real.start(1); y <= local_real.end(1); y++) {
      for (int x = local_real.start(0); x <= local_real.end(0); x++) {
        const uint64_t idx = static_cast<uint64_t>(grid(x, y, z));
        for (int32_t c = 0; c < C; c++) {
          const uint64_t dst = (p * C + c) * elem_size;
          if (source.datatype_ == DumpFieldType::INT32) {
            const int32_t v = reinterpret_cast<const int32_t*>(source.data_)[src_offset(idx, c)];
            std::memcpy(buffer.data() + dst, &v, elem_size);
          } else {
            const double v = reinterpret_cast<const double*>(source.data_)[src_offset(idx, c)];
            std::memcpy(buffer.data() + dst, &v, elem_size);
          }
        }
        p++;
      }
    }
  }
  return buffer;
}

/** @brief Writes fields to a checkpoint file previously created by write_dump_header().
 * @param comm MPI communicator; every rank must call this function.
 * @param filename Path to the checkpoint file (already created by write_dump_header()).
 * @param header The header written by write_dump_header() for this same file.
 * @param domain The LBM domain (used for this rank's real subdomain, in local and global coordinates).
 * @param sources The fields to dump, in the same order as passed to write_dump_header().
 * @param compression_level zlib compression level (1 = fastest, 9 = smallest, default: fastest).
 */
template <int Q>
inline void write_dump_fields(MPI_Comm comm, const std::string& filename, const DumpHeader& header,
                              const LBMDomain<Q>& domain, const std::vector<DumpFieldSource>& sources,
                              int compression_level = DUMP_COMPRESSION_FASTEST) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (static_cast<int32_t>(sources.size()) != header.num_fields_) {
    std::cerr << "[dump_fields] Warning: " << sources.size() << " fields passed to write_dump_fields(), but "
              << header.num_fields_ << " were declared in the header." << std::endl;
  }

  const LBMGrid& grid = domain.grid();
  Box3D local_real = grid.build_box<Area::Local, Traversal::Real>();
  Box3D global_real = grid.build_box<Area::Global, Traversal::Real>();
  const uint64_t num_points = static_cast<uint64_t>(local_real.number_of_points());
  const uint64_t num_local_points =
      static_cast<uint64_t>(grid.build_box<Area::Local, Traversal::All>().number_of_points());

  DumpRankIndex my_index;
  for (int dim = 0; dim < 3; dim++) {
    my_index.global_inf_[dim] = global_real.start(dim);
    my_index.global_sup_[dim] = global_real.end(dim);
  }

  std::vector<std::vector<uint8_t>> compressed(sources.size());
  uint64_t local_total_bytes = 0;
  for (size_t s = 0; s < sources.size(); s++) {
    std::vector<uint8_t> raw = pack_field(sources[s], local_real, grid, num_local_points, num_points);
    my_index.uncompressed_size_[s] = raw.size();
    bool is_compressed = false;
    compressed[s] = zlib_compress(raw.data(), raw.size(), compression_level, is_compressed);
    my_index.is_compressed_[s] = is_compressed ? 1 : 0;
    my_index.compressed_size_[s] = compressed[s].size();
    local_total_bytes += compressed[s].size();
  }

  uint64_t exclusive_prefix = 0;
  MPI_Exscan(&local_total_bytes, &exclusive_prefix, 1, MPI_UINT64_T, MPI_SUM, comm);
  if (rank == 0) exclusive_prefix = 0;  // MPI_Exscan leaves rank 0's result undefined

  const uint64_t index_section_offset = sizeof(DumpHeader);
  const uint64_t index_section_size = static_cast<uint64_t>(size) * sizeof(DumpRankIndex);
  const uint64_t data_section_offset = index_section_offset + index_section_size;

  uint64_t offset = data_section_offset + exclusive_prefix;
  for (size_t s = 0; s < sources.size(); s++) {
    my_index.offset_[s] = offset;
    offset += my_index.compressed_size_[s];
  }

  MPI_File file;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

  MPI_File_write_at(file, static_cast<MPI_Offset>(index_section_offset + rank * sizeof(DumpRankIndex)), &my_index,
                    sizeof(DumpRankIndex), MPI_BYTE, MPI_STATUS_IGNORE);
  for (size_t s = 0; s < sources.size(); s++) {
    mpi_write_at_bytes(file, my_index.offset_[s], compressed[s].data(), compressed[s].size());
  }

  MPI_File_close(&file);
}

}  // namespace hippoLBM
