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

#include <mpi.h>

#include <cstdlib>
#include <iostream>

// onika
#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/memory/allocator.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/grid/make_variant_operator.hpp>

// Implementation
#include <hippoLBM/io/dump_fields.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

class CheckDumpLBM : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator."});
  ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"Path to the checkpoint file to check."});

  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
    Checks a checkpoint file's consistency and prints its header. Stops the run (std::exit).

    YAML example:

    simulation:
      - check_dump:
         filename: "hippoLBMOutputDir/CheckPointRestart/hippoLBM_0000000100.dump"
    )EOF";
  }

  inline void execute() final {
    int rank;
    MPI_Comm_rank(*mpi, &rank);

    DumpHeader header = read_dump_header(*mpi, *filename);

    // MPI_File_open/close are collective, so every rank must call them; everything
    // else below (get_size, index read, the check itself, printing) only needs to
    // happen once, so it is done on rank 0 only.
    MPI_File file;
    MPI_File_open(*mpi, filename->c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    if (rank == 0) {
      MPI_Offset file_size_mpi = 0;
      MPI_File_get_size(file, &file_size_mpi);
      const uint64_t file_size = file_size_mpi;

      lout << "=================================" << std::endl;
      lout << "== Checkpoint file: " << *filename << std::endl;
      lout << "== Magic:       " << (header.magic_ == DumpHeader::MAGIC ? "OK" : "INVALID") << std::endl;
      lout << "== Version:     " << header.version_ << std::endl;
      lout << "== Q:           " << header.Q_ << std::endl;
      lout << "== Timestep:    " << header.timestep_ << std::endl;
      lout << "== Ranks:       " << header.num_ranks_ << std::endl;
      lout << "== Global size: (" << header.global_size_[0] << ", " << header.global_size_[1] << ", "
           << header.global_size_[2] << ")" << std::endl;
      lout << "== dx:          " << header.dx_ << std::endl;
      lout << "== Fields (" << header.num_fields_ << "):" << std::endl;
      const int int32_type = int(DumpFieldType::INT32);  // 0 -> 32, 1 -> 64
      for (int i = 0; i < header.num_fields_; i++) {
        lout << "==   " << header.fields_[i].name_ << ": " << header.fields_[i].components_ << " x "
             << (header.fields_[i].datatype_ == int32_type ? "int32" : "float64") << std::endl;
      }
      header.params_.print();

      bool ok = (header.magic_ == DumpHeader::MAGIC);

      const uint64_t num_ranks = header.num_ranks_;
      const uint64_t index_section_offset = sizeof(DumpHeader);
      const uint64_t index_section_size = num_ranks * sizeof(DumpRankIndex);
      const uint64_t data_section_offset = index_section_offset + index_section_size;

      if (file_size < data_section_offset) {
        std::cerr << "== ERROR: file is smaller than the header + index table (expected at least "
                  << data_section_offset << " bytes, got " << file_size << ")." << std::endl;
        ok = false;
      } else {
        std::vector<DumpRankIndex> index(header.num_ranks_);
        const MPI_Offset index_offset_mpi = index_section_offset;
        const int index_size_mpi = index_section_size;
        MPI_File_read_at(file, index_offset_mpi, index.data(), index_size_mpi, MPI_BYTE, MPI_STATUS_IGNORE);

        uint64_t max_offset = data_section_offset;
        // check every rank's block, for every field, against the data section and the actual file size
        for (int r = 0; r < header.num_ranks_; r++) {
          for (int s = 0; s < header.num_fields_; s++) {
            const uint64_t block_end = index[r].offset_[s] + index[r].compressed_size_[s];
            // check that the block is within the data section
            if (index[r].compressed_size_[s] > 0 && index[r].offset_[s] < data_section_offset) {
              std::cerr << "== ERROR: rank " << r << " field '" << header.fields_[s].name_ << "' offset "
                        << index[r].offset_[s] << " falls before the data section (starts at " << data_section_offset
                        << ")." << std::endl;
              ok = false;
            }
            if (block_end > file_size) {  // block ends beyond the file size
              std::cerr << "== ERROR: rank " << r << " field '" << header.fields_[s].name_ << "' block ends at "
                        << block_end << ", beyond the file size (" << file_size << ")." << std::endl;
              ok = false;
            }
            max_offset = std::max(max_offset, block_end);
          }
        }

        lout << "== Header + index size: " << data_section_offset << " bytes" << std::endl;
        lout << "== Data end (index):    " << max_offset << " bytes" << std::endl;
        lout << "== Actual file size:    " << file_size << " bytes" << std::endl;
        if (file_size != max_offset) {
          std::cerr << "== WARNING: file size does not exactly match the data described by the index." << std::endl;
          ok = false;
        }
      }

      lout << "== Consistency check: " << (ok ? "PASSED" : "FAILED") << std::endl;
      lout << "=================================" << std::endl;
      lout << "== [check_dump]: stopping execution here." << std::endl;
    }

    MPI_File_close(&file);

    MPI_Barrier(*mpi);
    std::exit(EXIT_SUCCESS);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(check_dump) {
  OperatorNodeFactory::instance()->register_factory("check_dump", make_simple_operator<CheckDumpLBM>);
}
}  // namespace hippoLBM
