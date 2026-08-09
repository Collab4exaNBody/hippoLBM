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
#include <zlib.h>

#include <filesystem>

// onika
#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/memory/allocator.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/string_utils.h>

// hippoLBM
#include <hippoLBM/grid/make_variant_operator.hpp>

// Implementation
#include <hippoLBM/io/dump_fields.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
class DumpLBM : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator."});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain containing the simulation data."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT, REQUIRED, DocString{"The LBM fields containing the simulation data."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"The LBM parameters for the simulation."});
  ADD_SLOT(std::string, filename, INPUT, "hippoLBM_%010d.dump", DocString{"The filename for the checkpoint file."});
  ADD_SLOT(std::string, output_directory, INPUT, "hippoLBMOutputDir",
           DocString{"The base directory for the checkpoint output."});
  ADD_SLOT(long, timestep, INPUT, 0, DocString{"The current timestep."});
  ADD_SLOT(bool, display_filename, INPUT, true, DocString{"Print filename"});
  ADD_SLOT(int, compression_level, INPUT, Z_BEST_SPEED, DocString{"zlib compression level (1 = fastest, 9 = smallest)."});

  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
    This operator writes a checkpoint/restart file for the LBM simulation: the obstacle
    and distribution function fields, compressed with zlib and written in parallel with
    MPI-IO, alongside a header describing the global grid, LBM parameters, and timestep.

    YAML example:

      - dump:
         filename: "hippoLBM_%010d.dump"
         output_directory: "hippoLBMOutputDir"
    )EOF";
  }

  inline void execute() final {
    std::string dump_directory = *output_directory + "/CheckPointRestart/";

    int rank;
    MPI_Comm_rank(*mpi, &rank);
    if (rank == 0) {
      std::filesystem::create_directories(dump_directory);
    }
    MPI_Barrier(*mpi);

    std::string name = onika::format_string(*filename, *timestep);
    std::string fullname = dump_directory + name;

    if (*display_filename) {
      lout << "writing checkpoint file: " << fullname << std::endl;
    }

    std::vector<DumpFieldSource> sources = hippolbm_dump_fields<Q>(*fields);
    DumpHeader header = write_dump_header(*mpi, fullname, *domain, *Params, *timestep, sources);
    write_dump_fields(*mpi, fullname, header, *domain, sources, *compression_level);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(dump) {
  OperatorNodeFactory::instance()->register_factory("dump", make_variant_operator<DumpLBM>);
}
}  // namespace hippoLBM
