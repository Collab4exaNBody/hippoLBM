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

// Implemtation
#include <hippoLBM/io/write_paraview_subbox.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
class WriteParaviewSubboxLBM : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator."});
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain containing the simulation data."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT, REQUIRED, DocString{"The LBM fields containing the simulation data."});
  ADD_SLOT(LBMParameters, Params, INPUT, REQUIRED, DocString{"The LBM parameters for the simulation."});
  ADD_SLOT(onika::math::AABB, bounds, INPUT, REQUIRED, DocString{"Physical-space sub-box of the domain to write out."});
  ADD_SLOT(std::string, filename, INPUT, "hippoLBM_subbox_%010d", DocString{"The filename for the Paraview output."});
  ADD_SLOT(std::string, output_directory, INPUT, "hippoLBMOutputDir",
           DocString{"The base directory for the Paraview output."});
  ADD_SLOT(long, timestep, INPUT, 0, DocString{"The current timestep."});
  ADD_SLOT(bool, display_filename, true, DocString{"Print filename"});

  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
    This operator writes a subbox of the simulation grid.

    Parameters:

    - bounds [AABB] : Physical-space sub-box of the domain to write out. Required.
    - filename [string] : Filename pattern for the Paraview output (printf-style, receives the timestep).
      Default: "hippoLBM_subbox_%010d".
    - display_filename [bool] : Whether to print the output filename when writing. Default: true.

    YAML example:

      dump_paraview:
        - write_paraview_subbox:
          filename: "hippoLBM_subbox_%010d"
          bounds: [[0.4, 0., 0.], [0.6, 2.0, 2.0]]
    )EOF";
  }

  inline void execute() final {
    std::string OutPutDirectory = *output_directory + "/ParaviewOutput/";
    write_paraview_subbox(*mpi, *filename, OutPutDirectory, *timestep, *fields, *Params, *domain, *bounds,
                          *display_filename);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(write_paraview_subbox) {
  OperatorNodeFactory::instance()->register_factory("write_paraview_subbox",
                                                    make_variant_operator<WriteParaviewSubboxLBM>);
}
}  // namespace hippoLBM
