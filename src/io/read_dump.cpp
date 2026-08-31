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

#include <array>

// onika
#include <onika/cuda/cuda.h>
#include <onika/log.h>
#include <onika/memory/allocator.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/grid/make_domain.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

// Implementation
#include <hippoLBM/io/dump_fields.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
class ReadDumpLBM : public OperatorNode {
 public:
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator."});
  ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"Path to the checkpoint file to read."});
  ADD_SLOT(bool, skip_params, INPUT, false, DocString{"If true, keep the current Params instead of the checkpoint's."});
  ADD_SLOT(bool, display_info, INPUT, true, DocString{"Print the checkpoint's domain and LBM parameters."});
  ADD_SLOT(std::vector<bool>, periodic, INPUT, OPTIONAL,
           DocString{"If set, use this periodicity instead of the checkpoint's."});

  ADD_SLOT(LBMDomain<Q>, domain, OUTPUT,
           DocString{"The domain rebuilt from the checkpoint's grid, bounds, and periodicity."});
  ADD_SLOT(LBMFields<Q>, fields, OUTPUT,
           DocString{"The LBM fields, sized for `domain` and filled in from the checkpoint."});
  ADD_SLOT(LBMParameters, Params, INPUT_OUTPUT, LBMParameters{},
           DocString{"The LBM parameters, overwritten from the checkpoint unless skip_params is set."});
  ADD_SLOT(long, timestep, INPUT_OUTPUT, 0, DocString{"Set to the checkpoint's timestep."});
  ADD_SLOT(double, physical_time, INPUT_OUTPUT, 0.0, DocString{"Set to the checkpoint's physical time."});
  ADD_SLOT(double, dt, OUTPUT,
           DocString{"Set to the checkpoint's LBM timestep (Params.dtLB_), unless skip_params is set."});
  ADD_SLOT(bool, do_set_distributions, OUTPUT,
           DocString{"Set to false: the checkpoint already provides f_, skip the default set_distributions."});

  inline std::string documentation() const final {
    return R"EOF(
    Reads back a checkpoint file written by `dump`: rebuilds the domain, sizes and fills
    in `fields`, and (unless skip_params) `Params` and `timestep`.

    YAML example:

      - read_dump:
         filename: "hippoLBMOutputDir/CheckPointRestart/hippoLBM_0000000100.dump"
    )EOF";
  }

  inline void execute() final {
    DumpHeader header = read_dump_header(*mpi, *filename);

    std::array<bool, 3> periodic_flags = {bool(header.periodic_[0]), bool(header.periodic_[1]),
                                          bool(header.periodic_[2])};
    if (periodic.has_value()) {
      auto p = *periodic;
      for (int dim = 0; dim < 3; dim++) {
        periodic_flags[dim] = p[dim];
      }
    }

    if (*display_info) {
      int rank;
      MPI_Comm_rank(*mpi, &rank);
      if (rank == 0) {
        lout << "== Reading checkpoint: " << *filename << std::endl;
        lout << "== Timestep:    " << header.timestep_ << std::endl;
        lout << "== Phys. time:  " << header.physical_time_ << std::endl;
        lout << "== Global size: (" << header.global_size_[0] << ", " << header.global_size_[1] << ", "
             << header.global_size_[2] << ")" << std::endl;
        lout << "== dx:          " << header.dx_ << std::endl;
        lout << "== Periodic:    (" << periodic_flags[0] << ", " << periodic_flags[1] << ", " << periodic_flags[2]
             << ")" << (periodic.has_value() ? " (overwritten)" : "") << std::endl;
        if (*skip_params) {
          lout << "== Params:      skipped (skip_params=true)" << std::endl;
        } else {
          header.params_.print();
        }
      }
    }

    // rebuild the domain from the checkpoint's grid, bounds, and periodicity
    GridConfig grid;
    grid.dims_ = onika::math::IJK{header.global_size_[0], header.global_size_[1], header.global_size_[2]};
    grid.bounds_.bmin = {header.bounds_bmin_[0], header.bounds_bmin_[1], header.bounds_bmin_[2]};
    grid.bounds_.bmax = {header.bounds_bmax_[0], header.bounds_bmax_[1], header.bounds_bmax_[2]};
    grid.periodic_ = periodic_flags;

    SubGridConfig sub_grid = load_balancing(grid, *mpi);
    *domain = make_domain<Q>(grid, sub_grid);

    resize_lbm_fields<Q>(*domain, *fields);

    if (!*skip_params) {
      *Params = header.params_;
      *dt = header.params_.dtLB_;
    }
    *timestep = header.timestep_;
    *physical_time = header.physical_time_;

    std::vector<DumpFieldSource> destinations = hippolbm_dump_fields<Q>(*fields);
    read_dump_fields(*mpi, *filename, header, *domain, destinations);

    *do_set_distributions = false;  // avoid some standard initialization of f_.
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(read_dump) {
  OperatorNodeFactory::instance()->register_factory("read_dump", make_variant_operator<ReadDumpLBM>);
}
}  // namespace hippoLBM
