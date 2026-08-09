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
#include <onika/math/basic_types.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// hippoLBM
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>

namespace hippoLBM {
using namespace onika;
using namespace scg;

template <int Q>
class DefineLBMFields : public OperatorNode {
 public:
  ADD_SLOT(LBMDomain<Q>, domain, INPUT, REQUIRED, DocString{"The LBM domain."});
  ADD_SLOT(LBMFields<Q>, fields, INPUT_OUTPUT, DocString{"The LBM fields."});

  inline std::string documentation() const final {
    return R"EOF(
    This operator defines the fields required for the LBM simulation.

    YAML example:

      - define_grid_3dq19
    )EOF";
  }

  inline void execute() final { resize_lbm_fields<Q>(*domain, *fields); }
};

// === register factories ===
ONIKA_AUTORUN_INIT(define_grid) {
  OperatorNodeFactory::instance()->register_factory("define_grid_3dq19", make_simple_operator<DefineLBMFields<19>>);
  // OperatorNodeFactory::instance()->register_factory( "define_grid_3dq15",
  // make_compatible_operator<DefineLBMFields<Q><15>>);
}
}  // namespace hippoLBM
