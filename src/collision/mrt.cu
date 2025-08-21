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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_operators.h>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/parallel_for_core.cu>
#include <hippoLBM/grid/grid_region.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/collision/mrt.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class CollisionMRT : public OperatorNode
  {
    public:
      ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( LBMGridRegion, grid_region, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
      ADD_SLOT( LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});

      inline std::string documentation() const override final
      {
        return R"EOF( The `CollisionMRT` operator implements the MRT collision model for the Lattice Boltzmann Method (LBM). This model assumes a single relaxation time approach  to approximate the collision process, driving the distribution functions toward equilibrium.
        )EOF";
      }

      inline void execute () override final
      {
        auto& data = *fields;
        auto& traversals = *grid_region;
        auto& params = *Params;

        // define functor
        mrt<Q> func = {params.Fext};

        // get fields
        int * const pobst = data.obstacles();
        FieldView<Q> pf = data.distributions();
        double * const pm0 = data.densities();
        const double * const w = data.weights();
        auto [pex, pey, pez] = data.exyz();

        // get traversal
        auto [ptr, size] = traversals.get_data<Traversal::Real>();

        // run kernel
        parallel_for_id(ptr, size, func, parallel_execution_context(), pobst, pf, pm0, pex, pey, pez, w, params.tau);
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(CollisionMRT)
  {
    OperatorNodeFactory::instance()->register_factory( "mrt", make_variant_operator<CollisionMRT>);
  }
}

