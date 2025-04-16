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
#include <grid/domain_lbm.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/traversal_lbm.hpp>
#include <grid/lbm_parameters.hpp>
#include <hippoLBM/collision/bgk.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class CollisionBGK : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_fields<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
      ADD_SLOT( LBMParameters, Params, INPUT, REQUIRED, DocString{"Contains global LBM simulation parameters"});

      inline std::string documentation() const override final
      {
        return R"EOF( The `CollisionBGK` operator implements the Bhatnagar-Gross-Krook (BGK) collision model for the Lattice Boltzmann Method (LBM). This model assumes a single relaxation time approach  to approximate the collision process, driving the distribution functions toward equilibrium.
        )EOF";
      }

      inline void execute () override final
      {
        auto& data = *GridDataQ;
        auto& traversals = *Traversals;
        auto& params = *Params;

        // define functor
        bgk<Q> func = {params.Fext};

        // get fields
        FieldView<3> pm1 = data.flux();
        int * const pobst = data.obstacles();
        FieldView<Q> pf = data.distributions();
        double * const pm0 = data.densities();
        const double * const w = data.weights();
        auto [pex, pey, pez] = data.exyz();

        // get traversal
        auto [ptr, size] = traversals.get_data<Traversal::Real>();

        // run kernel
        parallel_for_id(ptr, size, func, parallel_execution_context(), pm1, pobst, pf, pm0, pex, pey, pez, w, params.tau);
      }
  };

  using CollisionBGK3D19Q = CollisionBGK<19>;

  // === register factories ===  
  ONIKA_AUTORUN_INIT(CollisionBGK)
  {
    OperatorNodeFactory::instance()->register_factory( "bgk", make_compatible_operator<CollisionBGK3D19Q>);
  }
}

