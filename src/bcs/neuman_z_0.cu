#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <grid/make_variant_operator.hpp>
#include <onika/math/basic_types.h>
#include <hippoLBM/grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/traversal_lbm.hpp>
#include <hippoLBM/bcs/neumann.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class NeumannZ0 : public OperatorNode
  {
    typedef std::array<double,3> readVec3;
    ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
    ADD_SLOT( readVec3, U, INPUT, REQUIRED, DocString{"Prescribed velocity at the boundary (z = 0), enforcing the Neumann condition."});
    public:
    inline std::string documentation() const override final
    {
      return R"EOF( This operator enforces a Neumann boundary condition at z = 0 in an LBM simulation. 
                      The Neumann boundary condition ensures that the gradient of the distribution function 
                      follows a prescribed value.
        )EOF";
    }

    inline void execute () override final
    {
      auto& data = *LBMFieds;
      auto& traversals = *Traversals;

      // define functors
      neumann_z_0<Q> neumann = {};

      auto [ux,uy,uz] = *U;

      // get fields
      FieldView<Q> pf = data.distributions();
      int * const pobst = data.obstacles();

      // get traversal
      auto [ptr, size] = traversals.get_data<Traversal::Plan_xy_0>();
      // run kernel
      parallel_for_id(ptr, size, neumann, parallel_execution_context(), pobst, pf, ux, uy, uz);
    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT()
  {
    OperatorNodeFactory::instance()->register_factory( "neumann_z_0", make_variant_operator<NeumannZ0>);
  }
}

