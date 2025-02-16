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
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/comm.hpp>
#include <grid_lbm/enum.hpp>
#include <grid_lbm/grid_data_lbm.hpp>
#include <grid_lbm/parallel_for_core.cu>
#include <grid_lbm/traversal_lbm.hpp>
#include <grid_lbm/lbm_parameters.hpp>
#include <hipoLBM/bcs/neumann.hpp>

namespace hipoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class NeumannZL : public OperatorNode
  {
    typedef std::array<double,3> readVec3;
    ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
    ADD_SLOT( readVec3, U, INPUT, REQUIRED, DocString{"Prescribed velocity at the boundary (z = lz), enforcing the Neumann condition."});

    public:
    inline std::string documentation() const override final
    {
      return R"EOF( This operator enforces a Neumann boundary condition at z = lz in an LBM simulation. 
                      The Neumann boundary condition ensures that the gradient of the distribution function 
                      follows a prescribed value
        )EOF";
    }

    inline void execute () override final
    {
      auto& data = *GridDataQ;
      auto& traversals = *Traversals;

      // define functors
      neumann_z_l<Q> neumann = {};

      auto [ux,uy,uz] = *U;

      // get fields
      WrapperF<Q> pf = data.distributions();
      int * const pobst = data.obstacles();

      // get traversal
      auto [ptr, size] = traversals.get_data<Traversal::Plan_xy_l>();
      // run kernel
      parallel_for_id(ptr, size, neumann, parallel_execution_context(), pobst, pf, ux, uy, uz);
    }
  };

  using NeumannZL3D19Q = NeumannZL<19>;

  // === register factories ===  
  ONIKA_AUTORUN_INIT()
  {
    OperatorNodeFactory::instance()->register_factory( "neumann_z_l", make_compatible_operator<NeumannZL3D19Q>);
  }
}

