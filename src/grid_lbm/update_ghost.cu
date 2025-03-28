#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/comm.hpp>
#include <grid_lbm/enum.hpp>
#include <grid_lbm/grid_data_lbm.hpp>
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/update_ghost.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class UpdateGhost : public OperatorNode
  {
      ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);

    public:

      inline std::string documentation() const override final
      {
        return R"EOF(  A functor for computing macroscopic variables (densities and flux) for lattice Boltzmann method.
        )EOF";
      }

      inline void execute () override final
      {
        auto& data = *GridDataQ;
        auto& domain = *DomainQ;

        // capture the parallel execution context
        auto par_exec_ctx = [this] (const char* exec_name)
        { 
          return this->parallel_execution_context(exec_name);
        };

        // get fields
        WrapperF<Q> pf = data.distributions();
        update_ghost(domain, pf, par_exec_ctx);
      }
  };

  using UpdateGhost3D19Q = UpdateGhost<19>;

  // === register factories ===  
  ONIKA_AUTORUN_INIT(update_ghost)
  {
    OperatorNodeFactory::instance()->register_factory( "update_ghost", make_compatible_operator<UpdateGhost3D19Q>);
  }
}

