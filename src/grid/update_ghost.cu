#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <hippoLBM/grid/domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <grid/update_ghost.hpp>
#include <hippoLBM/grid/make_variant_operator.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class UpdateGhost : public OperatorNode
  {
      ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);

    public:

      inline std::string documentation() const override final
      {
        return R"EOF(  A functor for computing macroscopic variables (densities and flux) for lattice Boltzmann method.
        )EOF";
      }

      inline void execute () override final
      {
        auto& data = *fields;

        // capture the parallel execution context
        auto par_exec_ctx = [this] (const char* exec_name)
        { 
          return this->parallel_execution_context(exec_name);
        };

        // get fields
        FieldView<Q> pf = data.distributions();
        update_ghost(*domain, pf, par_exec_ctx);
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(update_ghost)
  {
    OperatorNodeFactory::instance()->register_factory( "update_ghost", make_variant_operator<UpdateGhost>);
  }
}

