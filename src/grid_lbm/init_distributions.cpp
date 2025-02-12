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
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/comm.hpp>
#include <grid_lbm/enum.hpp>
#include <grid_lbm/grid_data_lbm.hpp>
#include <grid_lbm/parallel_for_core.hpp>
#include <grid_lbm/traversal_lbm.hpp>
#include <grid_lbm/init_distributions.hpp>
#include <grid_lbm/update_ghost.hpp>

namespace hipoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class InitDistributionsLBM : public OperatorNode
	{
		public:
			ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT);
      ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);
			ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED);
      ADD_SLOT( bool, do_update, INPUT, false);

			inline void execute () override final
			{
        auto& domain = *DomainQ;
        auto& data = *GridDataQ;
        auto& traversals = *Traversals;

        double * const pf = data.distributions();
        const double * const pw = data.weights();

        init_distributions<Q> func = {};

        if( *do_update )
        {
          auto [ptr, size] = traversals.get_data<Traversal::Real>();
          parallel_for_id(ptr, size, func, parallel_execution_context(), pf, pw);
          update_ghost(domain, pf);
        }
        else
        {
          auto [ptr, size] = traversals.get_data<Traversal::All>();
          parallel_for_id(ptr, size, func, parallel_execution_context(), pf, pw);
        }
			}
	};

	using InitDistributionsLBM3D19Q = InitDistributionsLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(parallel_for_benchmark)
	{
		OperatorNodeFactory::instance()->register_factory( "init_distributions", make_compatible_operator<InitDistributionsLBM3D19Q>);
	}
}
