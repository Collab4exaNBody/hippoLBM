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
#include <grid_lbm/init_densities.hpp>

namespace hipoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class InitDensitiesLBM : public OperatorNode
	{
		public:
			ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT);
			ADD_SLOT( traversal_lbm, Traversals, INPUT);
      ADD_SLOT( bool, do_update, INPUT, false);

			inline void execute () override final
			{
        auto& data = *GridDataQ;
        auto& traversals = *Traversals;

        double * const ptr_f = onika::cuda::vector_data(data.f);
        const double * const ptr_w = onika::cuda::vector_data(data.scheme.w);

        init_densities<Q> func = {};

        if( *do_update )
        {
          auto [ptr, size] = traversals.get_data<Traversal::Real>();
          parallel_for_id(ptr, size, func, parallel_execution_context(), ptr_f, ptr_w);
          // update_ghost()
        }
        else
        {
          auto [ptr, size] = traversals.get_data<Traversal::All>();
          parallel_for_id(ptr, size, func, parallel_execution_context(), ptr_f, ptr_w);
        }
			}
	};

	using InitDensitiesLBM3D19Q = InitDensitiesLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(parallel_for_benchmark)
	{
		OperatorNodeFactory::instance()->register_factory( "init_densities", make_compatible_operator<InitDensitiesLBM3D19Q>);
	}
}

