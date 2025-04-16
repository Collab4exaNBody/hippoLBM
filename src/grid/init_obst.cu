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
#include <grid/domain_lbm.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/init_obst.hpp>

namespace hippoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class InitObstLBM : public OperatorNode
	{
		public:
			ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);
			ADD_SLOT( lbm_fields<Q>, GridDataQ, INPUT_OUTPUT);

			inline void execute () override final
			{
        auto& data = *GridDataQ;
        auto& domain = *DomainQ;
				init_obst func = {onika::cuda::vector_data(data.obst)};
				constexpr Area A = Area::Local;
				constexpr Traversal Tr = Traversal::All;
		    parallel_for_id<A,Tr>(domain.m_grid, func, parallel_execution_context());       
			}
	};

	using InitObstLBM3D19Q = InitObstLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(parallel_for_benchmark)
	{
		OperatorNodeFactory::instance()->register_factory( "init_obst", make_compatible_operator<InitObstLBM3D19Q>);
	}
}

