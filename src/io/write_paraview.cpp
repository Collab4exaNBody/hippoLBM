#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/enum.hpp>
#include <grid_lbm/grid_data_lbm.hpp>
#include <grid_lbm/traversal_lbm.hpp>


namespace hipoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class WriteParaviewLBM : public OperatorNode
	{
		public:
			ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT);
			ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT);
			ADD_SLOT( traversal_lbm, Traversals, INPUT);
			inline void execute () override final
			{
        auto& domain = *DomainQ;
        auto& data = *GridDataQ;
        auto& traversals = *Traversals;
			}
	};

	using WriteParaviewLBM3D19Q = WriteParaviewLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(parallel_for_benchmark)
	{
		OperatorNodeFactory::instance()->register_factory( "write_paraview", make_compatible_operator<WriteParaviewLBM3D19Q>);
	}
}

