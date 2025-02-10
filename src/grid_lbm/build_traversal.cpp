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
#include <grid_lbm/traversal_lbm.hpp>


namespace hipoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class BuildTraversalLBM : public OperatorNode
	{
		public:
			ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT);
			ADD_SLOT( traversal_lbm<Q>, TraversalQ, OUTPUT);
			inline void execute () override final
			{
				auto& domain = *DomainQ;
				traversal_lbm<Q> traversal;
				traversal.build_traversal(domain.m_grid, domain.MPI_coord, domain.MPI_grid_size);
				*TraversalQ = traversal;
			}
	};

	using BuildTraversalLBM3D19Q = BuildTraversalLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(parallel_for_benchmark)
	{
		OperatorNodeFactory::instance()->register_factory( "build_traversal", make_compatible_operator<BuildTraversalLBM3D19Q>);
	}
}

