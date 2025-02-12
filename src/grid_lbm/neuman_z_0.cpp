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
#include <grid_lbm/parallel_for_core.hpp>
#include <grid_lbm/traversal_lbm.hpp>
#include <grid_lbm/lbm_parameters.hpp>
#include <grid_lbm/neumann.hpp>

namespace hipoLBM
{
	using namespace onika;
	using namespace scg;
	using namespace onika::cuda;

	template<int Q>
		class NeumannZ0 : public OperatorNode
	{
		public:
			ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED);
			ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED);
      ADD_SLOT( math::Vec3d, U, INPUT, REQUIRED);

			inline void execute () override final
			{
				auto& data = *GridDataQ;
				auto& traversals = *Traversals;

				// define functors
				neumann_z_0<Q> neumann = {};

        auto [ux,uy,uz] = *U;

				// get fields
				double * const pf = data.distributions();
				int * const pobst = data.obstacles();

				// get traversal
				auto [ptr, size] = traversals.get_data<Traversal::Plan_xy_0>();
				// run kernel
				parallel_for_id(ptr, size, neumann, parallel_execution_context(), pobst, pf, ux, uy, uz);
			}
	};

	using NeumannZ03D19Q = NeumannZ0<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT()
	{
		OperatorNodeFactory::instance()->register_factory( "neumann_z_0", make_compatible_operator<NeumannZ03D19Q>);
	}
}

