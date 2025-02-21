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
#include <hippoLBM/bcs/bounce_back_manager.hpp>
#include <hippoLBM/bcs/cavity.hpp>

namespace hippoLBM
{
	using namespace onika;
	using namespace scg;
	using namespace onika::cuda;

	template<int Q>
		class CavityZL : public OperatorNode
	{
		typedef std::array<double,3> readVec3;
		ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
		ADD_SLOT( readVec3, U, INPUT, REQUIRED, DocString{"Prescribed velocity at the boundary (z = lz), enforcing the Cavity condition."});
    ADD_SLOT( bounce_back_manager<Q>, bbmanager, INPUT_OUTPUT, REQUIRED);

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( This operator enforces a Cavity boundary condition at z = lz in an LBM simulation. 
                      The Cavity boundary condition ensures that the gradient of the distribution function 
                      follows a prescribed value
        )EOF";
		}

		inline void execute () override final
		{
			auto& data = *GridDataQ;
			auto& bb = *bbmanager;

			// define functors
			cavity_z_l<Q> bcs = {};

			auto [ux,uy,uz] = *U;

			// get fields
      constexpr int dimZ = 2;
      constexpr int idx = helper_dim_idx<dimZ,Direction::Right>();
      WrapperF<5> pfi = bb.get_data(idx);
			int * const pobst = data.obstacles();
      auto [pex, pey, pez] = data.exyz();
      const double * const pw = data.weights();

			// run kernel
      auto params = make_tuple(pobst, pfi, ux, uy, uz, pw, pex, pey, pez, 30,30,30);
      parallel_for_id_runner runner = {bcs, params};
      //onika::lout << " Fi size: "<< pfi.N << std::endl;
      parallel_for(pfi.N, runner, parallel_execution_context(), ParallelForOptions());
			//parallel_for_simple(size, bcs, parallel_execution_context(), pobst, pf, ux, uy, uz, pw, pex, pey, pez, 30,30,30);
		}
	};

	using CavityZL3D19Q = CavityZL<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT()
	{
		OperatorNodeFactory::instance()->register_factory( "cavity_z_l", make_compatible_operator<CavityZL3D19Q>);
	}
}

