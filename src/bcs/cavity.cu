#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <hippoLBM/grid/make_variant_operator.hpp>
#include <onika/math/basic_types.h>
#include <grid/enum.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <grid/comm.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/traversal_lbm.hpp>
#include <hippoLBM/bcs/bounce_back_manager.hpp>
#include <hippoLBM/bcs/cavity.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

	template<int Dim, Side S, int Q>
		class Cavity : public OperatorNode
	{
		typedef std::array<double,3> readVec3;
		ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
		ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
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
			auto& data = *fields;
			auto& bb = *bbmanager;
			auto [lx, ly, lz] = domain->domain_size;
			auto [ux,uy,uz] = *U;

			// define functors
			cavity<Dim, S, Q> bcs = {};

			// get fields
			constexpr int idx = helper_dim_idx<Dim,S>();
			FieldView<5> pfi = bb.get_data(idx);
			int * const pobst = data.obstacles();
			auto [pex, pey, pez] = data.exyz();
			const double * const pw = data.weights();

			// initialize coefficients
			bcs.compute_coeff(ux, uy, uz, pw, pex, pey, pez, lx, ly, lz);

			// run kernel
			auto params = make_tuple(pobst, pfi);
			parallel_for_id_runner runner = {bcs, params};
			parallel_for(pfi.num_elements, runner, parallel_execution_context(), ParallelForOptions());
		}
	};

	template<int Q> using CavityZ0_3D19Q = Cavity<DIMZ, Side::Left, Q>;
	template<int Q> using CavityZL_3D19Q = Cavity<DIMZ, Side::Right,Q>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(cavity)
	{
		OperatorNodeFactory::instance()->register_factory( "cavity_z_0", make_variant_operator<CavityZ0_3D19Q>);
		OperatorNodeFactory::instance()->register_factory( "cavity_z_l", make_variant_operator<CavityZL_3D19Q>);
	}
}

