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

#include <hippoLBM/grid/make_variant_operator.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/grid/traversal_lbm.hpp>


namespace hippoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class PrintDomainLBM : public OperatorNode
	{
		public:
			ADD_SLOT( LBMDomain<Q>, domain, INPUT);
			inline void execute () override final
			{
        constexpr Area G = Area::Global;
        constexpr Traversal R = Traversal::Real;

        LBMGrid& g = domain->m_grid;
        const onika::math::AABB& bounds = domain->bounds;
        const int3d& domain_size = domain->domain_size;


        lout      << "=================================" << std::endl;
        lout      << "== Domain size: " << bounds << std::endl;
        lout      << "== Grid size: (LX: " << domain_size[0] << ", LY: " << domain_size[1] << ", LZ " << domain_size[2] << ")" << std::endl;
        lout      << "=================================" << std::endl;
        lout      << "== Grid info " << std::endl;
        lout      << "= Rank 0 " << std::endl;
        lout      << "= Real Grid:" << std::endl;
        auto real = g.build_box<G, R>();
        real.print();
        lout      << "= Extended Grid: " << std::endl;
        auto &ext = g.ext;
        ext.print();
        lout      << "=================================" << std::endl;

			}
	};

	// === register factories ===  
	ONIKA_AUTORUN_INIT(print_domain)
	{
		OperatorNodeFactory::instance()->register_factory( "print_domain", make_variant_operator<PrintDomainLBM>);
		OperatorNodeFactory::instance()->register_factory( "hippoLBM_print_domain", make_variant_operator<PrintDomainLBM>);
	}
}

