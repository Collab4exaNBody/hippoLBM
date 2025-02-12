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
#include <grid_lbm/enum.hpp>
#include <onika/string_utils.h>

namespace hipoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class LogLBM : public OperatorNode
	{
		public:
			ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT_OUTPUT, REQUIRED);
			ADD_SLOT( long , timestep , INPUT, REQUIRED);
	    ADD_SLOT( double , physical_time , INPUT, REQUIRED);
			ADD_SLOT( bool , print_log_header, INPUT_OUTPUT, true);

			inline void execute () override final
			{
				//constexpr Area G = Area::Global;
				//constexpr Traversal R = Traversal::Real;

				//domain_lbm<Q>& domain = *DomainQ; 
				//grid<3>& g = domain.m_grid;

				std::string  header = "     Step     Time          Mesh Size";
				std::string line = format_string("%9ld % .6e %13ld", *timestep, *physical_time,  1000);

				if(*print_log_header)
				{
					lout << header << std::endl;
				}
				lout << line << std::endl;

        *print_log_header = false;
			}
	};

	using LogLBM3D19Q = LogLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT()
	{
		OperatorNodeFactory::instance()->register_factory( "log", make_compatible_operator<LogLBM3D19Q>);
	}
}

