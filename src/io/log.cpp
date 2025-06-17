#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <grid/make_variant_operator.hpp>
#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <hippoLBM/grid/lbm_domain.hpp>
#include <grid/enum.hpp>
#include <onika/string_utils.h>

#include<chrono>

namespace hippoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class LogLBM : public OperatorNode
	{

    typedef std::chrono::time_point<std::chrono::steady_clock> time_point;

		public:
			ADD_SLOT( LBMDomain<Q>, lbm_domain, INPUT_OUTPUT, REQUIRED);
			ADD_SLOT( long , timestep , INPUT, REQUIRED);
	    ADD_SLOT( double , physical_time , INPUT, REQUIRED);
			ADD_SLOT( bool , print_log_header, INPUT_OUTPUT, true);
      ADD_SLOT( time_point , previous_time, PRIVATE , time_point() );
      ADD_SLOT( long , previous_step, PRIVATE , 0 );

			inline void execute () override final
			{
				auto& domain = *lbm_domain; 
        auto [lx, ly, lz] = domain.domain_size;
        long long int size_xyz = (long long int)(lx) * (long long int)(ly) * (long long int)(lz);

        double MLUPS; // Million Lattice Updates Per Second

        auto current_time = std::chrono::steady_clock::now();       
        if( *previous_step == 0 ) 
        {
          MLUPS = 0.0;
        }
        else
        {
          // basic timers
          double T = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - *previous_time).count() * 1e-9;
          assert(*timestep > *previous_step);
          long Nst = *timestep - *previous_step;
          MLUPS = double( size_xyz * Nst ) / (1e6 * T);
        }
        *previous_time = current_time;
        *previous_step = *timestep;

				std::string  header = "     Step     Time          Mesh Size     MLUPS";
				std::string line = format_string("%9ld % .6e %13lld  %.2e", *timestep, *physical_time,  size_xyz, MLUPS);

				if(*print_log_header)
				{
					lout << header << std::endl;
				}
				lout << line << std::endl;

        *print_log_header = false;
			}
	};

	// === register factories ===  
	ONIKA_AUTORUN_INIT()
	{
		OperatorNodeFactory::instance()->register_factory( "log", make_variant_operator<LogLBM>);
	}
}

