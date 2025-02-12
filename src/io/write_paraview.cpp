#include <mpi.h>
#include <filesystem>

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
#include <hipoLBM/io/write_paraview.hpp>

#include <onika/string_utils.h>

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
			ADD_SLOT( MPI_Comm, mpi, INPUT , MPI_COMM_WORLD);
			ADD_SLOT( std::string, filename, INPUT, "hipoLBM_%010d");
			ADD_SLOT( std::string, basedir, INPUT, "hipoLBMOutputDir/ParaviewOutput/");
			ADD_SLOT( long, timestep, INPUT, 0);
			ADD_SLOT( bool, distributions, INPUT, false);
			inline void execute () override final
			{
				auto& comm = *mpi;
				int rank, size;
				MPI_Comm_rank(comm, &rank);
				MPI_Comm_size(comm, &size);

				std::string file_name = *filename;
        file_name = format_string(file_name, *timestep);
        std::string fullname = *basedir + file_name;

				if(rank == 0)
        {
          std::filesystem::create_directories( fullname );
        }
        fullname += "/%06d";
        fullname = format_string(fullname, rank);

				auto& domain = *DomainQ;
				auto& data = *GridDataQ;
				auto& traversals = *Traversals;

				MPI_Barrier(comm);
				write_pvtr(*basedir, file_name, size, domain, *distributions);
        write_vtr( fullname, domain, data, traversals, *distributions);
			}
	};

	using WriteParaviewLBM3D19Q = WriteParaviewLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT()
	{
		OperatorNodeFactory::instance()->register_factory( "write_paraview", make_compatible_operator<WriteParaviewLBM3D19Q>);
	}
}

