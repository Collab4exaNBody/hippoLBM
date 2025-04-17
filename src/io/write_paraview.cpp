#include <mpi.h>
#include <filesystem>

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>

#include <grid/make_variant_operator.hpp>
#include <grid/lbm_domain.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/traversal_lbm.hpp>
#include <grid/lbm_parameters.hpp>
#include <hippoLBM/io/write_paraview.hpp>

#include <onika/string_utils.h>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class WriteParaviewLBM : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_domain<Q>, LBMDomain, INPUT);
      ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT);
      ADD_SLOT( traversal_lbm, Traversals, INPUT);
      ADD_SLOT( LBMParameters, Params, INPUT);
      ADD_SLOT( MPI_Comm, mpi, INPUT , MPI_COMM_WORLD);
      ADD_SLOT( std::string, filename, INPUT, "hippoLBM_%010d");
      ADD_SLOT( std::string, basedir, INPUT, "hippoLBMOutputDir/ParaviewOutput/");
      ADD_SLOT( long, timestep, INPUT, 0);
      ADD_SLOT( bool, distributions, INPUT, false);
      inline void execute () override final
      {
        auto& comm = *mpi;
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        std::string file_name = *filename;
        file_name = onika::format_string(file_name, *timestep);
        std::string fullname = *basedir + file_name;

        if(rank == 0)
        {
          std::filesystem::create_directories( fullname );
        }
        fullname += "/%06d";
        fullname = onika::format_string(fullname, rank);

        auto& domain = *LBMDomain;
        auto& data = *LBMFieds;
        auto& traversals = *Traversals;

        MPI_Barrier(comm);
        write_pvtr(*basedir, file_name, size, domain, *distributions);
        write_vtr( fullname, domain, data, traversals, *Params, *distributions);
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(write_paraview)
  {
    OperatorNodeFactory::instance()->register_factory( "write_paraview", make_variant_operator<WriteParaviewLBM>);
  }
}

