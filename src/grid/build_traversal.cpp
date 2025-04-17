#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <grid/make_variant_operator.hpp>
#include <grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/traversal_lbm.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class BuildTraversalLBM : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_domain<Q>, LBMDomain, INPUT);
      ADD_SLOT( traversal_lbm, Traversals, OUTPUT);
      inline void execute () override final
      {
        auto& domain = *LBMDomain;
        traversal_lbm traversal;
        traversal.build_traversal(domain.m_grid, domain.MPI_coord, domain.MPI_grid_size);
        *Traversals = traversal;
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(build_traversal)
  {
    OperatorNodeFactory::instance()->register_factory( "build_traversal", make_variant_operator<BuildTraversalLBM>);
  }
}

