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
#include <hippoLBM/grid/lbm_domain.hpp>
#include <hippoLBM/grid/make_domain.hpp>
#include <grid/comm.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using onika::math::AABB;
  using BoolVector = std::vector<bool>;

  template<int Q>
    class InitDomainLBM : public OperatorNode
  {
    public:
      ADD_SLOT( MPI_Comm, mpi, INPUT , MPI_COMM_WORLD);
      ADD_SLOT( LBMDomain<Q>, lbm_domain, OUTPUT);
      ADD_SLOT( BoolVector, periodic   , INPUT_OUTPUT , REQUIRED );
      ADD_SLOT( double, resolution, INPUT_OUTPUT, REQUIRED, DocString{"Resolution"});
      ADD_SLOT( AABB, bounds, INPUT_OUTPUT, REQUIRED, DocString{"Domain's bounds"});

      inline void execute () override final
      {
        static_assert(DIM == 3);
        *lbm_domain = make_domain<Q>(*bounds, *resolution, *periodic, *mpi);
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_domain)
  {
    OperatorNodeFactory::instance()->register_factory( "domain", make_variant_operator<InitDomainLBM>);
  }
}

