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
#include <grid/make_variant_operator.hpp>
#include <hippoLBM/grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/init_obst.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class InitObstLBM : public OperatorNode
  {
    public:
      ADD_SLOT( LBMDomain<Q>, lbm_domain, INPUT, REQUIRED);
      ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT);

      inline void execute () override final
      {
        auto& data = *LBMFieds;
        auto& domain = *lbm_domain;
        init_obst func = {onika::cuda::vector_data(data.obst)};
        constexpr Area A = Area::Local;
        constexpr Traversal Tr = Traversal::All;
        parallel_for_id<A,Tr>(domain.m_grid, func, parallel_execution_context());       
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_obstacles)
  {
    OperatorNodeFactory::instance()->register_factory( "init_obst", make_variant_operator<InitObstLBM>);
  }
}

