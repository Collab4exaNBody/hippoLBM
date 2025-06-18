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
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/parallel_for_core.cu>
#include <hippoLBM/grid/init_obst.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class InitObstLBM : public OperatorNode
  {
    public:
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
      ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT);

      inline void execute () override final
      {
        auto& data = *fields;
        init_obst func = {onika::cuda::vector_data(data.obst)};
        constexpr Area A = Area::Local;
        constexpr Traversal Tr = Traversal::All;
        parallel_for_id<A,Tr>(domain->m_grid, func, parallel_execution_context());       
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_obstacles)
  {
    OperatorNodeFactory::instance()->register_factory( "init_obst", make_variant_operator<InitObstLBM>);
  }
}

