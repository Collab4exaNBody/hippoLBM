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
#include <onika/math/basic_types_operators.h>
#include <grid/domain_lbm.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/traversal_lbm.hpp>
#include <hippoLBM/collision/streaming.hpp>
#include <grid/update_ghost.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class StreamingLBM : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_fields<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
      ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);
      ADD_SLOT( bool, asynchrone, INPUT, false, DocString{"The asynchrone option controls the execution style: when true, it allows asynchronous operations with overlapping computation and communication, improving parallel performance. When false, it runs synchronously, ensuring sequential execution of operations and data updates."});

      inline std::string documentation() const override final
      {
        return R"EOF(  The StreamingLBM class is described as part of the Lattice Boltzmann Method (LBM) implementation, specifically the streaming steps.)EOF";
      }


      inline void execute () override final
      {
        auto& data = *GridDataQ;
        auto& domain = *DomainQ;
        auto& traversals = *Traversals;
        grid<3>& Grid = domain.m_grid;

        // get fields
        FieldView<Q> pf = data.distributions();
        auto [pex, pey, pez] = data.exyz();

        // define functors
        streaming_step1<Q> step1 = {};
        streaming_step2<Q> step2 = {Grid, pf, pex, pey, pez};

        // capture the parallel execution context
        auto par_exec_ctx = [this] (const char* exec_name)
        { 
          return this->parallel_execution_context(exec_name);
        };

        if( *asynchrone )
        {
          /*
             constexpr Traversal Inside = Traversal::Inside;
             constexpr Traversal Rest = Traversal::Ghost_Edge;

             domain.m_ghost_manager.resize_request();
             domain.m_ghost_manager.do_recv();
             domain.m_ghost_manager.do_pack_send(pf, Grid.bx);

             auto [ptr, size] = traversals.get_data<Inside>();
             box<3> inside = Grid.build_box<Area::Local, Inside>();

             parallel_for_id(ptr, size, step1, parallel_execution_context(), pf);
             parallel_for_box(inside, step2, pf, pex, pey, pez);

             domain.m_ghost_manager.wait_all();
             domain.m_ghost_manager.do_unpack(pf, Grid.bx);

             auto [ptr2, size2] = traversals.get_data<Rest>();

             parallel_for_id(ptr2, size2, step1, parallel_execution_context(), pf);
             parallel_for_ghost_edge(Grid, step2, pf, pex, pey, pez);
           */
        }
        else
        {
          // get traversal
          auto [ptr, size] = traversals.get_data<Traversal::Real>();

          // run kernel
          parallel_for_id(ptr, size, step1, parallel_execution_context("streaming_step1"), pf);
          update_ghost(domain, pf, par_exec_ctx);
          box<3> extend = Grid.build_box<Area::Local, Traversal::Extend>();
          onika::parallel::ParallelExecutionSpace<3> parallel_range = set(extend);        
          parallel_for(parallel_range, step2, parallel_execution_context("streaming_step2"));
        }
      }
  };

  using StreamingLBM3D19Q = StreamingLBM<19>;

  // === register factories ===  
  ONIKA_AUTORUN_INIT()
  {
    OperatorNodeFactory::instance()->register_factory( "streaming", make_compatible_operator<StreamingLBM3D19Q>);
  }
}

