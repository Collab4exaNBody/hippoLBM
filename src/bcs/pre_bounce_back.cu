#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <onika/math/basic_types.h>
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/comm.hpp>
#include <grid_lbm/enum.hpp>
#include <grid_lbm/grid_data_lbm.hpp>
#include <grid_lbm/parallel_for_core.cu>
#include <grid_lbm/traversal_lbm.hpp>
#include <hippoLBM/bcs/bounce_back.hpp>
#include <hippoLBM/bcs/bounce_back_manager.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;
  using BoolVector = std::vector<bool>;

  template<int Q>
    class PreBounceBack : public OperatorNode
  {
    ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
    ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);
    ADD_SLOT( bounce_back_manager<Q>, bbmanager, INPUT_OUTPUT);
    ADD_SLOT( BoolVector, periodic   , INPUT , REQUIRED );
    public:
    inline std::string documentation() const override final
    {
      return R"EOF( 
        )EOF";
    }

    template<int dim, Direction dir> 
      void launcher(traversal_lbm& traversals, WrapperF<Q>& pf, bounce_back_manager<Q>& bbm)
      {
        int idx = helper_dim_idx<dim,dir>();
        WrapperF<bounce_back_manager<Q>::Un> pfi = bbm.get_data(idx);
        if( pfi.N > 0 )
        {
          constexpr Traversal Tr = get_traversal<dim, dir>();
          auto [ptr, size] = traversals.get_data<Tr>();
          assert(size == size_t(pfi.N));
          assert(ptr != nullptr);

          ParallelForOptions opts;
          opts.omp_scheduling = OMP_SCHED_STATIC;
          pre_bounce_back<dim, dir, Q> kernel = {ptr};
          auto params = make_tuple(pf, pfi);
          parallel_for_id_runner runner = {kernel, params};
          parallel_for(size, runner, parallel_execution_context(), opts);
        }
      }

    inline void execute () override final
    {
      auto& data = *GridDataQ;
      auto& traversals = *Traversals;
      auto& domain = *DomainQ;
      grid<3>& Grid = domain.m_grid;

      // fill grid size;
      constexpr Area L = Area::Local;
      constexpr Traversal R = Traversal::Real;
      auto br = Grid.build_box<L, R>();
      onika::math::IJK local_grid_size(br.get_length(0), br.get_length(1), br.get_length(2));

      // storage
      auto& bb = *bbmanager;
      bb.resize_data(*periodic, local_grid_size, domain.MPI_coord, domain.MPI_grid_size);

      // get fields
      WrapperF<Q> pf = data.distributions();

      // for clarity
      constexpr int dim_x = 0;
      constexpr int dim_y = 1;
      constexpr int dim_z = 2;
      launcher<dim_x, Direction::Left>(traversals, pf, bb);
      launcher<dim_x, Direction::Right>(traversals, pf, bb);
      launcher<dim_y, Direction::Left>(traversals, pf, bb);
      launcher<dim_y, Direction::Right>(traversals, pf, bb);
      launcher<dim_z, Direction::Left>(traversals, pf, bb);
      launcher<dim_z, Direction::Right>(traversals, pf, bb);
    }
  };

  using PreBounceBack3D19Q = PreBounceBack<19>;

  // === register factories ===  
  ONIKA_AUTORUN_INIT()
  {
    OperatorNodeFactory::instance()->register_factory( "pre_bounce_back", make_compatible_operator<PreBounceBack3D19Q>);
  }
}
