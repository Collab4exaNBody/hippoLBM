#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <hippoLBM/grid/make_variant_operator.hpp>
#include <onika/math/basic_types.h>
#include <hippoLBM/grid/domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/traversal_lbm.hpp>
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
    ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
    ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED, DocString{"It contains different sets of indexes categorizing the grid points into Real, Edge, or All."});
    ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
    ADD_SLOT( bounce_back_manager<Q>, bbmanager, INPUT_OUTPUT);
    ADD_SLOT( BoolVector, periodic   , INPUT , REQUIRED );
    public:
    inline std::string documentation() const override final
    {
      return R"EOF( 
        )EOF";
    }

    template<int dim, Side dir> 
      void launcher(traversal_lbm& traversals, FieldView<Q>& pf, bounce_back_manager<Q>& bbm)
      {
        int idx = helper_dim_idx<dim,dir>();
        FieldView<bounce_back_manager<Q>::Un> pfi = bbm.get_data(idx);
        if( pfi.num_elements > 0 )
        {
          constexpr Traversal Tr = get_traversal<dim, dir>();
          auto [ptr, size] = traversals.get_data<Tr>();
          assert(size == size_t(pfi.num_elements));
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
      auto& data = *fields;
      auto& traversals = *Traversals;
      LBMGrid& Grid = domain->m_grid;

      // fill grid size;
      constexpr Area L = Area::Local;
      constexpr Traversal R = Traversal::All;
      //constexpr Traversal R = Traversal::Real;
      auto br = Grid.build_box<L, R>();
      onika::math::IJK local_grid_size(br.get_length(0), br.get_length(1), br.get_length(2));

      // storage
      auto& bb = *bbmanager;
      bb.resize_data(*periodic, local_grid_size, domain->MPI_coord, domain->MPI_grid_size);

      // get fields
      FieldView<Q> pf = data.distributions();

      // for clarity
      constexpr int dim_x = 0;
      constexpr int dim_y = 1;
      constexpr int dim_z = 2;
      launcher<dim_x, Side::Left>(traversals, pf, bb);
      launcher<dim_x, Side::Right>(traversals, pf, bb);
      launcher<dim_y, Side::Left>(traversals, pf, bb);
      launcher<dim_y, Side::Right>(traversals, pf, bb);
      launcher<dim_z, Side::Left>(traversals, pf, bb);
      launcher<dim_z, Side::Right>(traversals, pf, bb);
    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(pre_bounce_back)
  {
    OperatorNodeFactory::instance()->register_factory( "pre_bounce_back", make_variant_operator<PreBounceBack>);
  }
}
