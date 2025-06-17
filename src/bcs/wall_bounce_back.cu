#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <grid/make_variant_operator.hpp>
#include <onika/math/basic_types.h>
#include <hippoLBM/grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/traversal_lbm.hpp>
#include <hippoLBM/bcs/bounce_back.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::cuda;

  template<int Q>
    class WallBounceBack : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( LBMDomain<Q>, lbm_domain, INPUT, REQUIRED);

      inline std::string documentation() const override final
      {
        return R"EOF(  The WallBounceBack class is described as part of the Lattice Boltzmann Method (LBM) implementation, specifically the wall bounce back steps.)EOF";
      }


      inline void execute () override final
      {
        auto& data = *LBMFieds;
        auto& domain = *lbm_domain;
        grid<3>& Grid = domain.m_grid;

        // get fields
        const int* const pobst = data.obstacles();
        FieldView<Q> pf = data.distributions();
        auto [pex, pey, pez] = data.exyz();

        // define functors
        wall_bounce_back<Q> func = {Grid, pobst, pf, pex, pey, pez};

        // run kernel
        box<3> extend = Grid.build_box<Area::Local, Traversal::Extend>();
        onika::parallel::ParallelExecutionSpace<3> parallel_range = set(extend);
        parallel_for(parallel_range, func, parallel_execution_context("wall_bounce_back"));
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(wall_bounce_back)
  {
    OperatorNodeFactory::instance()->register_factory( "wall_bounce_back", make_variant_operator<WallBounceBack>);
  }
}

