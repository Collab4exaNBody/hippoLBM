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
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/comm.hpp>
#include <grid_lbm/enum.hpp>
#include <grid_lbm/grid_data_lbm.hpp>
#include <grid_lbm/parallel_for_core.cu>
#include <grid_lbm/parallel_for_box.hpp>
#include <grid_lbm/traversal_lbm.hpp>
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
      ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT, REQUIRED, DocString{"Grid data for the LBM simulation, including distribution functions and macroscopic fields."});
      ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);

      inline std::string documentation() const override final
      {
        return R"EOF(  The WallBounceBack class is described as part of the Lattice Boltzmann Method (LBM) implementation, specifically the wall bounce back steps.)EOF";
      }


			inline void execute () override final
			{
				auto& data = *GridDataQ;
				auto& domain = *DomainQ;
				grid<3>& Grid = domain.m_grid;

				// define functors
				wall_bounce_back<Q> func = {Grid};

				// get fields
				const int* const pobst = data.obstacles();
				WrapperF<Q> pf = data.distributions();
				auto [pex, pey, pez] = data.exyz();

				// run kernel
				box<3> extend = Grid.build_box<Area::Local, Traversal::Extend>();
#ifdef ONIKA_CUDA_VERSION
				cuda_parallel_for_box(extend, func, pobst, pf, pex, pey, pez);
#else
				parallel_for_box(extend, func, pobst, pf, pex, pey, pez);
#endif
			}
	};

	using WallBounceBack3D19Q = WallBounceBack<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT()
	{
		OperatorNodeFactory::instance()->register_factory( "wall_bounce_back", make_compatible_operator<WallBounceBack3D19Q>);
	}
}

