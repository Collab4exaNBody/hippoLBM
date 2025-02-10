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
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/comm.hpp>
#include <grid_lbm/enum.hpp>
#include <grid_lbm/grid_data_lbm.hpp>


namespace hipoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class ResetGridDataLBM : public OperatorNode
	{
		public:
			ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);
			// Un = 5
			ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT);

			inline void execute () override final
			{
				constexpr Area L = Area::Local;
				constexpr Traversal Tr = Traversal::All;
				grid_data_lbm<Q>& grid_data = *GridDataQ;
				domain_lbm<Q>& domain = *DomainQ;
				grid<3>& Grid = domain.m_grid;
				box<3>& Box = domain.m_box;

				// compute sizes
        constexpr int Un = 5;
				auto bx = Grid.build_box<L, Tr>();
				int size_XYU = bx.get_length(0) * bx.get_length(1) * Un;
				int size_YZU = bx.get_length(1) * bx.get_length(2) * Un;
				int size_XZU = bx.get_length(0) * bx.get_length(2) * Un;
				const int np = Box.number_of_points();

				if(grid_data.obst.size() != np)
				{
					grid_data.f.resize(np*Q);
					grid_data.obst.resize(np);
					grid_data.m0.resize(np);
					grid_data.m1.resize(np);
					grid_data.fi_x_0.resize(size_YZU);
					grid_data.fi_x_l.resize(size_YZU);
					grid_data.fi_y_0.resize(size_XZU);
					grid_data.fi_y_l.resize(size_XZU);
					grid_data.fi_z_0.resize(size_XYU);
					grid_data.fi_z_l.resize(size_XYU);
				}
			}
	};

	using ResetGridDataLBM3D19Q = ResetGridDataLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(parallel_for_benchmark)
	{
		OperatorNodeFactory::instance()->register_factory( "reset_grid_data", make_compatible_operator<ResetGridDataLBM3D19Q>);
	}
}

