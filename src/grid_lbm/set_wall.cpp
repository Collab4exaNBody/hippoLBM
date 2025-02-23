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
#include <grid_lbm/traversal_lbm.hpp>
#include <grid_lbm/grid_data_lbm.hpp>

namespace hippoLBM
{
	using namespace onika;
	using namespace scg;

	template<int Q>
		class SetWall : public OperatorNode
	{
		public:
      ADD_SLOT( domain_lbm<Q>, DomainQ, INPUT, REQUIRED);
      ADD_SLOT( grid_data_lbm<Q>, GridDataQ, INPUT_OUTPUT);
			ADD_SLOT( AABB, bounds, INPUT_OUTPUT, REQUIRED, DocString{"Domain's bounds"});
			ADD_SLOT( double, dx, INPUT, REQUIRED, DocString{"Space step"});
			inline void execute () override final
			{
        grid_data_lbm<Q>& grid_data = *GridDataQ;
        domain_lbm<Q>& domain = *DomainQ;
        grid<3>& Grid = domain.m_grid;
       
        auto& bound = *bounds;
        Vec3d min = bound.bmin;
        Vec3d max = bound.bmax;
        double Dx = *dx;
        point<3> _min = {int(min.x/Dx), int(min.y/Dx), int(min.z/Dx)};
        point<3> _max = {int(max.x/Dx), int(max.y/Dx), int(max.z/Dx)};

        box<3> global_wall_box = {_min, _max};
        global_wall_box.print();

        auto [is_inside_subdomain, wall_box] = Grid.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_wall_box);
        wall_box.print();
        if( !is_inside_subdomain ) return;
 

				int * const obst = grid_data.obstacles();
				for(int z = wall_box.start(2) ; z <= wall_box.end(2) ; z++)
					for(int y = wall_box.start(1) ; y <= wall_box.end(1) ; y++)
						for(int x = wall_box.start(0) ; x <= wall_box.end(0) ; x++)
						{
              const int idx = Grid(x,y,z);
              obst[idx] = 0;
						}
			}
	};

	using SetWall3D19Q = SetWall<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(set_wall)
	{
		OperatorNodeFactory::instance()->register_factory( "set_wall", make_compatible_operator<SetWall3D19Q>);
	}
}

