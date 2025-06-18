#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <onika/math/basic_types.h>
#include <hippoLBM/grid/domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class DefineLBMFields : public OperatorNode
  {
    public:
      ADD_SLOT( LBMDomain<Q>, domain, INPUT, REQUIRED);
      ADD_SLOT( LBMFields<Q>, fields, INPUT_OUTPUT);

      inline void execute () override final
      {
        constexpr Area L = Area::Local;
        constexpr Traversal Tr = Traversal::All;
        LBMFields<Q>& grid_data = *fields;
        LBMGrid& Grid = domain->m_grid;
        box<3>& Box = domain->m_box;

        // compute sizes
        constexpr int Un = 5;
        auto bx = Grid.build_box<L, Tr>();
        int size_XYU = bx.get_length(0) * bx.get_length(1) * Un;
        int size_YZU = bx.get_length(1) * bx.get_length(2) * Un;
        int size_XZU = bx.get_length(0) * bx.get_length(2) * Un;
        const size_t np = Box.number_of_points();

        grid_data.grid_size = np;

        if(grid_data.obst.size() != np)
        {
          grid_data.f.resize(np*Q);
          grid_data.obst.resize(np);
          grid_data.m0.resize(np);
          grid_data.m1.resize(np*3);
          grid_data.fi_x_0.resize(size_YZU);
          grid_data.fi_x_l.resize(size_YZU);
          grid_data.fi_y_0.resize(size_XZU);
          grid_data.fi_y_l.resize(size_XZU);
          grid_data.fi_z_0.resize(size_XYU);
          grid_data.fi_z_l.resize(size_XYU);
        }
      }
  };



  // === register factories ===  
  ONIKA_AUTORUN_INIT(define_grid)
  {
    OperatorNodeFactory::instance()->register_factory( "define_grid_3dq19", make_compatible_operator<DefineLBMFields<19>>);
    //OperatorNodeFactory::instance()->register_factory( "define_grid_3dq15", make_compatible_operator<DefineLBMFields<Q><15>>);
  }
}

