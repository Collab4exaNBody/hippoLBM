#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <grid/make_variant_operator.hpp>
#include <onika/math/basic_types.h>
#include <grid/lbm_domain.hpp>
#include <grid/comm.hpp>
#include <grid/enum.hpp>
#include <grid/lbm_fields.hpp>
#include <grid/parallel_for_core.cu>
#include <grid/traversal_lbm.hpp>
#include <hippoLBM/grid/set_distribution.hpp>
#include <grid/update_ghost.hpp>

namespace hippoLBM
{
  using namespace onika;
  using namespace scg;

  template<int Q>
    class SetDistributionsLBM : public OperatorNode
  {
    public:
      ADD_SLOT( lbm_domain<Q>, LBMDomain, INPUT, REQUIRED);
      ADD_SLOT( lbm_fields<Q>, LBMFieds, INPUT_OUTPUT);
      ADD_SLOT( traversal_lbm, Traversals, INPUT, REQUIRED);
      ADD_SLOT( AABB, bounds, INPUT, OPTIONAL, DocString{"Domain's bounds"});
      ADD_SLOT( double, value, INPUT, double(1) );
      ADD_SLOT( bool, do_update, INPUT, false);

      inline void execute () override final
      {
        auto& data = *LBMFieds;
        auto& traversals = *Traversals;
        lbm_domain<Q>& domain = *LBMDomain;

        FieldView pf = data.distributions();
        const double * const pw = data.weights();

        // define kernel
        init_distributions<Q> func = {*value};

        // capture the parallel execution context
        auto par_exec_ctx = [this] (const char* exec_name)
        { 
          return this->parallel_execution_context(exec_name);
        };

        if(bounds.has_value())
        {
          grid<3>& Grid = domain.m_grid;

          auto& bound = *bounds;
          Vec3d min = bound.bmin;
          Vec3d max = bound.bmax;
          double Dx = Grid.dx;
          point<3> _min = {int(min.x/Dx), int(min.y/Dx), int(min.z/Dx)};
          point<3> _max = {int(max.x/Dx), int(max.y/Dx), int(max.z/Dx)};

          box<3> global_wall_box = {_min, _max};
          global_wall_box.print();

          auto [is_inside_subdomain, wall_box] = Grid.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_wall_box);
          wall_box.print();
          if( !is_inside_subdomain ) return;

          for(int z = wall_box.start(2) ; z <= wall_box.end(2) ; z++)
            for(int y = wall_box.start(1) ; y <= wall_box.end(1) ; y++)
              for(int x = wall_box.start(0) ; x <= wall_box.end(0) ; x++)
              {
                const int idx = Grid(x,y,z);
                func(idx, pf, pw);
              }

        }
        else  // all domain
        { 
          if( *do_update )
          {
            auto [ptr, size] = traversals.get_data<Traversal::Real>();
            parallel_for_id(ptr, size, func, parallel_execution_context(), pf, pw);
            update_ghost(domain, pf, par_exec_ctx);
          }
          else
          {
            auto [ptr, size] = traversals.get_data<Traversal::All>();
            parallel_for_id(ptr, size, func, parallel_execution_context(), pf, pw);
          }
        }
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_distributions)
  {
    OperatorNodeFactory::instance()->register_factory( "set_distribution", make_variant_operator<SetDistributionsLBM>);
  }
}
