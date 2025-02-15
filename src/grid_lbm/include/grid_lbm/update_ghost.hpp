#pragma once

#include<grid_lbm/wrapper_f.hpp>

namespace hipoLBM
{
  template<int Q>
    inline void update_ghost(domain_lbm<Q>& domain, double * const data)
    {
      grid<3>& Grid = domain.m_grid;
      constexpr Area L = Area::Local;
      constexpr Traversal Tr = Traversal::All;
      box<3> bx = Grid.build_box<L,Tr>();
      auto& manager = domain.m_ghost_manager; 
      //manager.debug_print_comm();
      manager.resize_request();
      manager.do_recv();
      manager.do_pack_send(data, bx);
      manager.wait_all();
      manager.do_unpack(data, bx);
    }
  template<int Q>
    inline void update_ghost(domain_lbm<Q>& domain, WrapperF& data)
    {
      grid<3>& Grid = domain.m_grid;
      constexpr Area L = Area::Local;
      constexpr Traversal Tr = Traversal::All;
      box<3> bx = Grid.build_box<L,Tr>();
      auto& manager = domain.m_ghost_manager; 
      //manager.debug_print_comm();
      manager.resize_request();
      manager.do_recv();
      manager.do_pack_send(data, bx);
      manager.wait_all();
      manager.do_unpack(data, bx);
    }
}
