#pragma once

#include<grid/field_view.hpp>

namespace hippoLBM
{
  template<int Q, int Components, typename ParExecCtxFunc>
    inline void update_ghost(LBMDomain<Q>& domain, FieldView<Components>& data, ParExecCtxFunc& par_exec_ctx_func)
    {
      LBMGrid& Grid = domain.m_grid;
      constexpr Area L = Area::Local;
      constexpr Traversal Tr = Traversal::All;
      box<3> bx = Grid.build_box<L,Tr>();
      auto& manager = domain.m_ghost_manager; 
      //manager.debug_print_comm();
      manager.resize_request();
      manager.do_recv();
      manager.do_pack_send(data, bx, par_exec_ctx_func);
      manager.wait_all();
      manager.do_unpack(data, bx, par_exec_ctx_func);
    }
}
