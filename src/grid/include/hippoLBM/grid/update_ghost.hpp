/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */


#pragma once

#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>
#include <hippoLBM/grid/enum.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/traversal_lbm.hpp>
#include <hippoLBM/grid/parallel_for_core.cu>

namespace hippoLBM
{
	/**
	 * @brief Updates ghost layers in a lattice Boltzmann domain.
	 *
	 * This function handles the communication and synchronization of ghost 
	 * layers (halo regions) for a given field across the computational grid. 
	 * It ensures consistency between neighboring subdomains when running 
	 * in parallel.
	 *
	 * @tparam Q Number of discrete velocity directions in the LBM scheme.
	 * @tparam Components Number of components stored in the field (e.g., scalar or vector field).
	 * @tparam ParExecCtxFunc Type of the parallel execution context function.
	 *
	 * @param domain LBM domain that contains the grid and ghost manager.
	 * @param data   Field view containing the data to be synchronized.
	 * @param par_exec_ctx_func Parallel execution context function (e.g., kernel execution wrapper).
	 *
	 * @note The function internally:
	 * - Builds the local computational box.
	 * - Resizes ghost layer requests.
	 * - Performs non-blocking receives and packing of data.
	 * - Waits for all communication to complete.
	 * - Unpacks received ghost data back into the field.
	 *
	 * This communication pattern is essential for domain decomposition 
	 * in distributed-memory parallel simulations.
	 */
	template<int Q, int Components, typename ParExecCtxFunc>
		inline void update_ghost(LBMDomain<Q>& domain, FieldView<Components>& data, ParExecCtxFunc& par_exec_ctx_func)
		{
			LBMGrid& Grid = domain.m_grid;
			constexpr Area L = Area::Local;
			constexpr Traversal Tr = Traversal::All;
			Box3D bx = Grid.build_box<L,Tr>();
			auto& manager = domain.m_ghost_manager; 
			//manager.debug_print_comm();
			manager.resize_request();
			manager.do_recv();
			manager.do_pack_send(data, bx, par_exec_ctx_func);
			manager.wait_all();
			manager.do_unpack(data, bx, par_exec_ctx_func);
		}
}
