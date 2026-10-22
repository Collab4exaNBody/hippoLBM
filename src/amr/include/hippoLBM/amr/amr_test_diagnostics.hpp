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

#include <cmath>
#include <sstream>

#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/grid.hpp>
#include <hippoLBM/grid/scheme.hpp>
#include <hippoLBM/grid/stencil.hpp>

namespace hippoLBM {

/** @brief Prints rho (summed directly from f_, not a possibly-stale macro_variables cache)
 * along the x axis at the (local) y,z nearest to `along_point`, over `grid`'s own Real nodes.
 * Shared between AMRTwoLevel's internal diagnostics and the standalone amr_print_line_profile
 * operator (used for the uniform-fine reference run), so both print in the exact same format
 * and can be compared numerically (docs/AMR_PLAN.md: quantitative convergence validation). */
template <int Q>
inline void print_rho_line_profile(const char* tag, LBMGrid& grid, const FieldView<Q>& pf,
                                   const onika::math::Vec3d& along_point) {
  Box3D real = grid.template build_box<Area::Local, Traversal::Real>();

  const onika::math::Vec3d proj = (along_point - grid.origin_) / grid.dx_;
  const Point3D global_pt = {int(std::lround(proj.x)), int(std::lround(proj.y)), int(std::lround(proj.z))};
  const Point3D local_pt = grid.template convert<Area::Local>(global_pt);

  std::ostringstream oss;
  oss << "[" << tag << "] line profile (x=" << real.start(0) << ".." << real.end(0) << "): ";
  for (int x = real.start(0); x <= real.end(0); x++) {
    const int idx = grid(x, local_pt[1], local_pt[2]);
    double rho = 0.0;
    for (int iLB = 0; iLB < Q; iLB++) rho += pf(idx, iLB);
    oss << rho << " ";
  }
  onika::lout << oss.str() << std::endl;
}

/** @brief Prints u_x (computed directly from f_, lattice units) along the z axis at the
 * (local) x,y nearest to `along_point`, over `grid`'s own Real nodes. Used for the Couette
 * validation (docs/AMR_PLAN.md): the known analytical profile is linear in z,
 * u_x(z) = U_bottom + (U_top - U_bottom) * z / L, far simpler to check numerically than the
 * transient acoustic pulse used in the earlier (density-only) validation attempt. */
template <int Q>
inline void print_ux_profile_z(const char* tag, LBMGrid& grid, const FieldView<Q>& pf,
                               const onika::math::Vec3d& along_point) {
  Box3D real = grid.template build_box<Area::Local, Traversal::Real>();

  const onika::math::Vec3d proj = (along_point - grid.origin_) / grid.dx_;
  const Point3D global_pt = {int(std::lround(proj.x)), int(std::lround(proj.y)), int(std::lround(proj.z))};
  const Point3D local_pt = grid.template convert<Area::Local>(global_pt);

  std::ostringstream oss;
  oss << "[" << tag << "] ux(z) profile (z=" << real.start(2) << ".." << real.end(2) << "): ";
  for (int z = real.start(2); z <= real.end(2); z++) {
    const int idx = grid(local_pt[0], local_pt[1], z);
    double rho = 0.0, ux = 0.0;
    stencil::for_each<typename LBMScheme<Q>::Coefficients>([&]<typename coeff>(int iLB) {
      const double s = pf(idx, iLB);
      rho += s;
      ux += s * coeff::ex;
    });
    oss << (rho > 1.0e-14 ? ux / rho : 0.0) << " ";
  }
  onika::lout << oss.str() << std::endl;
}

}  // namespace hippoLBM
