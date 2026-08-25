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

#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/grid.hpp>

namespace hippoLBM {
/**
 * @brief A functor returning the external force term, uniform over the whole grid.
 */
struct FextConstantFunc {
  const onika::math::Vec3d Fext_;

  ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator()(int idx, const onika::math::Vec3d& u_local) const {
    return Fext_;
  }
};

/**
 * @brief A functor returning a relaxation force that drives the local velocity towards a target linear
 * profile along k (global z index), going from U_inf_ (k = 0) to U_inf_ + dU_ * (domain_size_z - 1)
 * (k = domain_size_z - 1): Fext = (u_target(k) - u_local) / tau_relax_.
 */
struct FextCouetteFunc {
  LBMGrid g_;
  const onika::math::Vec3d U_inf_;
  const onika::math::Vec3d dU_;  // (U_sup_ - U_inf_) / (domain_size[DIMZ] - 1)
  const double inv_tau_relax_;   // 1 / tau_relax_

  ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator()(int idx, const onika::math::Vec3d& u_local) const {
    const Point3D pt = g_(idx);
    const double k = pt[2] + g_.offset_[2];
    const onika::math::Vec3d u_target = U_inf_ + dU_ * k;
    return (u_target - u_local) * inv_tau_relax_;
  }
};
}  // namespace hippoLBM
