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

#include <onika/math/basic_types.h>

#include <hippoLBM/compute/reduce.hpp>
#include <hippoLBM/grid/fields.hpp>

namespace hippoLBM {
/** @brief Structure to hold simulation statistics */
struct SimulationStatistics {
  double sum_density_ = 0.0;
  double min_velocity_norm_ = std::numeric_limits<double>::max();
  double max_velocity_norm_ = std::numeric_limits<double>::lowest();
  void display() {
    lout << "sum_density: " << sum_density_ << ", min_velocity_norm: " << min_velocity_norm_
         << ", max_velocity_norm: " << max_velocity_norm_ << std::endl;
  }
};

/** @brief Function object to compute simulation state */
struct ComputeSimulationStateFunc {
  double* const density_;  // Pointer to the density field.
  FieldView<3> velocity_;  // View of the velocity field.

  /** @brief Operator to compute simulation state
   * @param local Local simulation statistics to be updated.
   * @param idx Index of the current lattice node.
   * @param reduce_thread_local_t Tag to indicate thread-local reduction.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline void operator()(SimulationStatistics& local, const uint64_t idx, reduce_thread_local_t = {}) const {
    local.sum_density_ += density_[idx];
    onika::math::Vec3d v = velocity_.get(idx);
    double v_norm = onika::math::norm(v);
    local.min_velocity_norm_ = std::min(local.min_velocity_norm_, v_norm);
    local.max_velocity_norm_ = std::max(local.max_velocity_norm_, v_norm);
    // std::cout << "max_velocity_norm " << local.max_velocity_norm_ << " v_norm " << v_norm << std::endl;
  }

  /** @brief Operator to reduce simulation statistics across threads
   * @param global Global simulation statistics to be updated.
   * @param local Local simulation statistics to be reduced.
   * @param reduce_thread_block_t Tag to indicate thread-block reduction.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(SimulationStatistics& global, SimulationStatistics& local,
                                                reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_ADD(global.sum_density_, local.sum_density_);
    ONIKA_CU_ATOMIC_MIN(global.min_velocity_norm_, local.min_velocity_norm_);
    ONIKA_CU_ATOMIC_MAX(global.max_velocity_norm_, local.max_velocity_norm_);
  }

  /** @brief Operator to reduce simulation statistics globally
   * @param global Global simulation statistics to be updated.
   * @param local Local simulation statistics to be reduced.
   * @param reduce_global_t Tag to indicate global reduction.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(SimulationStatistics& global, SimulationStatistics& local,
                                                reduce_global_t) const {
    ONIKA_CU_ATOMIC_ADD(global.sum_density_, local.sum_density_);
    ONIKA_CU_ATOMIC_MIN(global.min_velocity_norm_, local.min_velocity_norm_);
    ONIKA_CU_ATOMIC_MAX(global.max_velocity_norm_, local.max_velocity_norm_);
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<hippoLBM::ComputeSimulationStateFunc> {
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
