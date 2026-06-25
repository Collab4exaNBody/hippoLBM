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

#include <onika/math/basic_types.h>

namespace hippoLBM {
/** @brief A view of a field in the grid.
 * @tparam Components The number of components in the field.
 */
template <int Components>
struct FieldView {
  double* const data_ = nullptr;  ///< Pointer to the field data.
  uint64_t num_elements_ = 0;     ///< The number of elements in the field.

  /** @brief Reset the field values at the specified index.
   * @param idx The index of the element to reset.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline void reset(size_t idx) const {
    for (size_t component_index = 0; component_index < Components; component_index++) {
      this->operator()(idx, component_index) = 0;
    }
  }

 private:
  /** @brief Access the field value at the specified index and component.
   * @param idx The index of the element.
   * @param component_index The index of the component.
   * @return Reference to the field value.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline double& access(size_t idx, size_t component_index) {
#ifdef WFAOS
    // Access the field value based on the memory layout defined by WFAOS
    return data_[idx * Components + component_index];
#else

    return data_[num_elements_ * component_index + idx];
#endif
  }

  /** @brief Access the field value at the specified index and component (const version).
   * @param idx The index of the element.
   * @param component_index The index of the component.
   * @return Reference to the field value.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline double& access(size_t idx, size_t component_index) const {
#ifdef WFAOS
    // Access the field value based on the memory layout defined by WFAOS
    return data_[idx * Components + component_index];
#else
    // Access the field value based on the memory layout defined by AOSOAOS
    return data_[num_elements_ * component_index + idx];
#endif
  }

 public:
  /** @brief Access the field value at the specified index and component.
   * @param idx The index of the element.
   * @param component_index The index of the component.
   * @return Reference to the field value.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline double& operator()(size_t idx, size_t component_index) {
    assert(idx < num_elements_);
    assert(component_index < Components);
    return access(idx, component_index);
  }

  /** @brief Access the field value at the specified index and component (const version).
   * @param idx The index of the element.
   * @param component_index The index of the component.
   * @return Reference to the field value.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline double& operator()(size_t idx, size_t component_index) const {
    assert(idx < num_elements_);
    assert(component_index < Components);
    return access(idx, component_index);
  }

  /** @brief Assign the values from another field view.
   * @param fv The field view to copy from.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline void operator=(FieldView<Components>& fv) {
    this->data_ = fv.data_;
    this->num_elements_ = fv.num_elements_;
  }

  /** @brief Get the field value at the specified index as a 3D vector.
   * @param idx The index of the element.
   * @return The 3D vector representing the field value.
   */
  ONIKA_HOST_DEVICE_FUNC
  onika::math::Vec3d get(size_t idx) const
    requires(Components == 3)
  {
    onika::math::Vec3d res;
    res.x = access(idx, 0);
    res.y = access(idx, 1);
    res.z = access(idx, 2);
    return res;
  }

  /** @brief Set the field value at the specified index to a 3D vector.
   * @param idx The index of the element.
   * @param in The 3D vector representing the field value.
   */
  ONIKA_HOST_DEVICE_FUNC
  void set(size_t idx, onika::math::Vec3d& in)
    requires(Components == 3)
  {
    access(idx, 0) = in.x;
    access(idx, 1) = in.y;
    access(idx, 2) = in.z;
  }

  /** @brief Set the field value at the specified index to a 3D vector (const version).
   * @param idx The index of the element.
   * @param in The 3D vector representing the field value.
   */
  ONIKA_HOST_DEVICE_FUNC
  void set(size_t idx, const onika::math::Vec3d& in) const
    requires(Components == 3)
  {
    access(idx, 0) = in.x;
    access(idx, 1) = in.y;
    access(idx, 2) = in.z;
  }
};

/** @brief Copy field values from one view to another.
 * @param dest_data The destination field view.
 * @param dest_idx The index of the destination element.
 * @param from_data The source field view.
 * @param from_idx The index of the source element.
 * @param size The number of elements to copy.
 */
template <int Components>
ONIKA_HOST_DEVICE_FUNC inline void copyTo(const FieldView<Components>& dest_data, int dest_idx,
                                          const FieldView<Components>& from_data, int from_idx, int size) {
#ifdef WFAOS
  // case 1
  double* from = &from_data(from_idx, 0);
  double* dest = &dest_data(dest_idx, 0);
  int nb_byte = size * Components * sizeof(double);
  std::memcpy(dest, from, nb_byte);
#else
  // case 2
  int nb_byte = size * sizeof(double);
  for (size_t component_index = 0; component_index < Components; component_index++) {
    std::memcpy(&dest_data(dest_idx, component_index), &from_data(from_idx, component_index), nb_byte);
  }
#endif
}
}  // namespace hippoLBM
