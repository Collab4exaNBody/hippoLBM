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

#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_yaml.h>

#include <hippoLBM/compute/parallel_for_core.hpp>

#pragma once

namespace hippoLBM {
/** @brief A null function writer that returns the input data unchanged. */
struct NullFuncWriter {
  template <typename T>
  inline T& operator()(const int idx, T& data) const {
    return data;
  }
  inline onika::math::Vec3d operator()(const int idx, const FieldView<3>& data) const {
    return onika::math::Vec3d{data(idx, 0), data(idx, 1), data(idx, 2)};
  }
};

/** @brief A writer for computing velocity fields. */
struct UWriter {
  const int* const obst;       ///< Pointer to the obstacle field.
  const double ratio_dx_dtLB;  ///< The ratio of grid spacing to LBM time step.
  inline onika::math::Vec3d operator()(const int idx, const FieldView<3>& m1) const {
    if (obst[idx] == FLUIDE_) {
      onika::math::Vec3d _m1 = {m1(idx, 0), m1(idx, 1), m1(idx, 2)};
      return ratio_dx_dtLB * _m1;
    }
    return onika::math::Vec3d{0, 0, 0};
  }
};

/** @brief A writer for computing pressure fields. */
struct PressionWriter {
  const int* const obst;               ///< Pointer to the obstacle field.
  const double c_c_avg_rho_div_three;  ///< A constant factor for pressure calculation.
  inline double operator()(const int idx, const double& m0) const {
    if (obst[idx] == FLUIDE_) {
      return c_c_avg_rho_div_three * (m0 - 1);
    }
    return 0;
  }
};

/** @brief A writer for writing data to a file. */
template <typename Func>
struct write_file {
  Func func;  ///< The function to compute the data to be written.
  template <typename T>
  inline void operator()(int idx, std::stringstream& output, T* const ptr) const {
    T tmp = ptr[idx];
    tmp = func(idx, tmp);
    output << (T)tmp << " ";
  }
};

/** @brief A writer for writing data to a file. */
struct WriterExternalData {
  int num_components;     // The number of components in the data to be written.
  uint64_t num_elements;  // The number of elements in the data to be written.
  inline void operator()(int idx, std::stringstream& output, double* const input_data) const {
    for (int i = 0; i < num_components; i++) {
      double tmp;
#ifdef WFAOS
      tmp = input_data[idx * num_components + i];
#else
      tmp = input_data[num_elements * i + idx];
#endif
      output << (float)tmp << " ";
    }
  }
};

template <>
struct ForAllGridTraits<WriterExternalData> {
  static constexpr bool UsedIJK = false;
};

/** @brief A writer for writing 3D vector data to a file. */
template <typename Func>
struct WriteVec3d {
  Func func;  // The function to compute the 3D vector data to be written.
  Box3D b;    // The computational box representing the local domain, including ghost layers.
  /** @brief Writes 3D vector data to a file.
   * @param x The x-coordinate of the point being written.
   * @param y The y-coordinate of the point being written.
   * @param z The z-coordinate of the point being written.
   * @param output The stringstream to which the data will be written.
   * @param ptr The pointer to the input data from which the 3D vector will
   */
  inline void operator()(const int x, const int y, const int z, std::stringstream& output,
                         onika::math::Vec3d* const ptr) const {
    const int idx = b(x, y, z);
    onika::math::Vec3d tmp = func(idx, ptr[idx]);
    output << (float)tmp.x << " " << (float)tmp.y << " " << (float)tmp.z << " ";
  }

  /** @brief Writes 3D vector data to a file.
   * @param x The x-coordinate of the point being written.
   * @param y The y-coordinate of the point being written.
   * @param z The z-coordinate of the point being written.
   * @param output The stringstream to which the data will be written.
   * @param WF The field view from which the 3D vector will be computed.
   */
  inline void operator()(const int x, const int y, const int z, std::stringstream& output,
                         const FieldView<3>& WF) const {
    const int idx = b(x, y, z);
    onika::math::Vec3d tmp = func(idx, WF);
    output << (float)tmp.x << " " << (float)tmp.y << " " << (float)tmp.z << " ";
  }
};
}  // namespace hippoLBM
