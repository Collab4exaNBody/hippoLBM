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

#include <onika/cuda/stl_adaptors.h>

#include <hippoLBM/grid/field_view.hpp>
#include <hippoLBM/grid/scheme.hpp>
#include <hippoLBM/grid/stencil.hpp>
using namespace std;

namespace hippoLBM {
template <int Q>
struct LBMFields {
  template <typename T>
  using vector_t = onika::memory::CudaMMVector<T>;
  uint64_t grid_size_;
  // fields
  vector_t<double> f_;   // fi
  vector_t<double> m0_;  // densities
  vector_t<double> m1_;  // flux
  vector_t<int> obst_;   // obstacles

  // dunno
  vector_t<double> fi_x_0_, fi_x_l_, fi_y_0_, fi_y_l_, fi_z_0_, fi_z_l_;

  LBMFields() {}

  // accessors
  uint64_t size() { return grid_size_; }
  FieldView<Q> distributions() { return FieldView<Q>{onika::cuda::vector_data(f_), grid_size_}; }
  double* densities() { return onika::cuda::vector_data(m0_); }
  FieldView<3> flux() { return FieldView<3>{onika::cuda::vector_data(m1_), grid_size_}; }
  int* obstacles() { return onika::cuda::vector_data(obst_); }
  const int* obstacles() const { return onika::cuda::vector_data(obst_); }
};
}  // namespace hippoLBM
