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
using namespace std;

namespace hippoLBM
{
  //number of unknowns fi in 3DQ19
  using namespace onika::math;

  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

  template<int Q>
    struct LBMScheme {};

  template<> struct LBMScheme<19>
  {
    const vector_t<double> w = {1. / 3, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36};
    const vector_t<int> ex = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
    const vector_t<int> ey = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
    const vector_t<int> ez = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
    const vector_t<int> iopp = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
  };

  template<int Q>
    struct LBMFields 
    {
      uint64_t grid_size;
      LBMScheme<Q> scheme;

      // fields
      vector_t<double> f; // fi
      vector_t<double> m0; // densities
      vector_t<double> m1; // flux
      vector_t<int> obst; // obstacles

      // dunno
      vector_t<double> fi_x_0, fi_x_l, fi_y_0, fi_y_l, fi_z_0, fi_z_l;

      LBMFields() {}

      // accessors
      uint64_t size() { return grid_size; }
      FieldView<Q> distributions() { 
        return FieldView<Q>{onika::cuda::vector_data(f), grid_size}; 
      }

      double * densities() { return onika::cuda::vector_data(m0); }

      FieldView<3> flux()  { return FieldView<3>{onika::cuda::vector_data(m1), grid_size}; }

      int * obstacles() { return onika::cuda::vector_data(obst); }

      const int * obstacles() const { return onika::cuda::vector_data(obst); }

      const double * weights() { return onika::cuda::vector_data(scheme.w); }

      const int * iopp() { return onika::cuda::vector_data(scheme.iopp); }

      std::tuple<const int *, const int * , const int *> exyz() 
      {
        const int * ex = onika::cuda::vector_data(scheme.ex); 
        const int * ey = onika::cuda::vector_data(scheme.ey); 
        const int * ez = onika::cuda::vector_data(scheme.ez); 
        return {ex,ey,ez}; 
      }

      const int * ex() { return onika::cuda::vector_data(scheme.ex); }

      const int * ey() { return onika::cuda::vector_data(scheme.ey); }

      const int * ez() { return onika::cuda::vector_data(scheme.ez); }
    };
}
