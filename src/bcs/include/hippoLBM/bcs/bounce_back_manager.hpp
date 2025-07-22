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

#include <grid/enum.hpp>

namespace hippoLBM
{
  using namespace onika;
  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

  template<int dim, Side dir> 
    inline constexpr int helper_dim_idx() 
    {
      static_assert(dim < DIM_MAX);
      if constexpr (dir == Side::Right) return dim*2 + 1;
      else return dim * 2;
    }

  template<int Q>
    struct bounce_back_manager{};


  template<> struct bounce_back_manager<19>
  {
    static constexpr int  Un = 5; 
    // data[0] : x left
    // data[1] : x right
    // data[2] : y left
    // data[3] : y right
    // data[4] : z bottom
    // data[5] : z top
    std::array<vector_t<double>, 2 * DIM_MAX> _data;


    FieldView<Un> get_data(int i)
    {
      assert( onika::cuda::vector_size(_data[i]) % Un == 0 );
      int size = onika::cuda::vector_size(_data[i]) / Un;
      double * ptr = onika::cuda::vector_data(_data[i]);
      return FieldView<Un>{ptr, size}; 
    }

    template<int dim>
      int get_size(const onika::math::IJK lgs)
      {
        if constexpr(dim == DIMX) return lgs.j * lgs.k;
        if constexpr(dim == DIMY) return lgs.i * lgs.k;
        if constexpr(dim == DIMZ) return lgs.i * lgs.j;
      }

    template<int Dim, Side S>
      void resize_data(const onika::math::IJK& lgs)
      {
        const size_t size_dim = get_size<Dim>(lgs) * Un; 
        int i = helper_dim_idx<Dim,S>();
        auto& data = _data[i];
        if(size_dim != onika::cuda::vector_size(data))
        {
          data.resize(size_dim); 
        }
      }

    void resize_data(const std::vector<bool>& periodic, const onika::math::IJK& lgs /* local grid size*/, const onika::math::IJK& MPI_coord, const onika::math::IJK& MPI_grid_size)
    {
      if(periodic[DIMX] == false) // not periodic
      {
        if(MPI_coord.i == 0) resize_data<DIMX,Left>(lgs);
        if(MPI_coord.i == MPI_grid_size.i-1) resize_data<DIMX,Right>(lgs);
      }

      if(periodic[DIMY] == false) // not periodic
      {
        if(MPI_coord.j == 0) resize_data<DIMY,Left>(lgs);
        if(MPI_coord.j == MPI_grid_size.j-1) resize_data<DIMY,Right>(lgs);
      }

      if(periodic[DIMZ] == false) // not periodic
      {
        if(MPI_coord.k == 0) resize_data<DIMZ,Left>(lgs); 
        if(MPI_coord.k == MPI_grid_size.k-1) resize_data<DIMZ,Right>(lgs);
      }
    }
  };
}
