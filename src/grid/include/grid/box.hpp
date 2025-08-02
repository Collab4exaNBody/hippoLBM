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

#include <grid/point.hpp>

namespace hippoLBM
{
  /**
   * @brief A geometric box in a multi-dimensional space.
   *
   * This template struct represents a geometric box in a multi-dimensional space.
   * It is defined by two points, `inf` and `sup`, which represent the lower-left and
   * upper-right corners of the box in 2D, respectively. The struct also provides methods
   * for computing the box's dimensions and number of points within it.
   *
   * @tparam DIM  The dimensionality of the box (number of spatial dimensions).
   *
   * @note The `DIM` template parameter must be greater than 0.
   *
   * @param inf  The lower-left corner of the box.
   * @param sup  The upper-right corner of the box.
   */
  template<int DIM>
    struct box
    {
      static_assert(DIM > 0);
      point<DIM> inf; /**< The lower-left corner of the box. */
      point<DIM> sup; /**< The upper-right corner of the box. */

      /**
       * @brief Get the length of the box along a specified dimension.
       *
       * @param dim  The dimension for which to compute the length.
       * @return     The length of the box along the specified dimension.
       */
      ONIKA_HOST_DEVICE_FUNC inline int get_length(int dim) { 
        assert(sup[dim] >= inf[dim]) ;
        return (sup[dim] - inf[dim]) + 1; 
      }

      /**
       * @brief Get the length of the box along a specified dimension.
       *
       * @param dim  The dimension for which to compute the length.
       * @return     The length of the box along the specified dimension.
       */
      ONIKA_HOST_DEVICE_FUNC inline int get_length(int dim) const { 
        assert(sup[dim] >= inf[dim]) ;
        return (sup[dim] - inf[dim]) + 1; 
      }

      ONIKA_HOST_DEVICE_FUNC inline int start(int dim) const
      {
        return inf[dim];
      }

      ONIKA_HOST_DEVICE_FUNC inline int end(int dim) const
      {
        return sup[dim];
      }

      /**
       * @brief Calculate the total number of points within the box.
       *
       * @return The total number of points within the box.
       */
      ONIKA_HOST_DEVICE_FUNC inline int number_of_points() const {
        int res = 1;
        for(int dim = 0 ; dim < DIM ; dim++) res *= (this->get_length(dim)); // sup is included (+1)
        return res;
      }

      ONIKA_HOST_DEVICE_FUNC inline bool contains(point<3>& p)
      {
        for( int dim = 0 ; dim < DIM ; dim++ )
        {
          if( (p[dim] < inf[dim]) || (p[dim] > sup[dim]) ) return false; 
        }
        return true;
      }

      ONIKA_HOST_DEVICE_FUNC inline bool contains(point<3>&& p)
      {
        for( int dim = 0 ; dim < DIM ; dim++ )
        {
          if( (p[dim] < inf[dim]) || (p[dim] > sup[dim]) ) return false; 
        }
        return true;
      }

      /**
       * @brief Print the box's lower-left and upper-right corners.
       */
      void print() 
      {
        onika::lout << " inf:"; 
        inf.print();
        onika::lout << " sup:";
        sup.print();
      }

      /**
       * @brief Compute the index of a point within the box using Cartesian coordinates.
       *
       * @param x  The x-coordinate of the point.
       * @param y  The y-coordinate of the point.
       * @param z  The z-coordinate of the point.
       * @return   The index of the point within the box.
       */
      ONIKA_HOST_DEVICE_FUNC inline int operator()(const int x, const int y, const int z)
      {
        int idx = z * (this->get_length(1)) + y;
        idx *= this->get_length(0);
        idx += x;
        return idx;
      }

      /**
       * @brief Compute the index of a point within the box using Cartesian coordinates.
       *
       * @param x  The x-coordinate of the point.
       * @param y  The y-coordinate of the point.
       * @param z  The z-coordinate of the point.
       * @return   The index of the point within the box.
       */
      ONIKA_HOST_DEVICE_FUNC inline int operator()(const int x, const int y, const int z) const
      {
        int idx = z * (this->get_length(1)) + y;
        idx *= this->get_length(0);
        idx += x;
        return idx;
      }

      ONIKA_HOST_DEVICE_FUNC inline std::tuple<int,int,int> operator()(int idx) const
      {
        int size_y = this->get_length(1);
        int size_x = this->get_length(0);
        int size_xy = size_y * size_x;
        int z = idx / size_xy;
        idx = idx - z * size_xy;
        int y = idx / size_x;
        int x = idx % size_x;
        return {x,y,z};
      }

      /**
       * @brief compute the length of the box along a specified dimension.
       *
       * @param dim  The dimension for which to compute the length.
       * @return     The length of the box along the specified dimension.
       */
      ONIKA_HOST_DEVICE_FUNC inline int operator[](int dim) { return get_length(dim); }

      /**
       * @brief compute the length of the box along a specified dimension.
       *
       * @param dim  The dimension for which to compute the length.
       * @return     The length of the box along the specified dimension.
       */
      ONIKA_HOST_DEVICE_FUNC inline int operator[](int dim) const { return get_length(dim); }

      /** 
       * @brief accessor to the `inf` member
       */
      ONIKA_HOST_DEVICE_FUNC inline point<DIM>& lower() { return inf;} 
      /** 
       * @brief accessor to the `inf` member
       */
      ONIKA_HOST_DEVICE_FUNC inline point<DIM>& lower() const { return inf;} 
      /** 
       * @brief accessor to the `sup` member
       */
      ONIKA_HOST_DEVICE_FUNC inline point<DIM>& upper() { return sup;} 
      /** 
       * @brief accessor to the `sup` member
       */
      ONIKA_HOST_DEVICE_FUNC inline point<DIM>& upper() const { return sup;} 

    };

  /**
   * @brief Compute the index of a point within a multi-dimensional box using Cartesian coordinates.
   *
   * This template function calculates the index of a point within a multi-dimensional box
   * using Cartesian coordinates (x, y, z). It takes a `box` instance as input and delegates
   * the computation to the `box`'s operator() method.
   *
   * @tparam DIM  The dimensionality of the box (number of spatial dimensions).
   *
   * @param b    A reference to a `box` instance representing the multi-dimensional box.
   * @param x    The x-coordinate of the point.
   * @param y    The y-coordinate of the point.
   * @param z    The z-coordinate of the point.
   * @return     The index of the point within the box.
   */
  template<int DIM>
    ONIKA_HOST_DEVICE_FUNC inline int compute_idx(const box<DIM>& b, const int x, const int y, const int z)
    {
      return b(x,y,z);
    }


  template<int DIM>
    ONIKA_HOST_DEVICE_FUNC inline bool intersect(box<DIM>& a, box<DIM>& b)
    {
      // Vérifier les conditions de non-intersection sur l'axe x
      for(int dim = 0 ; dim < DIM ; dim++)
      {
        if (a.sup[dim] < b.inf[dim] || b.sup[dim] < a.inf[dim])
        {
          return false;
        }
      }
      // Si aucune des conditions de non-intersection n'est remplie, les boîtes s'intersectent
      return true;
    }

  ONIKA_HOST_DEVICE_FUNC inline onika::parallel::ParallelExecutionSpace<3> set(box<3>& bx)
  {
    return onika::parallel::ParallelExecutionSpace<3>{
      {bx.start(0), bx.start(1), bx.start(2)}, 
        {bx.end(0)+1, bx.end(1)+1, bx.end(2)+1}
    };
  }
}

