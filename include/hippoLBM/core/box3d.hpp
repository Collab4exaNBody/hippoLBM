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

#include <hippoLBM/core/point3d.hpp>

namespace hippoLBM {
/**
 * @brief A geometric box in a multi-dimensional space.
 *
 * This template struct represents a geometric box in a multi-dimensional space.
 * It is defined by two points, `inf_` and `sup_`, which represent the lower-left and
 * upper-right corners of the box in 2D, respectively. The struct also provides methods
 * for computing the box's dimensions and number of points within it.
 *
 * @tparam DIM  The dimensionality of the box (number of spatial dimensions).
 *
 * @param inf_  The lower-left corner of the box.
 * @param sup_  The upper-right corner of the box.
 */
struct Box3D {
  static constexpr int DIM = 3;
  Point3D inf_; /**< The lower-left corner of the box. */
  Point3D sup_; /**< The upper-right corner of the box. */

  /**
   * @brief Get the length of the box along a specified dimension.
   *
   * @param dim  The dimension for which to compute the length.
   * @return     The length of the box along the specified dimension.
   */
  ONIKA_HOST_DEVICE_FUNC inline int get_length(int dim) {
    assert(sup_[dim] >= inf_[dim]);
    return (sup_[dim] - inf_[dim]) + 1;
  }

  /**
   * @brief Get the length of the box along a specified dimension.
   *
   * @param dim  The dimension for which to compute the length.
   * @return     The length of the box along the specified dimension.
   */
  ONIKA_HOST_DEVICE_FUNC inline int get_length(int dim) const {
    assert(sup_[dim] >= inf_[dim]);
    return (sup_[dim] - inf_[dim]) + 1;
  }

  ONIKA_HOST_DEVICE_FUNC inline int start(int dim) const { return inf_[dim]; }

  ONIKA_HOST_DEVICE_FUNC inline int end(int dim) const { return sup_[dim]; }

  /**
   * @brief Calculate the total number of points within the box.
   *
   * @return The total number of points within the box.
   */
  ONIKA_HOST_DEVICE_FUNC inline int number_of_points() const {
    int res = 1;
    for (int dim = 0; dim < DIM; dim++) res *= (this->get_length(dim));  // sup_ is included (+1)
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool contains(Point3D& p) {
    for (int dim = 0; dim < DIM; dim++) {
      if ((p[dim] < inf_[dim]) || (p[dim] > sup_[dim])) return false;
    }
    return true;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool contains(Point3D&& p) {
    for (int dim = 0; dim < DIM; dim++) {
      if ((p[dim] < inf_[dim]) || (p[dim] > sup_[dim])) return false;
    }
    return true;
  }

  /**
   * @brief Print the box's lower-left and upper-right corners.
   */
  void print() {
    onika::lout << " inf_:";
    inf_.print();
    onika::lout << " sup_:";
    sup_.print();
  }

  /**
   * @brief Compute the index of a point within the box using Cartesian coordinates.
   *
   * @param x  The x-coordinate of the point.
   * @param y  The y-coordinate of the point.
   * @param z  The z-coordinate of the point.
   * @return   The index of the point within the box.
   */
  ONIKA_HOST_DEVICE_FUNC inline int operator()(const int x, const int y, const int z) {
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
  ONIKA_HOST_DEVICE_FUNC inline int operator()(const int x, const int y, const int z) const {
    int idx = z * (this->get_length(1)) + y;
    idx *= this->get_length(0);
    idx += x;
    return idx;
  }

  ONIKA_HOST_DEVICE_FUNC inline std::tuple<int, int, int> operator()(int idx) const {
    int size_y = this->get_length(1);
    int size_x = this->get_length(0);
    int size_xy = size_y * size_x;
    int z = idx / size_xy;
    idx = idx - z * size_xy;
    int y = idx / size_x;
    int x = idx % size_x;
    return {x, y, z};
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
   * @brief accessor to the `inf_` member
   */
  ONIKA_HOST_DEVICE_FUNC inline Point3D& lower() { return inf_; }
  /**
   * @brief accessor to the `inf_` member
   */
  ONIKA_HOST_DEVICE_FUNC inline const Point3D& lower() const { return inf_; }
  /**
   * @brief accessor to the `sup_` member
   */
  ONIKA_HOST_DEVICE_FUNC inline Point3D& upper() { return sup_; }
  /**
   * @brief accessor to the `sup_` member
   */
  ONIKA_HOST_DEVICE_FUNC inline const Point3D& upper() const { return sup_; }
};

/**
 * @brief Compute the index of a point within a multi-dimensional box using Cartesian coordinates.
 *
 * This template function calculates the index of a point within a multi-dimensional box
 * using Cartesian coordinates (x, y, z). It takes a `box` instance as input and delegates
 * the computation to the `box`'s operator() method.
 *
 * @param b    A reference to a `box` instance representing the multi-dimensional box.
 * @param x    The x-coordinate of the point.
 * @param y    The y-coordinate of the point.
 * @param z    The z-coordinate of the point.
 * @return     The index of the point within the box.
 */
ONIKA_HOST_DEVICE_FUNC inline int compute_idx(const Box3D& b, const int x, const int y, const int z) {
  return b(x, y, z);
}

ONIKA_HOST_DEVICE_FUNC inline bool intersect(Box3D& a, Box3D& b) {
  // Vérifier les conditions de non-intersection sur l'axe x
  for (int dim = 0; dim < Box3D::DIM; dim++) {
    if (a.sup_[dim] < b.inf_[dim] || b.sup_[dim] < a.inf_[dim]) {
      return false;
    }
  }
  // Si aucune des conditions de non-intersection n'est remplie, les boîtes s'intersectent
  return true;
}

ONIKA_HOST_DEVICE_FUNC inline onika::parallel::ParallelExecutionSpace<3> set(Box3D& bx) {
  return onika::parallel::ParallelExecutionSpace<3>{{bx.start(0), bx.start(1), bx.start(2)},
                                                    {bx.end(0) + 1, bx.end(1) + 1, bx.end(2) + 1}};
}
}  // namespace hippoLBM
