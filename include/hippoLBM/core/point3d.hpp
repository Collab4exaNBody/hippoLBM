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

typedef std::array<int, 3> int3d;

inline ONIKA_HOST_DEVICE_FUNC int3d operator+(int3d& a, int b) {
  int3d res;
  for (int dim = 0; dim < 3; dim++) res[dim] = a[dim] + b;
  return res;
}

/** @brief A 3D point in space. */
struct Point3D {
  int3d position_;  ///<! The coordinates of the point in 3D space.

  ONIKA_HOST_DEVICE_FUNC Point3D() {};
  ONIKA_HOST_DEVICE_FUNC Point3D(int3d in) { position_ = in; };
  ONIKA_HOST_DEVICE_FUNC Point3D(int x, int y, int z) {
    position_[0] = x;
    position_[1] = y;
    position_[2] = z;
  }
  ONIKA_HOST_DEVICE_FUNC inline int get_val(int dim) { return position_[dim]; }
  ONIKA_HOST_DEVICE_FUNC inline void set_val(int dim, int val) { position_[dim] = val; }
  ONIKA_HOST_DEVICE_FUNC inline int& operator[](int dim) { return position_[dim]; }
  ONIKA_HOST_DEVICE_FUNC inline const int& operator[](int dim) const { return position_[dim]; }
  void print() {
    for (int dim = 0; dim < 3; dim++) {
      onika::lout << " " << position_[dim];
    }

    onika::lout << std::endl;
  }

  ONIKA_HOST_DEVICE_FUNC Point3D operator+(Point3D& p) {
    Point3D res = {position_[0] + p[0], position_[1] + p[1], position_[2] + p[2]};
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC Point3D operator+(const Point3D& p) {
    Point3D res = {position_[0] + p[0], position_[1] + p[1], position_[2] + p[2]};
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC Point3D operator-(Point3D& p) {
    Point3D res = {position_[0] - p[0], position_[1] - p[1], position_[2] - p[2]};
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC Point3D operator-(const Point3D& p) {
    Point3D res = {position_[0] - p[0], position_[1] - p[1], position_[2] - p[2]};
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC Point3D& operator+=(const Point3D& p) {
    for (int d = 0; d < 3; d++) position_[d] += p[d];
    return *this;
  }
  ONIKA_HOST_DEVICE_FUNC Point3D& operator-=(const Point3D& p) {
    for (int d = 0; d < 3; d++) position_[d] -= p[d];
    return *this;
  }
};

inline ONIKA_HOST_DEVICE_FUNC Point3D min(Point3D& a, Point3D& b) {
  Point3D res;
  for (int dim = 0; dim < 3; dim++) {
    res[dim] = std::min(a[dim], b[dim]);
  }
  return res;
}

inline ONIKA_HOST_DEVICE_FUNC Point3D max(Point3D& a, Point3D& b) {
  Point3D res;
  for (int dim = 0; dim < 3; dim++) {
    res[dim] = std::max(a[dim], b[dim]);
  }
  return res;
}

// Vec3d compound assignment with Point3D
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d& operator+=(onika::math::Vec3d& v, const Point3D& p) {
  v.x += p[0];
  v.y += p[1];
  v.z += p[2];
  return v;
}
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d& operator-=(onika::math::Vec3d& v, const Point3D& p) {
  v.x -= p[0];
  v.y -= p[1];
  v.z -= p[2];
  return v;
}
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d& operator*=(onika::math::Vec3d& v, const Point3D& p) {
  v.x *= p[0];
  v.y *= p[1];
  v.z *= p[2];
  return v;
}

// Point3D <-> Vec3d mixed arithmetic (result is Vec3d)
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator+(const Point3D& p, const onika::math::Vec3d& v) {
  return {p[0] + v.x, p[1] + v.y, p[2] + v.z};
}
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator+(const onika::math::Vec3d& v, const Point3D& p) {
  return {v.x + p[0], v.y + p[1], v.z + p[2]};
}
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator-(const Point3D& p, const onika::math::Vec3d& v) {
  return {p[0] - v.x, p[1] - v.y, p[2] - v.z};
}
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator-(const onika::math::Vec3d& v, const Point3D& p) {
  return {v.x - p[0], v.y - p[1], v.z - p[2]};
}
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator*(const Point3D& p, double s) {
  return {p[0] * s, p[1] * s, p[2] * s};
}
ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d operator*(double s, const Point3D& p) {
  return {s * p[0], s * p[1], s * p[2]};
}

inline std::ostream& operator<<(std::ostream& os, const Point3D& p) {
  return os << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
}

// Structured binding support: auto [x, y, z] = point3d;
template <std::size_t N>
ONIKA_HOST_DEVICE_FUNC inline int& get(Point3D& p) {
  return p[N];
}
template <std::size_t N>
ONIKA_HOST_DEVICE_FUNC inline const int& get(const Point3D& p) {
  return p[N];
}

}  // namespace hippoLBM

// YAML
namespace YAML {
using hippoLBM::Point3D;

template <>
struct convert<Point3D> {
  static inline Node encode(const Point3D& v) {
    Node node;
    node.push_back(v[0]);
    node.push_back(v[1]);
    node.push_back(v[2]);
    return node;
  }
  static inline bool decode(const Node& node, Point3D& v) {
    if (!node.IsSequence() || node.size() != 3) {
      return false;
    }
    v[0] = node[0].as<int>();
    v[1] = node[1].as<int>();
    v[2] = node[2].as<int>();
    return true;
  }
};
}  // namespace YAML

// trick to do auto [x,y,z] = grid(idx)
namespace std {
template <>
struct tuple_size<hippoLBM::Point3D> : integral_constant<size_t, 3> {};
template <size_t N>
struct tuple_element<N, hippoLBM::Point3D> {
  using type = int;
};
}  // namespace std
