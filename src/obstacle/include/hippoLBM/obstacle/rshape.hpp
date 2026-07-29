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
#include <onika/math/matrix4d.h>
#include <onika/memory/allocator.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <span>
#include <string>

namespace hippoLBM {
ONIKA_HOST_DEVICE_FUNC inline onika::math::AABB compute_aabb(std::span<const onika::math::Vec3d> vertices,
                                                             double minkowski = 0.0) {
  assert(vertices.size() >= 3);
  onika::math::AABB res;
  res.bmin = vertices[0];
  res.bmax = vertices[0];
  for (const auto& v : vertices) {
    res.bmin.x = std::min(res.bmin.x, v.x);
    res.bmin.y = std::min(res.bmin.y, v.y);
    res.bmin.z = std::min(res.bmin.z, v.z);
    res.bmax.x = std::max(res.bmax.x, v.x);
    res.bmax.y = std::max(res.bmax.y, v.y);
    res.bmax.z = std::max(res.bmax.z, v.z);
  }

  res.bmin.x -= minkowski;
  res.bmin.y -= minkowski;
  res.bmin.z -= minkowski;
  res.bmax.x += minkowski;
  res.bmax.y += minkowski;
  res.bmax.z += minkowski;
  return res;
}

ONIKA_HOST_DEVICE_FUNC inline bool intersect_point_face(const onika::math::Vec3d& p,
                                                        std::span<const onika::math::Vec3d> vertices,
                                                        double minkowski) {
  assert(vertices.size() >= 3);

  onika::math::Vec3d normal = onika::math::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
  normal = normal * (1.0 / onika::math::norm(normal));

  double dist = onika::math::dot(p - vertices[0], normal);

  onika::math::Vec3d proj = p - dist * normal;
  const size_t n = vertices.size();
  bool inside = true;
  for (size_t i = 0; i < n; i++) {
    const onika::math::Vec3d& a = vertices[i];
    const onika::math::Vec3d& b = vertices[(i + 1) % n];
    onika::math::Vec3d edge = b - a;
    onika::math::Vec3d to_proj = proj - a;
    if (onika::math::dot(onika::math::cross(edge, to_proj), normal) < 0.0) {
      inside = false;
      break;
    }
  }

  if (inside) {
    return std::abs(dist) <= minkowski;
  }

  const double mink2 = minkowski * minkowski;
  for (size_t i = 0; i < n; i++) {
    const onika::math::Vec3d& a = vertices[i];
    const onika::math::Vec3d& b = vertices[(i + 1) % n];
    onika::math::Vec3d edge = b - a;
    double t = onika::math::dot(p - a, edge) / onika::math::dot(edge, edge);
    t = std::clamp(t, 0.0, 1.0);
    onika::math::Vec3d closest = a + edge * t;
    onika::math::Vec3d diff = p - closest;
    if (onika::math::dot(diff, diff) <= mink2) {
      return true;
    }
  }
  return false;
}

struct RShape {
  double minkowski_;
  uint32_t faces_;
  onika::memory::CudaMMVector<onika::math::Vec3d> vertices_;
  onika::memory::CudaMMVector<uint32_t> offset_;
  onika::memory::CudaMMVector<uint32_t> size_;

  void add_face(std::span<onika::math::Vec3d> vertices) {
    uint32_t offset = faces_ == 0 ? 0 : offset_.back() + size_.back();
    offset_.push_back(offset);
    size_.push_back(vertices.size());
    vertices_.insert(vertices_.end(), vertices.begin(), vertices.end());
    faces_++;
  }

  void read_stl(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open());

    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    bool is_binary = false;
    uint32_t num_triangles = 0;
    if (file_size >= 84) {
      char header[80];
      file.read(header, sizeof(header));
      file.read(reinterpret_cast<char*>(&num_triangles), sizeof(num_triangles));
      const std::streamoff expected_size = 84 + std::streamoff(num_triangles) * 50;
      is_binary = (expected_size == file_size);
    }

    if (is_binary) {
      // Stream is already positioned right after the 80-byte header and the
      // triangle count (4 bytes), i.e. at the first triangle record.
      for (uint32_t t = 0; t < num_triangles; t++) {
        float normal[3];
        file.read(reinterpret_cast<char*>(normal), sizeof(normal));
        onika::math::Vec3d tri[3];
        for (auto& v : tri) {
          float p[3];
          file.read(reinterpret_cast<char*>(p), sizeof(p));
          v = {double(p[0]), double(p[1]), double(p[2])};
        }
        uint16_t attribute_byte_count = 0;
        file.read(reinterpret_cast<char*>(&attribute_byte_count), sizeof(attribute_byte_count));
        add_face(std::span<onika::math::Vec3d>(tri, 3));
      }
      return;
    }

    file.close();
    std::ifstream ascii(filename);
    assert(ascii.is_open());
    std::string token;
    onika::math::Vec3d tri[3];
    int count = 0;
    while (ascii >> token) {
      if (token == "vertex") {
        double x, y, z;
        ascii >> x >> y >> z;
        tri[count++] = {x, y, z};
        if (count == 3) {
          add_face(std::span<onika::math::Vec3d>(tri, 3));
          count = 0;
        }
      }
    }
  }

  inline std::span<onika::math::Vec3d> face(uint32_t idx) {
    return std::span<onika::math::Vec3d>(vertices_.data() + offset_[idx], size_[idx]);
  }

  ONIKA_HOST_DEVICE_FUNC inline std::span<const onika::math::Vec3d> face(uint32_t idx) const {
    return std::span<const onika::math::Vec3d>(vertices_.data() + offset_[idx], size_[idx]);
  }

  inline onika::math::AABB face_aabb(uint32_t idx) { return compute_aabb(face(idx), minkowski_); }

  void print() {
    onika::lout << "RShape: " << faces_ << " faces, " << vertices_.size() << " vertices, minkowski: " << minkowski_
                << std::endl;
  }
};
}  // namespace hippoLBM