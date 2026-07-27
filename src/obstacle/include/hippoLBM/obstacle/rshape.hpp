#pragma once

#include <onika/math/basic_types.h>
#include <onika/math/matrix4d.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/block_parallel_for.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/grid.hpp>
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

  inline void apply_to_grid(const LBMGrid& grid, int* const obst, onika::parallel::ParallelExecutionContext* exec_ctx,
                            int value = WALL_);
};

struct ApplyRShapeToGridFunctor {
  const onika::math::Vec3d* const __restrict__ vertices_;
  const uint32_t* const __restrict__ offset_;
  const uint32_t* const __restrict__ size_;
  double minkowski_;
  LBMGrid grid_;
  int* const __restrict__ obst_;
  int value_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const {
    const size_t f = coord.x;
    std::span<const onika::math::Vec3d> vertices(vertices_ + offset_[f], size_[f]);
    onika::math::AABB bounds = compute_aabb(vertices, minkowski_);
    Point3D pmin = {int(bounds.bmin.x / grid_.dx_), int(bounds.bmin.y / grid_.dx_), int(bounds.bmin.z / grid_.dx_)};
    Point3D pmax = {int(bounds.bmax.x / grid_.dx_), int(bounds.bmax.y / grid_.dx_), int(bounds.bmax.z / grid_.dx_)};
    Box3D global_box = {pmin, pmax};

    auto [is_inside_subdomain, local_box] = grid_.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);
    if (!is_inside_subdomain) return;

    for (int k = local_box.start(2) + ONIKA_CU_THREAD_COORD.z; k <= local_box.end(2); k += ONIKA_CU_BLOCK_DIMS.z) {
      for (int j = local_box.start(1) + ONIKA_CU_THREAD_COORD.y; j <= local_box.end(1); j += ONIKA_CU_BLOCK_DIMS.y) {
        for (int i = local_box.start(0) + ONIKA_CU_THREAD_COORD.x; i <= local_box.end(0); i += ONIKA_CU_BLOCK_DIMS.x) {
          onika::math::Vec3d p = grid_.compute_position<Area::Global>(i, j, k);
          if (intersect_point_face(p, vertices, minkowski_)) {
            obst_[grid_(i, j, k)] = value_;
          }
        }
      }
    }
  }
};

inline void RShape::apply_to_grid(const LBMGrid& grid, int* const obst,
                                  onika::parallel::ParallelExecutionContext* exec_ctx, int value) {
  if (faces_ == 0) return;
  ApplyRShapeToGridFunctor func = {vertices_.data(), offset_.data(), size_.data(), minkowski_, grid, obst, value};
  onika::parallel::ParallelExecutionSpace<3> space = {{0, 0, 0}, {ssize_t(faces_), 1, 1}};
  onika::parallel::block_parallel_for(space, func, exec_ctx);
}
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <>
struct BlockParallelForFunctorTraits<hippoLBM::ApplyRShapeToGridFunctor> {
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
