#pragma once

#include <onika/math/basic_types.h>
#include <onika/math/matrix4d.h>

#include <algorithm>
#include <cassert>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/grid.hpp>
#include <span>

namespace hippoLBM {

inline onika::math::AABB compute_aabb(std::span<onika::math::Vec3d> vertices, double minkowski = 0.0) {
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
  // Inflate by the Minkowski radius: the shape is the Minkowski sum of the
  // polygon with a ball of that radius, so its true bounds extend by it.
  res.bmin.x -= minkowski;
  res.bmin.y -= minkowski;
  res.bmin.z -= minkowski;
  res.bmax.x += minkowski;
  res.bmax.y += minkowski;
  res.bmax.z += minkowski;
  return res;
}

// Checks whether `p` lies within `minkowski` distance of the planar,
// convex face described by `vertices` (ordered, consistent winding). Covers
// the three regions of the (rounded) face: the flat interior, the rounded
// edges, and the rounded corners (vertices).
inline bool intersect_point_face(const onika::math::Vec3d& p, std::span<onika::math::Vec3d> vertices,
                                 double minkowski) {
  assert(vertices.size() >= 3);

  onika::math::Vec3d normal = onika::math::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
  normal = normal * (1.0 / onika::math::norm(normal));

  double dist = onika::math::dot(p - vertices[0], normal);

  // Project the point onto the face plane, then check it falls inside the
  // polygon: for a convex polygon with consistent winding, the projection is
  // inside iff it stays on the same side of every edge (same sign as normal).
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

  // Inside the polygon: the closest point of the face is the projection
  // itself, so the distance to test is simply the (perpendicular) plane
  // distance.
  if (inside) {
    return std::abs(dist) <= minkowski;
  }

  // Outside the polygon: the closest point of the face lies on its boundary.
  // Clamping the projection of `p` onto each edge's *segment* (not the
  // infinite line) covers both the edges and, when the closest point falls
  // on an endpoint, the vertices.
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
  std::vector<onika::math::Vec3d> vertices_;
  std::vector<uint32_t> offset_;
  std::vector<uint32_t> size_;

  void add_face(std::span<onika::math::Vec3d> vertices) {
    uint32_t offset = faces_ == 0 ? 0 : offset_.back() + size_.back();
    offset_.push_back(offset);
    size_.push_back(vertices.size());
    vertices_.insert(vertices_.end(), vertices.begin(), vertices.end());
    faces_++;
  }

  inline std::span<onika::math::Vec3d> face(uint32_t idx) {
    return std::span<onika::math::Vec3d>(vertices_.data() + offset_[idx], size_[idx]);
  }

  inline onika::math::AABB face_aabb(uint32_t idx) { return compute_aabb(face(idx), minkowski_); }

  // Marks every local LBM grid node covered by this shape (i.e. within
  // `minkowski_` of one of its faces) as `value` (WALL_ by default) in the
  // `obst` field. For each face, only the nodes inside its (Minkowski
  // inflated) bounding box, restricted to the local subdomain, are tested.
  inline void apply_to_grid(LBMGrid& grid, int* const obst, int value = WALL_) {
    for (uint32_t f = 0; f < faces_; f++) {
      std::span<onika::math::Vec3d> vertices = face(f);
      onika::math::AABB bounds = compute_aabb(vertices, minkowski_);
      Point3D pmin = {int(bounds.bmin.x / grid.dx_), int(bounds.bmin.y / grid.dx_), int(bounds.bmin.z / grid.dx_)};
      Point3D pmax = {int(bounds.bmax.x / grid.dx_), int(bounds.bmax.y / grid.dx_), int(bounds.bmax.z / grid.dx_)};
      Box3D global_box = {pmin, pmax};

      auto [is_inside_subdomain, local_box] = grid.restrict_box_to_grid<Area::Local, Traversal::Extend>(global_box);
      if (!is_inside_subdomain) continue;

      for (int k = local_box.start(2); k <= local_box.end(2); k++) {
        for (int j = local_box.start(1); j <= local_box.end(1); j++) {
          for (int i = local_box.start(0); i <= local_box.end(0); i++) {
            onika::math::Vec3d p = grid.compute_position<Area::Global>(i, j, k);
            if (intersect_point_face(p, vertices, minkowski_)) {
              obst[grid(i, j, k)] = value;
            }
          }
        }
      }
    }
  }
};
}  // namespace hippoLBM
