#pragma once

#include <onika/math/basic_types.h>

namespace hippoLBM {
using namespace onika::math;

inline bool intersect(onika::math::AABB& aabb, onika::math::Vec3d& v) {
  auto& min = aabb.bmin;
  auto& max = aabb.bmax;
  return min.x < v.x && v.x < max.x && min.y < v.y && v.y < max.y && min.z < v.z && v.z < max.z;
}

enum OBSTACLE_TYPE {
  BALL = 0,     /**< Ball driver type. */
  WALL = 1,     /**< Wall driver type. */
  STL_MESH = 2, /**< STL mesh driver type. */
  UNDEFINED = 3 /**< Undefined driver type. */
};

class AbstractObject {
  virtual onika::math::AABB covered() = 0;
  virtual constexpr OBSTACLE_TYPE type() = 0;
  virtual bool solid(onika::math::Vec3d&& pos) = 0;
};

template <typename Object, typename Func, typename... Args>
inline void apply(Object& obj, Func& func, Args... args) {}

class Ball : public AbstractObject {
  onika::math::Vec3d m_center;
  double m_radius;
  double m_r2;

 public:
  Ball(onika::math::Vec3d c, double rad) : m_center(c), m_radius(rad) { m_r2 = rad * rad; }

  onika::math::AABB covered() {
    onika::math::AABB res = {m_center - m_radius, m_center + m_radius};
    return res;
  }

  constexpr OBSTACLE_TYPE type() { return OBSTACLE_TYPE::BALL; }

  ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d& center() { return m_center; }
  ONIKA_HOST_DEVICE_FUNC inline const onika::math::Vec3d& center() const { return m_center; }
  ONIKA_HOST_DEVICE_FUNC inline double rcut2() { return m_r2; }
  ONIKA_HOST_DEVICE_FUNC inline const double rcut2() const { return m_r2; }

  ONIKA_HOST_DEVICE_FUNC bool solid(onika::math::Vec3d&& pos) {
    onika::math::Vec3d r = pos - m_center;
    return dot(r, r) <= m_r2;
  }
};

class Wall : public AbstractObject {
  onika::math::AABB bounds;

 public:
  Wall(onika::math::AABB bds) : bounds(bds) {}

  onika::math::AABB covered() { return bounds; }

  constexpr OBSTACLE_TYPE type() { return OBSTACLE_TYPE::WALL; }

  ONIKA_HOST_DEVICE_FUNC bool solid(onika::math::Vec3d&& pos) { return intersect(bounds, pos); }
};

template <typename T>
inline constexpr OBSTACLE_TYPE get_type();
template <>
constexpr OBSTACLE_TYPE get_type<Ball>() {
  return OBSTACLE_TYPE::BALL;
}
template <>
constexpr OBSTACLE_TYPE get_type<Wall>() {
  return OBSTACLE_TYPE::WALL;
}
}  // namespace hippoLBM
