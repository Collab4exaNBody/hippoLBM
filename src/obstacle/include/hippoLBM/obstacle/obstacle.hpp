#pragma once

#include <onika/math/basic_types.h>
#include <onika/math/matrix4d.h>

namespace hippoLBM {

/** @brief Check if a point intersects with an axis-aligned bounding box.
 *  @param aabb The axis-aligned bounding box.
 *  @param v The point to check.
 *  @return True if the point intersects with the bounding box, false otherwise.
 */
inline bool intersect(onika::math::AABB& aabb, onika::math::Vec3d& v) {
  auto& min = aabb.bmin;
  auto& max = aabb.bmax;
  return min.x < v.x && v.x < max.x && min.y < v.y && v.y < max.y && min.z < v.z && v.z < max.z;
}

/** @brief Enumerate the different types of obstacles. */
enum OBSTACLE_TYPE {
  BALL = 0,     /**< Ball driver type. */
  WALL = 1,     /**< Wall driver type. */
  QUADRICS = 2, /**< Quadric driver type. */
  STL_MESH = 3, /**< STL mesh driver type. */
  UNDEFINED = 4 /**< Undefined driver type. */
};

template <typename Object, typename Func, typename... Args>
inline void apply(Object& obj, Func& func, Args... args) {}

class Ball {
  onika::math::Vec3d m_center_;  // The center of the ball.
  double m_radius_;              // The radius of the ball.
  double m_r2_;                  // The squared radius of the ball, used for efficient distance calculations.

 public:
  /** @brief Construct a ball obstacle.
   *  @param c The center of the ball.
   *  @param rad The radius of the ball.
   */
  Ball(onika::math::Vec3d c, double rad) : m_center_(c), m_radius_(rad) { m_r2_ = rad * rad; }

  /** @brief Get the axis-aligned bounding box covering the ball.
   *  @return The axis-aligned bounding box.
   */
  onika::math::AABB covered() {
    onika::math::AABB res = {m_center_ - m_radius_, m_center_ + m_radius_};
    return res;
  }

  /** @brief Get the type of the obstacle.
   *  @return The type of the obstacle.
   */
  constexpr OBSTACLE_TYPE type() { return OBSTACLE_TYPE::BALL; }

  ONIKA_HOST_DEVICE_FUNC inline onika::math::Vec3d& center() { return m_center_; }
  ONIKA_HOST_DEVICE_FUNC inline const onika::math::Vec3d& center() const { return m_center_; }
  ONIKA_HOST_DEVICE_FUNC inline double rcut2() { return m_r2_; }
  ONIKA_HOST_DEVICE_FUNC inline const double rcut2() const { return m_r2_; }

  ONIKA_HOST_DEVICE_FUNC bool solid(onika::math::Vec3d&& pos) {
    onika::math::Vec3d r = pos - m_center_;
    return dot(r, r) <= m_r2_;
  }
};

class Wall {
  onika::math::AABB bounds_;  // The axis-aligned bounding box representing the wall's spatial extent.

 public:
  /** @brief Construct a wall obstacle.
   *  @param bds The axis-aligned bounding box representing the wall's spatial extent.
   */
  Wall(onika::math::AABB bds) : bounds_(bds) {}

  /** @brief Get the axis-aligned bounding box covering the wall.
   *  @return The axis-aligned bounding box.
   */
  onika::math::AABB covered() { return bounds_; }

  /** @brief Get the type of the obstacle.
   *  @return The type of the obstacle.
   */
  constexpr OBSTACLE_TYPE type() { return OBSTACLE_TYPE::WALL; }

  /** @brief Check if a point is inside the wall.
   *  @param pos The point to check.
   *  @return True if the point is inside the wall, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool solid(onika::math::Vec3d&& pos) { return intersect(bounds_, pos); }
};

class Quadric {
  onika::math::Mat4d quadric_;  // The quadric matrix representing the quadric surface.

 public:
  /** @brief Construct a quadric obstacle.
   *  @param q The quadric matrix representing the quadric surface.
   */
  Quadric(onika::math::Mat4d q) : quadric_(q) {}

  /** @brief Get the axis-aligned bounding box covering the quadric.
   *  @return The axis-aligned bounding box.
   */
  onika::math::AABB covered() {
    // For simplicity, we return a large bounding box. In practice, you may want to compute the actual bounds.
    // Complicated and depending of the quadric type. For now, we return a large bounding box.
    return onika::math::AABB{{-1e6, -1e6, -1e6}, {1e6, 1e6, 1e6}};
  }

  /** @brief Get the type of the obstacle.
   *  @return The type of the obstacle.
   */
  constexpr OBSTACLE_TYPE type() { return OBSTACLE_TYPE::QUADRICS; }

  /** @brief Check if a point is inside the wall.
   *  @param pos The point to check.
   *  @return True if the point is inside the wall, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool solid(onika::math::Vec3d&& pos) {
    return onika::math::quadric_eval(quadric_, pos) <= 0.0;
  }
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
template <>
constexpr OBSTACLE_TYPE get_type<Quadric>() {
  return OBSTACLE_TYPE::QUADRICS;
}
}  // namespace hippoLBM
