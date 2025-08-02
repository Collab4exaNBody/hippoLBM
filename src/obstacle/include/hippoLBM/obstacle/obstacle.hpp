#pragma once

#include <onika/math/basic_types.h>

namespace hippoLBM
{
  using namespace onika::math;

  inline bool intersect(AABB& aabb, Vec3d& v)
  {
    auto& min = aabb.bmin;
    auto& max = aabb.bmax;
		return min.x < v.x && v.x < max.x &&
			min.y < v.y && v.y < max.y &&
			min.z < v.z && v.z < max.z ;
	}

	enum OBSTACLE_TYPE
	{
		BALL      = 0, /**< Ball driver type. */
		WALL      = 1, /**< Wall driver type. */
		STL_MESH  = 2,  /**< STL mesh driver type. */
		UNDEFINED = 3 /**< Undefined driver type. */
	};


	class AbstractObject
	{
		virtual AABB covered() = 0;
		virtual constexpr OBSTACLE_TYPE type() = 0;
		virtual bool solid(Vec3d&& pos) = 0;
	};

	template<typename Object, typename Func, typename... Args>
		inline void apply(Object& obj, Func& func, Args... args) { }

	class Ball : public AbstractObject
	{
		Vec3d center;
		double radius;
		double r2;

		public:

		Ball() = delete;
		Ball(Vec3d c, double rad) : center(c), radius(rad)
		{
			r2 = rad * rad;
		}

		AABB covered()
		{
			AABB res = { center - radius , center + radius };
			return res;
		}

		constexpr OBSTACLE_TYPE type() { return OBSTACLE_TYPE::BALL; } 


		bool solid(Vec3d&& pos)
		{
			Vec3d r = pos - center;
			return dot(r,r) <= r2;
		}
	};

	class Wall : public AbstractObject
	{
		AABB bounds;

		public:

		Wall() = delete;
		Wall(AABB bds) : bounds(bds) {	}

		AABB covered()
		{
			return bounds;
		}

		constexpr OBSTACLE_TYPE type() { return OBSTACLE_TYPE::WALL; } 

		bool solid(Vec3d&& pos)
		{
			return intersect(bounds, pos);
		}
	};

	template<typename T> inline constexpr OBSTACLE_TYPE get_type();
	template<> constexpr OBSTACLE_TYPE get_type<Ball>() { return OBSTACLE_TYPE::BALL; }
	template<> constexpr OBSTACLE_TYPE get_type<Wall>() { return OBSTACLE_TYPE::WALL; }
}
