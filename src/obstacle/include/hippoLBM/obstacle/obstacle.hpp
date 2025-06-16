#pragma once

#include <onika/math/basic_types.h>

namespace hippoLBM
{
  using namespace onika::math;

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


	template<typename T> inline constexpr OBSTACLE_TYPE get_type();
	template<> constexpr OBSTACLE_TYPE get_type<Ball>() { return OBSTACLE_TYPE::BALL; }
}
