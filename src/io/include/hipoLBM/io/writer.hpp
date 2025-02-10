#pragma once

#include <grid_lbm/parallel_for_core.hpp>

namespace hipoLBM
{
	struct write_file
	{
		template<typename T>
			inline void operator()(int idx, std::ofstream& output, T* const ptr) const
			{
				T tmp = ptr[idx];
				output << (float)tmp << " ";
			}
	};

	struct write_vec3d
	{
		inline void operator()(const int x, const int y, const int z, std::ofstream& output, onika::math::Vec3d* const ptr) const
		{
			const int idx = b(x,y,z);
			onika::math::Vec3d& tmp = ptr[idx];
			output << (float)tmp.x << " " << (float)tmp.y << " " << (float)tmp.z << " ";
		}
		box<3> b;
	};

  template<> struct ParallelForIdDataFunctorTraits<write_file>
  {
    static inline constexpr bool OpenMPCompatible = false;
  };

/*
  template<> struct ParallelForIdDataFunctorTraits<write_vec3d>
  {
    static inline constexpr bool OpenMPCompatible = false;
  };*/
}
