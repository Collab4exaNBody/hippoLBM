#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/basic_types_operators.h>

#pragma once

namespace hippoLBM
{
  struct NullFuncWriter
  {
    template<typename T>
    inline T& operator()(const int idx, T& data) const{ return data; }
  }; 

  struct UWriter
  {
    const int * const obst;
    const double ratio_dx_dtLB;
    inline Vec3d operator()(const int idx, const Vec3d& m1) const
    {
      if(obst[idx] == FLUIDE_)
      {
        return ratio_dx_dtLB * m1;
      }
      return Vec3d{0,0,0};
    }
  };

  struct PressionWriter
  {
    const int * const obst;
    const double c_c_avg_rho_div_three;
    inline double operator()(const int idx, const double& m0) const
    {
      if(obst[idx] == FLUIDE_)
      {
        return c_c_avg_rho_div_three * (m0 - 1);
      }
      return 0;
    }
  };

	template<typename Func>
		struct write_file
		{
      Func func;
      template<typename T>
			inline void operator()(int idx, std::stringstream& output, T* const ptr) const 
			{
				T tmp = ptr[idx];
        tmp = func(idx, tmp);
				output << (T)tmp << " ";
			}
		};

	template<int Q>
		struct write_distributions
		{
			inline void operator()(int idx, std::stringstream& output, const WrapperF<Q>& fi) const
			{
				for(int i = 0 ; i < Q ; i ++) 
				{
					double tmp = fi(idx,i);
					output << (float)tmp << " ";
				}
			}
		};
  
  template<typename Func>
	struct write_vec3d
	{
    Func func;
		box<3> b;
		inline void operator()(const int x, const int y, const int z, std::stringstream& output, onika::math::Vec3d* const ptr) const
		{
			const int idx = b(x,y,z);
			onika::math::Vec3d tmp = func(idx, ptr[idx]);
			output << (float)tmp.x << " " << (float)tmp.y << " " << (float)tmp.z << " ";
		}
	};
}
