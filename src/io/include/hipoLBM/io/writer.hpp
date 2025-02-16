#pragma once

namespace hipoLBM
{
  template<typename T>
    struct write_file
    {
      inline void operator()(int idx, std::ofstream& output, T* const ptr) const
      {
	T tmp = ptr[idx];
	output << (float)tmp << " ";
      }
    };

  template<int Q>
    struct write_distributions
    {
      inline void operator()(int idx, std::ofstream& output, const WrapperF<Q>& ptr) const
      {
	for(int i = 0 ; i < Q ; i ++) 
	{
	  double tmp = ptr(idx,i);
	  output << (float)tmp << " ";
	}
      }
    };

  struct write_vec3d
  {
    box<3> b;
    inline void operator()(const int x, const int y, const int z, std::ofstream& output, onika::math::Vec3d* const ptr) const
    {
      const int idx = b(x,y,z);
      onika::math::Vec3d& tmp = ptr[idx];
      output << (float)tmp.x << " " << (float)tmp.y << " " << (float)tmp.z << " ";
    }
  };
}
