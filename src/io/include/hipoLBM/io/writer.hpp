#pragma once

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
		inline void operator()(const int x, const int y, const int z, std::ofstream& output, vec3d* const ptr) const
		{
			const int idx = b(x,y,z);
			vec3d& tmp = ptr[idx];
			output << (float)tmp.x1 << " " << (float)tmp.x2 << " " << (float)tmp.x3 << " ";
		}
		box<3> b;
	}
}
