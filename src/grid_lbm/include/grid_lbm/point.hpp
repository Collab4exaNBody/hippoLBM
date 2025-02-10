#pragma once

namespace hipoLBM
{
	typedef std::array<int,3> int3d;

	inline
		int3d operator+(int3d& a, int b)
		{
			int3d res;
			for (int dim = 0 ; dim < 3 ; dim++) res[dim] = a[dim] + b;
			return res;
		}

	// should change it to n-DIM
	template<int DIM>
		struct point
		{
			int3d position;
			inline int get_val(int dim) {return position[dim];}
			inline void set_val(int dim, int val) { position[dim] = val;}
			inline int& operator[](int dim) {return position[dim];}
			inline int operator[](int dim) const {return position[dim];}	
			void print() 
			{
				for(int dim = 0; dim < DIM ; dim++) 
				{
					onika::lout << " " << position[dim];
				}

				onika::lout << std::endl;
			}

			point<DIM> operator+(point<DIM>& p)
			{
				point<DIM> res = {position[0] + p[0], position[1] + p[1], position[2] + p[2]};
				return res;
			} 

			point<DIM> operator-(point<DIM>& p)
			{
				point<DIM> res = {position[0] - p[0], position[1] - p[1], position[2] - p[2]};
				return res;
			} 
		};

	template<int DIM>
		point<DIM> min(point<DIM>& a, point<DIM>& b)
		{
			point<DIM> res;
			for(int dim = 0 ; dim < DIM ; dim++)
			{
				res[dim] = std::min(a[dim], b[dim]);
			}
			return res;
		}

	template<int DIM>
		point<DIM> max(point<DIM>& a, point<DIM>& b)
		{
			point<DIM> res;
			for(int dim = 0 ; dim < DIM ; dim++)
			{
				res[dim] = std::max(a[dim], b[dim]);
				//std::cout << " res " << res[dim] << " max( " << a[dim] << " , " << b[dim] << ")" << std::endl;
			}
			return res;
		}
}
