#pragma once

#include <grid_lbm/wrapper_f.hpp>
using namespace std;

namespace hippoLBM
{

  //number of unknowns fi in 3DQ19
  using namespace onika::math;

  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

  template<int Q>
    struct scheme_lbm {};

  template<> struct scheme_lbm<19>
  {
    const vector_t<double> w = {1. / 3, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36};
    const vector_t<int> ex = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
    const vector_t<int> ey = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
    const vector_t<int> ez = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
    const vector_t<int> iopp = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
  };

  template<int Q>
    struct grid_data_lbm 
    {
      int grid_size;
      scheme_lbm<Q> scheme;

      // fields
      vector_t<double> f; // fi
      vector_t<double> m0; // densities
      vector_t<Vec3d> m1; // flux
      vector_t<int> obst; // obstacles

      // dunno
      vector_t<double> fi_x_0, fi_x_l, fi_y_0, fi_y_l, fi_z_0, fi_z_l;

      grid_data_lbm() {}

      // accessors
      WrapperF<Q> distributions() { 
        return WrapperF<Q>{onika::cuda::vector_data(f), grid_size}; 
      }
      //double * distributions() { return onika::cuda::vector_data(f); }
      double * densities() { return onika::cuda::vector_data(m0); }
      Vec3d * flux() { return onika::cuda::vector_data(m1); }
      int * obstacles() { return onika::cuda::vector_data(obst); }
      const double * weights() { return onika::cuda::vector_data(scheme.w); }
      std::tuple<const int *, const int * , const int *> exyz() 
      {
        const int * ex = onika::cuda::vector_data(scheme.ex); 
        const int * ey = onika::cuda::vector_data(scheme.ey); 
        const int * ez = onika::cuda::vector_data(scheme.ez); 
        return {ex,ey,ez}; 
      }
    };
}
