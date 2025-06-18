#pragma once

namespace hippoLBM
{

  typedef std::array<int,3> int3d;

  inline
    ONIKA_HOST_DEVICE_FUNC int3d operator+(int3d& a, int b)
    {
      int3d res;
      for (int dim = 0 ; dim < 3 ; dim++) res[dim] = a[dim] + b;
      return res;
    }

  struct LBMPoint
  {
    int3d position;

    ONIKA_HOST_DEVICE_FUNC LBMPoint() {};
    ONIKA_HOST_DEVICE_FUNC LBMPoint(int x, int y, int z) { position[0] = x; position[1] = y; position[2] = z; }
    ONIKA_HOST_DEVICE_FUNC inline int get_val(int dim) {return position[dim];}
    ONIKA_HOST_DEVICE_FUNC inline void set_val(int dim, int val) { position[dim] = val;}
    ONIKA_HOST_DEVICE_FUNC inline int& operator[](int dim) {return position[dim];}
    ONIKA_HOST_DEVICE_FUNC inline const int& operator[](int dim) const {return position[dim];}  
    void print() 
    {
      for(int dim = 0; dim < 3 ; dim++) 
      {
        onika::lout << " " << position[dim];
      }

      onika::lout << std::endl;
    }

    ONIKA_HOST_DEVICE_FUNC LBMPoint operator+(LBMPoint& p)
    {
      LBMPoint res = {position[0] + p[0], position[1] + p[1], position[2] + p[2]};
      return res;
    } 

    ONIKA_HOST_DEVICE_FUNC LBMPoint operator+(const LBMPoint& p)
    {
      LBMPoint res = {position[0] + p[0], position[1] + p[1], position[2] + p[2]};
      return res;
    } 

    ONIKA_HOST_DEVICE_FUNC LBMPoint operator-(LBMPoint& p)
    {
      LBMPoint res = {position[0] - p[0], position[1] - p[1], position[2] - p[2]};
      return res;
    } 

    ONIKA_HOST_DEVICE_FUNC LBMPoint operator-(const LBMPoint& p)
    {
      LBMPoint res = {position[0] - p[0], position[1] - p[1], position[2] - p[2]};
      return res;
    } 
  };

  inline ONIKA_HOST_DEVICE_FUNC LBMPoint min(LBMPoint& a, LBMPoint& b)
  {
    LBMPoint res;
    for(int dim = 0 ; dim < 3 ; dim++)
    {
      res[dim] = std::min(a[dim], b[dim]);
    }
    return res;
  }

  inline ONIKA_HOST_DEVICE_FUNC LBMPoint max(LBMPoint& a, LBMPoint& b)
  {
    LBMPoint res;
    for(int dim = 0 ; dim < 3 ; dim++)
    {
      res[dim] = std::max(a[dim], b[dim]);
      //std::cout << " res " << res[dim] << " max( " << a[dim] << " , " << b[dim] << ")" << std::endl;
    }
    return res;
  }
}
