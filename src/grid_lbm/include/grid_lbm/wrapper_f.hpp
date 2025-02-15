#pragma once

namespace hipoLBM
{
  struct WrapperF
  {
    double * const f;
    int N;
    ONIKA_HOST_DEVICE_FUNC double& operator()(int idx, int iLB) 
    {
      return f[N * iLB + idx];
    }
    
    

    ONIKA_HOST_DEVICE_FUNC double& operator()(int idx, int iLB) const 
    {
      return f[N * iLB + idx];
    }
  };
}
