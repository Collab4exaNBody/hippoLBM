#pragma once

namespace hippoLBM
{
  template<int Q>
    struct WrapperF
    {
      double * const f;
      int N;
      ONIKA_HOST_DEVICE_FUNC inline double& operator()(int idx, int iLB) 
      {
        assert(idx < N);
        assert(iLB < Q);
#ifdef WFAOS
        // case 1 
        return f[idx*Q + iLB];
#else
        // case 2
        return f[N * iLB + idx];
#endif
      }

      ONIKA_HOST_DEVICE_FUNC inline double& operator()(int idx, int iLB) const 
      {
        assert(idx < N);
        assert(iLB < Q);
#ifdef WFAOS
        // case 1
        return f[idx*Q + iLB];
#else
        // case 2
        return f[N * iLB + idx];
#endif
      }

    };

  template<int Q>
    ONIKA_HOST_DEVICE_FUNC inline void copyTo(const WrapperF<Q>& dest_data, int dest_idx, const WrapperF<Q>& from_data, int from_idx, int size)
    {
#ifdef WFAOS
      // case 1
      double * from = &from_data(from_idx, 0);
      double * dest = &dest_data(dest_idx, 0);
      int nb_byte = size * Q * sizeof(double);
      std::memcpy(dest, from, nb_byte);
#else
      // case 2
      int nb_byte = size * sizeof(double);
      for(int i = 0 ; i < Q ; i++)
      {
        std::memcpy(&dest_data(dest_idx,i), &from_data(from_idx,i), nb_byte);
      }
#endif
    }
}
