#pragma once

namespace hippoLBM
{
  template<int Components>
    struct FieldView
    {
      double * const __restrict__ data;
      int num_elements;
      ONIKA_HOST_DEVICE_FUNC 
        inline double& operator()(
            size_t idx, 
            size_t component_index) 
        {
          assert(idx < num_elements);
          assert(component_index < Components);
#ifdef WFAOS
          // case 1 
          return data[idx*Components + component_index];
#else
          // case 2
          return data[num_elements * component_index + idx];
#endif
        }

      ONIKA_HOST_DEVICE_FUNC 
        inline double& operator()(
            size_t idx, 
            size_t component_index) const 
        {
          assert(idx < num_elements);
          assert(component_index < Components);
#ifdef WFAOS
          // case 1
          return data[idx*Components + component_index];
#else
          // case 2
          return data[num_elements * component_index + idx];
#endif
        }

    };

  template<int Components>
    ONIKA_HOST_DEVICE_FUNC inline void copyTo(
        const FieldView<Components>& dest_data, 
        int dest_idx, 
        const FieldView<Components>& from_data, 
        int from_idx, 
        int size)
    {
#ifdef WFAOS
      // case 1
      double * from = &from_data(from_idx, 0);
      double * dest = &dest_data(dest_idx, 0);
      int nb_byte = size * Components * sizeof(double);
      std::memcpy(dest, from, nb_byte);
#else
      // case 2
      int nb_byte = size * sizeof(double);
      for(size_t component_index = 0 ; component_index < Components ; component_index++)
      {
        std::memcpy(&dest_data(dest_idx,component_index), &from_data(from_idx,component_index), nb_byte);
      }
#endif
    }
}
