/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */


#pragma once

namespace hippoLBM
{
  template<int Components>
    struct FieldView
    {
      double * const data = nullptr;
      uint64_t num_elements = 0;

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

      inline void operator=(FieldView<Components>& fv)
      {
        this->data = fv.data;
        this->num_elements = fv.num_elements;
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
