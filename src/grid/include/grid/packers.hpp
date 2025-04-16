#pragma once

#include <onika/parallel/parallel_for.h>
#include <grid/box.hpp>
#include <grid/field_view.hpp>

namespace hippoLBM
{
  template<int Components, int DIM>
    struct packer
    {
      FieldView<Components> dst;
      FieldView<Components> src; 
      box<DIM> dst_box; 
      box<DIM> mesh_box;

      ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const
      {
        const auto& inf = dst_box.inf;
        const int dst_idx = compute_idx<DIM>(dst_box, coord.x - inf[0], coord.y-inf[1], coord.z-inf[2]);
        const int src_idx = compute_idx<DIM>(mesh_box, coord.x, coord.y, coord.z);
        copyTo<Components>(dst, dst_idx, src, src_idx, 1);
      }
    };

  template<int Components, int DIM>
    struct unpacker
    {
      FieldView<Components> dst;
      FieldView<Components> src;
      box<DIM> src_box;
      box<DIM> mesh_box;

      ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const
      {
        const auto& inf = src_box.inf;
        const int dst_idx = compute_idx(mesh_box, coord.x , coord.y , coord.z);
        const int src_idx = compute_idx(src_box, coord.x - inf[0], coord.y - inf[1], coord.z - inf[2]);
        copyTo<Components>(dst, dst_idx, src, src_idx, 1);
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<int C, int D> struct ParallelForFunctorTraits< hippoLBM::packer<C,D> >
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };

    template<int C, int D> struct ParallelForFunctorTraits< hippoLBM::unpacker<C,D> >
    {
      static inline constexpr bool RequiresBlockSynchronousCall = false;
      static inline constexpr bool CudaCompatible = true;
    };
  }
}

