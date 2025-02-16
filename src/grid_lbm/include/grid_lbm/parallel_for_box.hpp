#pragma once

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>


namespace hippoLBM
{
  using namespace onika::parallel;


  template<typename Func, typename... Args>
    static inline void parallel_for_box(box<3>& bx, Func& func, Args &&...args)
    //static inline void parallel_for_box(box<3>& bx, Func& func, ParallelExecutionContext *exec_ctx, Args &&...args)
    {
      //ParallelForOptions opts;
      //opts.omp_scheduling = OMP_SCHED_STATIC;
      //wrapper_parallel_for_ijk WrapperForAllIJK = {func, args...};

#pragma omp parallel for collapse(3)
      for(int k = bx.start(2) ; k <= bx.end(2) ; k++)
	for(int j = bx.start(1) ; j <= bx.end(1) ; j++)
	  for(int i = bx.start(0) ; i <= bx.end(0) ; i++)
	  {
	    func(i, j, k, args...);
	    //WrapperForAllIJK(i, j, k);
	  }
    }

#ifdef ONIKA_CUDA_VERSION
  template<typename Func, typename... Args>
    __global__
    void apply_grid(box<3> bx, Func func, Args ...args)
    {
      const int x = threadIdx.x + (blockDim.x*blockIdx.x) + bx.start(0);
      const int y = threadIdx.y + (blockDim.y*blockIdx.y) + bx.start(1);
      const int z = threadIdx.z + (blockDim.z*blockIdx.z) + bx.start(2);
      if( x <= bx.end(0) && y <= bx.end(1) && z <= bx.end(2))
      {
	func(x, y, z, args...);
      }
    }

  template<typename Func, typename... Args>
    static inline void cuda_parallel_for_box(box<3>& bx, Func& func, Args&& ...args)
    {
      const int size_block = 32;
      const int size_x = bx.end(0) - bx.start(0) + 1;
      const int size_y = bx.end(1) - bx.start(1) + 1;
      const int size_z = bx.end(2) - bx.start(2) + 1;
      const int nBlockX = (size_x + size_block - 1) / size_block;
      const int nBlockY = (size_y + size_block - 1) / size_block;
      dim3 dimBlock(nBlockX, nBlockY, size_z + 1);
      dim3 BlockSize(size_block, size_block, 1);
      //if(bx.end(0) - bx.start(0) + 1 < size_block) BlockSize.x = bx.end(0) - bx.start(0) + 1;
      //if(bx.end(1) - bx.start(1) + 1 < size_block) BlockSize.y = bx.end(1) - bx.start(1) + 1;
      //apply_grid<<<dimBlock, size_block>>>(bx, func, args...);
      apply_grid<<<dimBlock, BlockSize>>>(bx, func, args...);
    }
#endif

}

