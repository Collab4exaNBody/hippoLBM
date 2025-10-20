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

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>

namespace hippoLBM
{
  struct reduce_thread_local_t {};
  struct reduce_thread_block_t {};
  struct reduce_global_t {};
  using namespace onika::parallel;

  /**
   * @struct ReduceTFunctor
   * @brief Functor for reducing data in a parallel execution.
   *
   * @tparam T The data type.
   * @tparam FuncT The function type to perform the reduction.
   * @tparam ResultT The result type that stores the reduction result.
   */
  template <class FuncT, class ResultT> struct ReduceFuncT
  {
    const FuncT m_func;     /**< Functor that defines how reduction is performed. */
    ResultT *m_reduced_val; /**< Pointer to the result of the reduction. */
    const int * const idxs;       /**< Conatains lattice indexes */

    /**
     * @brief Operator to perform the reduction.
     * @param i The index of the data to reduce.
     */
    ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t idx) const
    {
      int i = idxs[idx];
      ResultT local_val = ResultT();
      m_func(local_val, i, reduce_thread_local_t{});

      ONIKA_CU_BLOCK_SHARED onika::cuda::UnitializedPlaceHolder<ResultT> team_val_place_holder;
      ResultT &team_val = team_val_place_holder.get_ref();

      if (ONIKA_CU_THREAD_IDX == 0)
      {
        team_val = local_val;
      }
      ONIKA_CU_BLOCK_SYNC();

      if (ONIKA_CU_THREAD_IDX != 0)
      {
        m_func(team_val, local_val, reduce_thread_block_t{});
      }
      ONIKA_CU_BLOCK_SYNC();

      if (ONIKA_CU_THREAD_IDX == 0)
      {
        m_func(*m_reduced_val, team_val, reduce_global_t{});
      }
    }
  };

  template <class ResultT>
  struct ResetScratch
  {
    ResultT *m_reduced_val; /**< Pointer to the result of the reduction. */
    ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t idx) const
    {
      *m_reduced_val = ResultT{}; 
    }
  };
}


namespace hippoLBM
{
  using namespace onika::parallel;
  using namespace onika::memory;

	template <class ResultT>
		static inline ParallelExecutionWrapper reset_scratch(
				CudaMMVector<ResultT>& result,
				ParallelExecutionContext *exec_ctx
				)
		{
      if( result.size() != 1 ) result.resize(1);
			ResetScratch func = {result.data()};
			return parallel_for(1, func, exec_ctx, ParallelForOptions{});
		}

	template <class FuncT, class ResultT, class RegionT>
		static inline ParallelExecutionWrapper local_reduce(
				FuncT& func, 
				CudaMMVector<ResultT>& result, 
				ParallelExecutionContext *exec_ctx, 
				RegionT& region, 
				Traversal Tr = Traversal::Real)
		{
			// get traversal
			auto [indexes, size] = get_traversal(region, Tr);
			// parallel_for options
			ParallelForOptions opts;
			opts.omp_scheduling = OMP_SCHED_STATIC;
			// reduce functor
			ReduceFuncT<FuncT&, ResultT> rfunc = {func, result.data(), indexes};
			return parallel_for(size, rfunc, exec_ctx, opts);
		}

	inline void local_reduce_sync() { ONIKA_CU_DEVICE_SYNCHRONIZE(); }
}

namespace onika
{
	namespace parallel
	{
		template <class FuncT, class ResultT> struct ParallelForFunctorTraits<hippoLBM::ReduceFuncT<FuncT,ResultT>>
		{
			static inline constexpr bool CudaCompatible = FuncT::CudaCompatible;
		};
		template <class ResultT> struct ParallelForFunctorTraits<hippoLBM::ResetScratch<ResultT>>
		{
			static inline constexpr bool CudaCompatible = true;
		};
	} // namespace parallel
} // namespace onika
