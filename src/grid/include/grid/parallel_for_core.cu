#pragma once

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>

namespace hippoLBM
{
  using namespace onika::parallel;

  /**
   * @brief Namespace for utilities related to tuple manipulation.
   */
  namespace tuple_helper
  {
    template <size_t... Is> struct index {};
    template <size_t N, size_t... Is> struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};
    template <size_t... Is> struct gen_seq<0, Is...> : index<Is...>{ };
  } // namespace tuple_helper


  template<typename Func, typename... Args>
    struct parallel_for_id_runner
    {
      Func kernel;
      std::tuple<Args...> params; /**< Tuple of parameters to be passed to the kernel function. */
      template <size_t... Is> ONIKA_HOST_DEVICE_FUNC inline void apply(uint64_t i, tuple_helper::index<Is...> indexes) const { kernel(i, std::get<Is>(params)...);}
      ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const { apply(i, tuple_helper::gen_seq<sizeof...(Args)>{}); }
    };

  template<typename Func, typename... Args>
    struct parallel_for_id_traversal_runner
    {
      const int * const __restrict__ traversal;
      Func kernel;
      std::tuple<Args...> params; /**< Tuple of parameters to be passed to the kernel function. */
      parallel_for_id_traversal_runner(const int * const trvl, Func& func, Args ... parameters) : traversal(trvl), kernel(func), params(parameters...) {}
      template <size_t... Is> ONIKA_HOST_DEVICE_FUNC inline void apply(uint64_t i, tuple_helper::index<Is...> indexes) const { kernel(traversal[i], std::get<Is>(params)...);}
      ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const { apply(i, tuple_helper::gen_seq<sizeof...(Args)>{}); }
    };

  template<typename Func, typename... Args>
    struct wrapper_parallel_for_ijk
    {
      Func kernel;
      std::tuple<Args...> params; /**< Tuple of parameters to be passed to the kernel function. */
      wrapper_parallel_for_ijk(Func& func, Args ... parameters) : kernel(func), params(parameters...) {}
      template <size_t... Is> ONIKA_HOST_DEVICE_FUNC inline void apply(int i, int j, int k, tuple_helper::index<Is...> indexes) const { kernel(i, j, k, std::get<Is>(params)...);}
      ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int k) const { apply(i, j, k, tuple_helper::gen_seq<sizeof...(Args)>{}); }
    };

  template<Area A, Traversal Tr, typename Func, typename... Args>
    inline void for_all(LBMGrid& Grid, Func& a_func, Args&&... a_args)
    {
      auto bx = Grid.build_box<A,Tr>();

      for(int k = bx.start(2) ; k <= bx.end(2) ; k++)
        for(int j = bx.start(1) ; j <= bx.end(1) ; j++)
          for(int i = bx.start(0) ; i <= bx.end(0) ; i++)
          {
            a_func(i, j , k, std::forward<Args>(a_args)...);
          }
    }

  template<typename Func, typename... Args>
    inline void for_all(const int * const indexes, int size, Func& func, Args &&...args)
    {
      for(int i = 0 ; i < size ; i++)
      {
        func(i, args...);
      }
    }

  template<Area A, Traversal Tr, typename Func, typename... Args>
    static inline ParallelExecutionWrapper parallel_for_id(LBMGrid& g, Func& func, ParallelExecutionContext *exec_ctx, Args &&...args)
    {
      static_assert(A == Area::Local && Tr == Traversal::All);
      if constexpr (A == Area::Local && Tr == Traversal::All)
      {
        ParallelForOptions opts;
        opts.omp_scheduling = OMP_SCHED_STATIC;
        auto bx = g.build_box<A, Tr>();
        uint64_t size = bx.number_of_points();
        parallel_for_id_runner runner= {func, args...};
        assert(size > 0);
        return parallel_for(size, runner, exec_ctx, opts);
      } 
    }

  template<typename Func, typename... Args>
		static inline ParallelExecutionWrapper parallel_for_simple(int size, Func& func, ParallelExecutionContext *exec_ctx, Args &&...args)
		{
			ParallelForOptions opts;
			opts.omp_scheduling = OMP_SCHED_STATIC;
			parallel_for_id_runner<Func, Args...> runner{func, std::tuple<Args...>(args...)};
			assert(size > 0);
			return parallel_for(size, runner, exec_ctx, opts);
		}

	template<typename Func, typename... Args>
		static inline ParallelExecutionWrapper parallel_for_id(const int * const traversal, const int size, Func& func, ParallelExecutionContext *exec_ctx, Args && ...args)
		{
			ParallelForOptions opts;
			opts.omp_scheduling = OMP_SCHED_STATIC;
			parallel_for_id_traversal_runner runner(traversal, func, args...);
			assert(size > 0);
			return parallel_for(size, runner, exec_ctx, opts);
		}
}

namespace onika
{
	namespace parallel
	{
		template<typename Func, typename... Args> struct ParallelForFunctorTraits<hippoLBM::parallel_for_id_traversal_runner<Func,Args...>>
		{
			static inline constexpr bool RequiresBlockSynchronousCall = false;
			static inline constexpr bool CudaCompatible = ParallelForFunctorTraits<Func>::CudaCompatible;
		};
	}
	namespace parallel
	{
		template<typename Func, typename... Args> struct ParallelForFunctorTraits<hippoLBM::parallel_for_id_runner<Func,Args...>>
		{
			static inline constexpr bool RequiresBlockSynchronousCall = false;
			static inline constexpr bool CudaCompatible = ParallelForFunctorTraits<Func>::CudaCompatible;
		};
	}
}
