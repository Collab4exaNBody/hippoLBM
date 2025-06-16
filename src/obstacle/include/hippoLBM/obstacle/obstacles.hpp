#pragma once

#include <onika/math/basic_types.h>
#include<hippoLBM/obstacle/obstacle.hpp>

namespace hippoLBM
{
	/**
	 * @brief Alias template for a CUDA memory managed vector.
	 * @tparam T The type of elements in the vector.
	 */
	template <typename T> using vector_t = onika::memory::CudaMMVector<T>;


	struct Obstacles
	{
		struct ObstacleTypeAndIndex
		{
			OBSTACLE_TYPE m_type = OBSTACLE_TYPE::UNDEFINED;
			int m_index = -1;
		};

		vector_t<ObstacleTypeAndIndex> m_type_index; /**< Vector storing the types of obstacles. */
		onika::FlatTuple< vector_t<Ball> > m_data;

		inline size_t size() const { return m_type_index.size(); }
		template<size_t obstacle_type>
			inline const auto & get_obstacle_vec() const
			{
				static_assert( obstacle_type != OBSTACLE_TYPE::UNDEFINED );
				return m_data.get_nth_const< obstacle_type >();
			}

		template<size_t obstacle_type>
			inline auto & get_obstacle_vec()
			{
				static_assert( obstacle_type != OBSTACLE_TYPE::UNDEFINED );
				return m_data.get_nth< obstacle_type >();
			}

		template<class T>
			inline const T& get_typed_obstacle(const int idx) const
			{
				constexpr OBSTACLE_TYPE t = get_type<T>();
				static_assert( t != OBSTACLE_TYPE::UNDEFINED );
				const auto & obstacle_vec = m_data.get_nth_const<t>();
				assert( idx>=0 && idx<m_type_index.size() );
				assert( m_type_index[idx].m_type == t );
				assert( m_type_index[idx].m_index >= 0 && m_type_index[idx].m_index < obstacle_vec.size() );
				return obstacle_vec[m_type_index[idx].m_index];
			}

		template<class T>
			inline T& get_typed_obstacle(const int idx)
			{
				constexpr OBSTACLE_TYPE t = get_type<T>();
				static_assert( t != OBSTACLE_TYPE::UNDEFINED );
				auto & obstacle_vec = m_data.get_nth<t>();
				assert( idx>=0 && idx<m_type_index.size() );
				assert( m_type_index[idx].m_type == t );
				assert( m_type_index[idx].m_index >= 0 && m_type_index[idx].m_index < obstacle_vec.size() );
				return obstacle_vec[m_type_index[idx].m_index];
			}

		template<class FuncT>
			inline auto apply(const int idx , const FuncT& func)
			{
				assert( idx>=0 && idx<m_type_index.size() );
				OBSTACLE_TYPE t = m_type_index[idx].m_type;
				assert( t != OBSTACLE_TYPE::UNDEFINED );
				if (t == OBSTACLE_TYPE::BALL)     return func( m_data.get_nth<OBSTACLE_TYPE::BALL    >()[ m_type_index[idx].m_index ] );
				/*
					 else if (t == OBSTACLE_TYPE::WALL)  return func( m_data.get_nth<OBSTACLE_TYPE::WALL >()[ m_type_index[idx].m_index ] );
					 else if (t == OBSTACLE_TYPE::STL_MESH) return func( m_data.get_nth<OBSTACLE_TYPE::STL_MESH>()[ m_type_index[idx].m_index ] );
				 */
				::onika::fatal_error() << "Internal error: unsupported obstacle type encountered"<<std::endl;
				static Ball tmp({0,0,0}, 0);
				return func( tmp );
			}


		template <typename T>
			inline void add(const int idx, T &Obstacle)
			{
				constexpr OBSTACLE_TYPE t = get_type<T>();
				static_assert(t != OBSTACLE_TYPE::UNDEFINED);
				//assert(m_type_index.size() == m_data.size());
				const int size = m_type_index.size();
				if (idx < size) // reallocation
				{
					OBSTACLE_TYPE current_type = type(idx);
					if (current_type != OBSTACLE_TYPE::UNDEFINED)
					{
						::onika::lout << "You are currently removing a obstacle at index " << idx << std::endl;
					//	Obstacle.print();
					}
				}
				else // allocate
				{
					m_type_index.resize(idx+1);
				}
				m_type_index[idx].m_type = t;
				auto & obstacle_vec = get_obstacle_vec<t>();
				m_type_index[idx].m_index = obstacle_vec.size();
				obstacle_vec.push_back( Obstacle );
			}

		/**
		 * @brief Clears the Obstacles collection, removing all obstacles.
		 */
		void clear()
		{
			m_type_index.clear();
			m_data.get_nth<OBSTACLE_TYPE::BALL>().clear();
			/*
				 m_data.get_nth<OBSTACLE_TYPE::WALL>().clear();
				 m_data.get_nth<OBSTACLE_TYPE::STL_MESH>().clear();
			 */
		}
		/**
		 * @brief Returns the type of obstacle at the specified index.
		 * @param idx The index of the obstacle.
		 * @return The type of the obstacle at the specified index.
		 */
		ONIKA_HOST_DEVICE_FUNC
			inline OBSTACLE_TYPE type(size_t idx)
			{
				assert(idx < m_type_index.size());
				return m_type_index[idx].m_type;
			}
	}
;
	// read only proxy for obstacles list
	struct ObstaclesGPUAccessor
	{
		size_t m_nb_obstacles = 0;
		Obstacles::ObstacleTypeAndIndex * const __restrict__ m_type_index = nullptr;
		onika::FlatTuple< Ball * __restrict__ /*, Stl_mesh* __restrict__ */ > m_data = { nullptr /*, nullptr, nullptr,*/ };
		onika::FlatTuple< size_t /*, size_t , size_t ,*/ > m_data_size = { 0 /*, 0, 0,*/ };

		ObstaclesGPUAccessor() = default;
		ObstaclesGPUAccessor(const ObstaclesGPUAccessor &) = default;
		ObstaclesGPUAccessor(ObstaclesGPUAccessor &&) = default;
		inline ObstaclesGPUAccessor(Obstacles& drvs)
			: m_nb_obstacles( drvs.m_type_index.size() )
				, m_type_index( drvs.m_type_index.data() )
				 , m_data( { drvs.m_data.get_nth<0>().data() /*, drvs.m_data.get_nth<1>().data() , drvs.m_data.get_nth<2>().data() , */ } )
				 , m_data_size( { drvs.m_data.get_nth<0>().size() /*, drvs.m_data.get_nth<1>().size() , drvs.m_data.get_nth<2>().size() */ } )
				 {}

		template<class T>
			ONIKA_HOST_DEVICE_FUNC inline T& get_typed_obstacle(const int idx) const
			{
				constexpr OBSTACLE_TYPE t = get_type<T>();
				static_assert( t != OBSTACLE_TYPE::UNDEFINED );
				auto * __restrict__ obstacle_vec = m_data.get_nth_const<t>();
				[[maybe_unused]]const size_t obstacle_vec_size = m_data_size.get_nth_const<t>();
				assert( idx>=0 && idx<m_nb_obstacles );
				assert( m_type_index[idx].m_type == t );
				assert( m_type_index[idx].m_index >= 0 && m_type_index[idx].m_index < obstacle_vec_size );
				return obstacle_vec[m_type_index[idx].m_index];
			}

	};
}
