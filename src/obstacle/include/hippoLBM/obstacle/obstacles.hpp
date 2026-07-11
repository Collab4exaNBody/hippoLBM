#pragma once

#include <onika/math/basic_types.h>

#include <hippoLBM/obstacle/obstacle.hpp>

namespace hippoLBM {
/** @brief Structure to hold information about obstacles. */
struct Obstacles {
  template <typename T>
  using vector_t = onika::memory::CudaMMVector<T>;

  /** @brief Structure to hold the type and index of an obstacle. */
  struct ObstacleTypeAndIndex {
    OBSTACLE_TYPE m_type_ = OBSTACLE_TYPE::UNDEFINED;  // The type of the obstacle.
    int m_index_ = -1;                                 // The index of the obstacle in its respective vector.
  };

  vector_t<ObstacleTypeAndIndex> type_index_;  // Vector to store the type and index.
  onika::FlatTuple<vector_t<Ball>, vector_t<Wall>, vector_t<Quadric> /*, vector_t<Stl_mesh>*/>
      data_;  // Tuple storing vectors of different obstacle types.

  inline size_t size() const { return type_index_.size(); }

  template <size_t obstacle_type>
  inline const auto& get_obstacle_vec() const {
    static_assert(obstacle_type != OBSTACLE_TYPE::UNDEFINED);
    return data_.get_nth_const<obstacle_type>();
  }

  template <size_t obstacle_type>
  inline auto& get_obstacle_vec() {
    static_assert(obstacle_type != OBSTACLE_TYPE::UNDEFINED);
    return data_.get_nth<obstacle_type>();
  }

  template <class T>
  inline const T& get_typed_obstacle(const int idx) const {
    constexpr OBSTACLE_TYPE t = get_type<T>();
    static_assert(t != OBSTACLE_TYPE::UNDEFINED);
    const auto& obstacle_vec = data_.get_nth_const<t>();
    assert(idx >= 0 && idx < type_index_.size());
    assert(type_index_[idx].m_type_ == t);
    assert(type_index_[idx].m_index_ >= 0 && type_index_[idx].m_index_ < obstacle_vec.size());
    return obstacle_vec[type_index_[idx].m_index_];
  }

  template <class T>
  inline T& get_typed_obstacle(const int idx) {
    constexpr OBSTACLE_TYPE t = get_type<T>();
    static_assert(t != OBSTACLE_TYPE::UNDEFINED);
    auto& obstacle_vec = data_.get_nth<t>();
    assert(idx >= 0 && idx < type_index_.size());
    assert(type_index_[idx].m_type_ == t);
    assert(type_index_[idx].m_index_ >= 0 && type_index_[idx].m_index_ < obstacle_vec.size());
    return obstacle_vec[type_index_[idx].m_index_];
  }

  template <class FuncT>
  inline auto apply(const int idx, const FuncT& func) {
    assert(idx >= 0 && idx < type_index_.size());
    OBSTACLE_TYPE t = type_index_[idx].m_type_;
    assert(t != OBSTACLE_TYPE::UNDEFINED);
    if (t == OBSTACLE_TYPE::BALL) {
      return func(data_.get_nth<OBSTACLE_TYPE::BALL>()[type_index_[idx].m_index_]);
    } else if (t == OBSTACLE_TYPE::WALL) {
      return func(data_.get_nth<OBSTACLE_TYPE::WALL>()[type_index_[idx].m_index_]);
    } else if (t == OBSTACLE_TYPE::QUADRIC) {
      return func(data_.get_nth<OBSTACLE_TYPE::QUADRIC>()[type_index_[idx].m_index_]);
    }
    /*
             else if (t == OBSTACLE_TYPE::STL_MESH) return func( data_.get_nth<OBSTACLE_TYPE::STL_MESH>()[
       type_index_[idx].m_index_ ] );
     */
    ::onika::fatal_error() << "Internal error: unsupported obstacle type encountered" << std::endl;
    static Ball tmp({0, 0, 0}, 0);
    return func(tmp);
  }

  template <typename T>
  inline void add(const int idx, T& Obstacle) {
    constexpr OBSTACLE_TYPE t = get_type<T>();
    static_assert(t != OBSTACLE_TYPE::UNDEFINED);
    // assert(type_index_.size() == data_.size());
    const int size = type_index_.size();
    if (idx < size)  // reallocation
    {
      OBSTACLE_TYPE current_type = type(idx);
      if (current_type != OBSTACLE_TYPE::UNDEFINED) {
        ::onika::lout << "You are currently removing a obstacle at index " << idx << std::endl;
        //	Obstacle.print();
      }
    } else  // allocate
    {
      type_index_.resize(idx + 1);
    }
    type_index_[idx].m_type_ = t;
    auto& obstacle_vec = get_obstacle_vec<t>();
    type_index_[idx].m_index_ = obstacle_vec.size();
    obstacle_vec.push_back(Obstacle);
  }

  /**
   * @brief Clears the Obstacles collection, removing all obstacles.
   */
  void clear() {
    type_index_.clear();
    data_.get_nth<OBSTACLE_TYPE::BALL>().clear();
    data_.get_nth<OBSTACLE_TYPE::WALL>().clear();
    data_.get_nth<OBSTACLE_TYPE::QUADRIC>().clear();

    /*
             data_.get_nth<OBSTACLE_TYPE::STL_MESH>().clear();
     */
  }
  /**
   * @brief Returns the type of obstacle at the specified index.
   * @param idx The index of the obstacle.
   * @return The type of the obstacle at the specified index.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline OBSTACLE_TYPE type(size_t idx) {
    assert(idx < type_index_.size());
    return type_index_[idx].m_type_;
  }
};
// read only proxy for obstacles list
struct ObstaclesGPUAccessor {
  size_t m_nb_obstacles_ = 0;
  Obstacles::ObstacleTypeAndIndex* const __restrict__ type_index_ = nullptr;
  onika::FlatTuple<Ball* __restrict__, Wall* __restrict__, Quadric* __restrict__ /*, Stl_mesh* __restrict__ */> data_ =
      {nullptr, nullptr /*, nullptr,*/};
  onika::FlatTuple<size_t, size_t, size_t /*, size_t ,*/> data_size_ = {0, 0, 0 /*, 0,*/};

  ObstaclesGPUAccessor() = default;
  ObstaclesGPUAccessor(const ObstaclesGPUAccessor&) = default;
  ObstaclesGPUAccessor(ObstaclesGPUAccessor&&) = default;
  inline ObstaclesGPUAccessor(Obstacles& drvs)
      : m_nb_obstacles_(drvs.type_index_.size()),
        type_index_(drvs.type_index_.data()),
        data_({drvs.data_.get_nth<0>().data(), drvs.data_.get_nth<1>().data(),
               drvs.data_.get_nth<2>().data() /*, drvs.data_.get_nth<3>().data() , */}),
        data_size_({drvs.data_.get_nth<0>().size(), drvs.data_.get_nth<1>().size(),
                    drvs.data_.get_nth<2>().size() /*, drvs.data_.get_nth<3>().size() */}) {}

  template <class T>
  ONIKA_HOST_DEVICE_FUNC inline T& get_typed_obstacle(const int idx) const {
    constexpr OBSTACLE_TYPE t = get_type<T>();
    static_assert(t != OBSTACLE_TYPE::UNDEFINED);
    auto* __restrict__ obstacle_vec = data_.get_nth_const<t>();
    [[maybe_unused]] const size_t obstacle_vec_size = data_size_.get_nth_const<t>();
    assert(idx >= 0 && idx < m_nb_obstacles_);
    assert(type_index_[idx].m_type_ == t);
    assert(type_index_[idx].m_index_ >= 0 && type_index_[idx].m_index_ < obstacle_vec_size);
    return obstacle_vec[type_index_[idx].m_index_];
  }
};
}  // namespace hippoLBM
