#pragma once
/**
 * @brief A functor for the second step of streaming in the lattice Boltzmann method for XYZ directions.
 */
template <int Q, Traversal Tr>
struct streaming_step2 {
  const int* __restrict__ levels_;  // It contains the traversal level (0 inside, 0 1 Real, 0 1 2 Extend, and 0 1 2 3
                                    // All
  LBMGrid g_;                       // The LBM grid containing the lattice structure and related information.
  const FieldView<Q> f_;            // The field view for the distribution functions.

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx) const {
    if (check_level<Tr>(levels_[idx])) {
      const auto [x, y, z] = g_(idx);
      stencil::for_each<typename LBMScheme<Q>::Coefficients, 1, Q, 2>([&]<typename coeff>(int iLB) {
        const int next_x = x + coeff::ex;
        const int next_y = y + coeff::ey;
        const int next_z = z + coeff::ez;

        if (g_.is_defined(next_x, next_y, next_z)) {
          const int next_idx = g_(next_x, next_y, next_z);
          std::swap(f_(idx, iLB + 1), f_(next_idx, iLB));
        }
      });
    }
  }
  /**
   * @brief Operator for performing the second step of streaming at given coordinates (x, y, z).
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t&& coord) const {
    const int idx = g_(coord.x, coord.y, coord.z);
    this->operator()(idx);
  }
};
}  // namespace hippoLBM
