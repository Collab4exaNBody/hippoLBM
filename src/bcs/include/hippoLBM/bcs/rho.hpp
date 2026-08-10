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

#include <hippoLBM/grid/field_view.hpp>
#define FLUIDE_ -1  // TODO: move this to a more appropriate place (from LBMDEM3D code)

namespace hippoLBM {

/** @brief Struct for handling rho boundary conditions at x=0. */
template <int Q>
struct rho_x_0 {};

/** @brief Struct for handling rho boundary conditions at x=lx. */
template <int Q>
struct rho_x_l {};

/**
 * @brief A functor for handling rho boundary conditions at x=lx in the lattice Boltzmann method.
 */
template <>
struct rho_x_l<19> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double ux,
                                                const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
    }
  }
};

/**
 * @brief A functor for handling rho boundary conditions at x=0 in the lattice Boltzmann method.
 */
template <>
struct rho_x_0<19> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double& ux,
                                                const double& uy, const double& uz) const {
    if (obst[idx] == FLUIDE_) {
    }
  }
};

/** @brief Struct for handling rho boundary conditions at y=0. */
template <int Q>
struct rho_y_0 {};

/** @brief Struct for handling rho boundary conditions at y=ly. */
template <int Q>
struct rho_y_l {};

/**
 * @brief A functor for handling rho boundary conditions at y=ly in the lattice Boltzmann method.
 */
template <>
struct rho_y_l<19> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double ux,
                                                const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
    }
  }
};

/**
 * @brief A functor for handling rho boundary conditions at y=0 in the lattice Boltzmann method.
 */
template <>
struct rho_y_0<19> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double& ux,
                                                const double& uy, const double& uz) const {
    if (obst[idx] == FLUIDE_) {
    }
  }
};

/** @brief Struct for handling rho boundary conditions at z=0. */
template <int Q>
struct rho_z_0 {};
/** @brief Struct for handling rho boundary conditions at z=lz. */
template <int Q>
struct rho_z_l {};

/**
 * @brief A functor for handling rho boundary conditions at z=lz in the lattice Boltzmann method.
 */
template <>
struct rho_z_l<19> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double ux,
                                                const double uy, const double uz) const {
    if (obst[idx] == FLUIDE_) {
    }
  }
};

/**
 * @brief A functor for handling rho boundary conditions at z=0 in the lattice Boltzmann method.
 */
template <>
struct rho_z_0<19> {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int idx, int* const obst, const FieldView<19>& f, const double& ux,
                                                const double& uy, const double& uz) const {
    if (obst[idx] == FLUIDE_) {
    }
  }
};
}  // namespace hippoLBM

namespace onika {
namespace parallel {
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::rho_x_0<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::rho_x_l<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::rho_y_0<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::rho_y_l<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::rho_z_0<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <int Q>
struct ParallelForFunctorTraits<hippoLBM::rho_z_l<Q>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
