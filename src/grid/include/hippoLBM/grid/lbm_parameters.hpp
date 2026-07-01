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

#include <iomanip>

namespace hippoLBM {
enum UNITS { PHYSICAL_UNITS, LBM_UNITS, ADIMENSIONAL };

/** @brief Human-readable name of a unit system, for display purposes. */
inline const char* units_name(UNITS u) {
  switch (u) {
    case PHYSICAL_UNITS:
      return "physical units";
    case LBM_UNITS:
      return "LBM units";
    case ADIMENSIONAL:
      return "adimensional";
  }
  return "unknown units";
}

/** @brief Lattice Boltzmann Method parameters. */
struct LBMParameters {
  static constexpr UNITS Fext_units_ = LBM_UNITS;
  static constexpr UNITS celerity_units_ = PHYSICAL_UNITS;
  static constexpr UNITS dtLB_units_ = PHYSICAL_UNITS;
  static constexpr UNITS nuth_units_ = PHYSICAL_UNITS;
  static constexpr UNITS nu_units_ = LBM_UNITS;
  static constexpr UNITS tau_units_ = ADIMENSIONAL;
  static constexpr UNITS avg_rho_units_ = PHYSICAL_UNITS;

  onika::math::Vec3d Fext_;  // External forces
  double celerity_;          // Netword celerity
  double dtLB_;              // Celerity time step
  double nuth_;              // Viscosity in real unit
  double nu_;                // Viscoty in LB unit
  double tau_;               // relexation time
  double avg_rho_;           // Average density in real unit, 1 in LB unit

  LBMParameters() {}

  // fill nuth before
  void define_by_dt(const double dtLB, const double dx) {
    dtLB_ = dtLB;
    tau_ = 3 * nuth_ * dtLB_ / (dx * dx) + 0.5;
    nu_ = (tau_ - 0.5) / 3.0;  // looks like <-> nu_ = nuth_ * dtLB_ / (dx * dx);

    celerity_ = dx / dtLB_;
  }

  // fill nuth before
  void define_by_tau(const double tau, const double dx) {
    tau_ = tau;
    dtLB_ = dx * dx * (tau_ - 0.5) / (3.0 * nuth_);
    nu_ = nuth_ * dtLB_ / (dx * dx);
    celerity_ = dx / dtLB_;
  }

  void define_by_c(const double c, const double dx) {
    celerity_ = c;
    dtLB_ = dx / celerity_;
    nu_ = nuth_ * dtLB_ / (dx * dx);
    tau_ = 3. * nu_ + 0.5;
  }

  void print();
};

template <UNITS To>
inline double convert_velocity(double value, const LBMParameters& params);

template <>
inline double convert_velocity<LBM_UNITS>(double value, const LBMParameters& params) {
  return value / params.celerity_;
}

template <>
inline double convert_velocity<PHYSICAL_UNITS>(double value, const LBMParameters& params) {
  return value * params.celerity_;
}

template <UNITS To>
inline double convert_viscosity(double value, const LBMParameters& params);

template <>
inline double convert_viscosity<LBM_UNITS>(double value, const LBMParameters& params) {
  const double dx = params.celerity_ * params.dtLB_;
  return value * params.dtLB_ / (dx * dx);
}

template <>
inline double convert_viscosity<PHYSICAL_UNITS>(double value, const LBMParameters& params) {
  const double dx = params.celerity_ * params.dtLB_;
  return value * (dx * dx) / params.dtLB_;
}

template <UNITS To>
inline onika::math::Vec3d convert_force(const onika::math::Vec3d& value, const LBMParameters& params);

template <>
inline onika::math::Vec3d convert_force<LBM_UNITS>(const onika::math::Vec3d& value, const LBMParameters& params) {
  const double dx = params.celerity_ * params.dtLB_;
  return value * params.dtLB_ * params.dtLB_ / dx;
}

template <>
inline onika::math::Vec3d convert_force<PHYSICAL_UNITS>(const onika::math::Vec3d& value, const LBMParameters& params) {
  const double dx = params.celerity_ * params.dtLB_;
  return value * dx / (params.dtLB_ * params.dtLB_);
}

inline void LBMParameters::print() {
  using onika::lout;
  lout << std::setprecision(4);
  lout << "=================================" << std::endl;
  lout << "= LBM Parameters" << std::endl;
  lout << "= External forces Fext:           [" << convert_force<PHYSICAL_UNITS>(Fext_, *this) << "] (physical) / ["
       << Fext_ << "] (LBM)" << std::endl;
  lout << "= Network celerity celerity:      " << celerity_ << " (physical) / "
       << convert_velocity<LBM_UNITS>(celerity_, *this) << " (LBM)" << std::endl;
  lout << "= Celerity time step dtLB:        " << dtLB_ << " (physical) [dx / celerity] / 1 (LBM, by convention)"
       << std::endl;
  lout << "= Viscosity nuth / nu:            " << nuth_ << " (physical) / " << nu_
       << " (LBM) [nu = nuth * dtLB / (dx²)]" << std::endl;
  lout << "= Relaxation time tau:            " << tau_ << " [3nu + 0.5] (" << units_name(tau_units_) << ")"
       << std::endl;
  lout << "= Average Rho avg_rho:            " << avg_rho_ << " (physical) / 1 (LBM, by convention)" << std::endl;
  lout << "=================================" << std::endl;
  lout << std::setprecision(6);  // restore the default precision
}

}  // namespace hippoLBM
