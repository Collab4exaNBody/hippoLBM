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

namespace hippoLBM {
/** @brief Lattice Boltzmann Method parameters. */
struct LBMParameters {
  onika::math::Vec3d Fext_;  // External forces
  double celerity_;          // Netword celerity
  double dtLB_;              // Celerity time step
  double nuth_;              // Viscosity in real unit
  double nu_;                // Viscoty in LB unit
  double tau_;               // relexation time
  double avg_rho_;           // Average density in real unit, 1 in LB unit

  LBMParameters() {}

  void print() {
    using onika::lout;
    lout << "=================================" << std::endl;
    lout << "= LBM Parameters" << std::endl;
    lout << "= External forces Fext:           [" << Fext_ << "]" << std::endl;
    lout << "= Network celerity celerity:      " << celerity_ << std::endl;
    lout << "= Celerity time step dtLB:        " << dtLB_ << " [dx / celerity]" << std::endl;
    lout << "= Viscosity nuth:                 " << nuth_ << std::endl;
    lout << "= Viscosity with lattice unit nu: " << nu_ << " [nuth * dtLB / (dx²)]" << std::endl;
    lout << "= Relaxation time tau:            " << tau_ << " [3nu + 0.5]" << std::endl;
    lout << "= Average Rho avg_rho:            " << avg_rho_ << std::endl;
    lout << "=================================" << std::endl;
  }
};
}  // namespace hippoLBM
