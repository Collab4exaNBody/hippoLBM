#pragma once

namespace hipoLBM
{
  using namespace onika;

  struct LBMParameters
  {
    onika::math::Vec3d Fext;
    double celerity; // Netword celerity
    double dtLB;     // Celerity time step
    double nuth;     // Viscosity in real unit
    double nu;       // Viscoty in LB unit
    double tau;     // relexation time
    double avg_rho;  // Average density in real unit, 1 in LB unit 

    LBMParameters() {}

    void print()
    {
      lout << "=================================" << std::endl;
      lout << "= LBM Parameters" << std::endl;
      lout << "= External forces Fext:           [" << Fext << "]" << std::endl;
      lout << "= Network celerity celerity:      " << celerity << std::endl;
      lout << "= Celerity time step dtLB:        " << dtLB << " [dx / celerity]" << std::endl;
      lout << "= Viscosity nuth:                 " << nuth << std::endl;
      lout << "= Viscosity with lattice unit nu: " << nu << " [nuth * dtLB / (dx²)]" << std::endl;
      lout << "= Relaxation time tau:            " << tau << " [3nu + 0.5]" << std::endl;
      lout << "= Average Rho avg_rho:            " << avg_rho << std::endl;
      lout << "=================================" << std::endl;
    }
  };
}
