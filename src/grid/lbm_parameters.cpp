#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <grid/make_variant_operator.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/grid/lbm_domain.hpp>


namespace hippoLBM
{
  using namespace onika;
  using namespace scg;
  using namespace onika::math;

  template<int Q>
    class LBMParametersOp : public OperatorNode
  {
    public:
      ADD_SLOT( LBMDomain<Q>, lbm_domain, INPUT, REQUIRED);
      ADD_SLOT( Vec3d, Fext, INPUT, Vec3d{0,0,0});
      ADD_SLOT( double, celerity, INPUT, 1);
      ADD_SLOT( double, nuth, INPUT, 1e-4);
      ADD_SLOT( double, avg_rho, INPUT, 1000.0);

      ADD_SLOT( LBMParameters, Params, OUTPUT);
      ADD_SLOT( double , dtLB, OUTPUT);

      inline void execute () override final
      {
        double Dx = lbm_domain->dx();
        LBMParameters params;
        params.Fext = *Fext;
        params.celerity = *celerity;
        params.dtLB = Dx / params.celerity;
        params.nuth = *nuth;
        params.nu = params.nuth * params.dtLB / (Dx * Dx);
        params.tau = 3. * params.nu + 0.5;
        params.avg_rho = *avg_rho;
        params.print();
        *dtLB = params.dtLB;
        *Params = params;
      }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(lbm_parameters)
  {
    OperatorNodeFactory::instance()->register_factory( "lbm_parameters", make_variant_operator<LBMParametersOp>);
  }
}

