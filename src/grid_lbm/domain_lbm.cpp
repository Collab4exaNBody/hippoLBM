#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

#include <onika/math/basic_types_yaml.h>
#include <grid_lbm/domain_lbm.hpp>
#include <grid_lbm/comm.hpp>


namespace hippoLBM
{
	using namespace onika;
	using namespace scg;
  using onika::math::AABB;
  using BoolVector = std::vector<bool>;

	template<int Q>
		class InitDomainLBM : public OperatorNode
	{
		public:
			ADD_SLOT( domain_lbm<Q>, DomainQ, OUTPUT);
			ADD_SLOT( BoolVector, periodic   , INPUT , OPTIONAL );
			ADD_SLOT( double, resolution, INPUT_OUTPUT, REQUIRED, DocString{"Resolution"});
			ADD_SLOT( AABB, bounds, INPUT_OUTPUT, REQUIRED, DocString{"Domain's bounds"});
			ADD_SLOT( double, dx, OUTPUT, DocString{"Space step"});
      ADD_SLOT( MPI_Comm, mpi, INPUT , MPI_COMM_WORLD);

			inline void execute () override final
			{
				static_assert(DIM == 3);
				AABB& bx = *bounds;
				double& res = *resolution;

				BoolVector& pbc = *periodic; // periodic boundary conditions
        MPI_Comm comm = *mpi;

        double Dx;
        Dx = (bx.bmax.x - bx.bmin.x) / res;
        Dx = std::min( Dx, (bx.bmax.y - bx.bmin.y) / res );
        Dx = std::min( Dx, (bx.bmax.z - bx.bmin.z) / res );
        *dx = Dx;
        lout << "The space step dx is: " << Dx << std::endl;
        int lx = (bx.bmax.x - bx.bmin.x) / Dx; 
        int ly = (bx.bmax.y - bx.bmin.y) / Dx; 
        int lz = (bx.bmax.z - bx.bmin.z) / Dx; 

        bx.bmax.x = bx.bmin.x + lx * Dx;
        bx.bmax.y = bx.bmin.y + ly * Dx;
        bx.bmax.z = bx.bmin.z + lz * Dx;

        //lout << "The Domain Boundaries are: [" << bx.bmin << "," << bx.bmax << "]" << std::endl;

				constexpr int ghost_layer = 2; // should be 1 lbm and 2 for dem lbm

				// init periodic conditions 
				int * periods = new int[DIM];
				for(size_t dim = 0 ; dim < pbc.size() ; dim++) 
				{ 
					if(pbc[dim]) periods[dim] = 1; 
					else periods[dim] = 0; 
				}
				int ndims[DIM];
				MPI_Comm MPI_COMM_CART;
				int mpi_rank, mpi_size;
				MPI_Comm_rank(comm, &mpi_rank);
				MPI_Comm_size(comm, &mpi_size);


				for (int dim = 0; dim < DIM; dim++) {
					ndims[dim] = 0; // do not remove it
				}

				MPI_Dims_create(mpi_size, DIM, ndims);
				MPI_Cart_create(comm, DIM, ndims, periods, true, &MPI_COMM_CART);

				int coord[DIM];
				MPI_Cart_coords(MPI_COMM_CART, mpi_rank, DIM, coord);
				int3d domain_size = {lx, ly, lz};
				int3d subdomain, relative_position, offset;
				for (int dim = 0; dim < DIM; dim++) {
					double size = domain_size[dim] / ndims[dim];
					if (coord[dim] == ndims[dim] - 1)
						subdomain[dim] = domain_size[dim] - coord[dim] * size;
					else
						subdomain[dim] = size;
					offset[dim] = -ghost_layer + coord[dim] * size;
				}


				/** create the box that contains the local grid indexes **/
				box<3> local_box = {{0, 0, 0}, {subdomain[0] + 2 * ghost_layer - 1, subdomain[1] + 2 * ghost_layer - 1, subdomain[2] + 2 * ghost_layer - 1}};
				/** create the extended real box to skip point that does not exist ( next points ) **/
				box<3> ext = local_box;
				for (int dim = 0; dim < DIM; dim++) {
					if (periods[dim] == 0) // not periodic
					{
						if (coord[dim] == 0)
							ext.inf[dim] += ghost_layer;
						if (coord[dim] == ndims[dim] - 1)
							ext.sup[dim] -= ghost_layer;
					}
				}

				// set grid
				grid<DIM> g;
				g.set_box(local_box);
				g.set_ext(ext);
				g.set_offset({offset[0], offset[1], offset[2]});
				g.set_ghost_layer(ghost_layer);
				g.set_dx(Dx);

				// get data
				onika::math::IJK MPI_coord = {coord[0], coord[1], coord[2]};
				onika::math::IJK MPI_grid_size = {ndims[0], ndims[1], ndims[2]};

				ghost_manager<Q, DIM> manager;
        // fill ghost comm
				int default_tag = 100;
				for (int i = 0; i < 27; i++) {
					int idx = i;
					for (int dim = 0; dim < 3; dim++) {
						int d = idx % 3; // keep 3 first bits
						relative_position[dim] = d - 1;
						idx = (idx - d) / 3; // shift 3 bits
					}
					if (relative_position[0] == 0 && relative_position[1] == 0 && relative_position[2] == 0)
						continue;

					auto [not_exist, coord_neig] = build_comm<DIM>(relative_position, coord, domain_size, periods, ndims);
					if (not_exist == false)
						continue;
					int neig;
					MPI_Cart_rank(MPI_COMM_CART, coord_neig.data(), &neig);

					auto [send_box, recv_box] = build_boxes<DIM>(relative_position, g);

					// need to fix send_box in periodic case if the receiver is the sender.
					if(neig == mpi_rank) // 
					{
						send_box = fix_box_with_periodicity<DIM>(relative_position, g);
						int send_tag = (relative_position[0] + 1) + 3 * (relative_position[1] + 1) * 3 + (relative_position[2] + 1) * 9;
						int recv_tag = (relative_position[0] + 1) + 3 * (relative_position[1] + 1) * 3 + (relative_position[2] + 1) * 9;
						hippoLBM::comm<Q, DIM> send(neig, default_tag + neig * 27 + send_tag, send_box);
						hippoLBM::comm<Q, DIM> recv(neig, default_tag + mpi_rank * 27 + recv_tag, recv_box);
						assert(recv.get_size() == send.get_size());
						if (recv.get_size() != 0)
							manager.add_comm(send, recv);
					}
					else
					{

						int send_tag = (relative_position[0] + 1) + 3 * (relative_position[1] + 1) * 3 + (relative_position[2] + 1) * 9;
						int recv_tag = (2 - (relative_position[0] + 1)) + 3 * (2 - (relative_position[1] + 1)) * 3 + (2 - (relative_position[2] + 1)) * 9;
						hippoLBM::comm<Q, DIM> send(neig, default_tag + neig * 27 + send_tag, send_box);
						hippoLBM::comm<Q, DIM> recv(neig, default_tag + mpi_rank * 27 + recv_tag, recv_box);
						assert(recv.get_size() == send.get_size());
						if (recv.get_size() != 0)
							manager.add_comm(send, recv);
					}
				}

				domain_lbm<Q> domain(manager, local_box, g, bx, domain_size, MPI_coord, MPI_grid_size);
				*DomainQ = domain;

        //domain.m_ghost_manager.debug_print_comm();
			}
	};

	//template<int Q> using InitDomainLBMQ = InitDomainLBM<Q>;
	using InitDomainLBM3D19Q = InitDomainLBM<19>;

	// === register factories ===  
	ONIKA_AUTORUN_INIT(parallel_for_benchmark)
	{
		//OperatorNodeFactory::instance()->register_factory( "domain", make_compatible_operator<InitDomainLBMQ>);
		OperatorNodeFactory::instance()->register_factory( "domain", make_compatible_operator<InitDomainLBM3D19Q>);
	}
}

