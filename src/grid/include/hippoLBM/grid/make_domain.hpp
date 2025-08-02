#include <mpi.h>
#include <onika/math/basic_types_yaml.h>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>

namespace hippoLBM
{
	using namespace onika::math;
	template<int Q>
		LBMDomain<Q> make_domain(AABB& bounds, double resolution, std::vector<bool>& periodic, MPI_Comm& comm)
		{
      constexpr int DIM = 3;
			double Dx;
			Dx = (bounds.bmax.x - bounds.bmin.x) / resolution;
			Dx = std::min( Dx, (bounds.bmax.y - bounds.bmin.y) / resolution );
			Dx = std::min( Dx, (bounds.bmax.z - bounds.bmin.z) / resolution );
			int lx = (bounds.bmax.x - bounds.bmin.x) / Dx; 
			int ly = (bounds.bmax.y - bounds.bmin.y) / Dx; 
			int lz = (bounds.bmax.z - bounds.bmin.z) / Dx; 

			bounds.bmax.x = bounds.bmin.x + lx * Dx;
			bounds.bmax.y = bounds.bmin.y + ly * Dx;
			bounds.bmax.z = bounds.bmin.z + lz * Dx;

			//lout << "The Domain Boundaries are: [" << bounds.bmin << "," << bounds.bmax << "]" << std::endl;

			constexpr int ghost_layer = 2; // should be 1 lbm and 2 for dem lbm

			// init periodic conditions 
			int * periods = new int[DIM];
			for(size_t dim = 0 ; dim < periodic.size() ; dim++) 
			{ 
				if(periodic[dim]) periods[dim] = 1; 
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
			Box3D local_box = {{0, 0, 0}, {subdomain[0] + 2 * ghost_layer - 1, subdomain[1] + 2 * ghost_layer - 1, subdomain[2] + 2 * ghost_layer - 1}};
			/** create the extended real box to skip point that does not exist ( next points ) **/
			Box3D ext = local_box;
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
			LBMGrid g;
			g.set_box(local_box);
			g.set_ext(ext);
			g.set_offset({offset[0], offset[1], offset[2]});
			g.set_ghost_layer(ghost_layer);
			g.set_dx(Dx);

			// get data
			onika::math::IJK MPI_coord = {coord[0], coord[1], coord[2]};
			onika::math::IJK MPI_grid_size = {ndims[0], ndims[1], ndims[2]};

			LBMGhostManager<Q> manager;
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

				auto [not_exist, coord_neig] = build_comm(relative_position, coord, domain_size, periods, ndims);
				if (not_exist == false)
					continue;
				int neig;
				MPI_Cart_rank(MPI_COMM_CART, coord_neig.data(), &neig);

				auto [send_box, recv_box] = build_boxes(relative_position, g);

				// need to fix send_box in periodic case if the receiver is the sender.
				if(neig == mpi_rank) // 
				{
					send_box = fix_box_with_periodicity(relative_position, g);
					int send_tag = (relative_position[0] + 1) + 3 * (relative_position[1] + 1) * 3 + (relative_position[2] + 1) * 9;
					int recv_tag = (relative_position[0] + 1) + 3 * (relative_position[1] + 1) * 3 + (relative_position[2] + 1) * 9;
					hippoLBM::LBMComm<Q> send(neig, default_tag + neig * 27 + send_tag, send_box);
					hippoLBM::LBMComm<Q> recv(neig, default_tag + mpi_rank * 27 + recv_tag, recv_box);
					assert(recv.get_size() == send.get_size());
					if (recv.get_size() != 0)
						manager.add_comm(send, recv);
				}
				else
				{

					int send_tag = (relative_position[0] + 1) + 3 * (relative_position[1] + 1) * 3 + (relative_position[2] + 1) * 9;
					int recv_tag = (2 - (relative_position[0] + 1)) + 3 * (2 - (relative_position[1] + 1)) * 3 + (2 - (relative_position[2] + 1)) * 9;
					hippoLBM::LBMComm<Q> send(neig, default_tag + neig * 27 + send_tag, send_box);
					hippoLBM::LBMComm<Q> recv(neig, default_tag + mpi_rank * 27 + recv_tag, recv_box);
					assert(recv.get_size() == send.get_size());
					if (recv.get_size() != 0)
						manager.add_comm(send, recv);
				}
			}
			LBMDomain<Q> domain(manager, local_box, g, bounds, domain_size, MPI_coord, MPI_grid_size);
			return domain;
		}
}

