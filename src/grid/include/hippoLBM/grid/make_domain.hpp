#include <mpi.h>
#include <onika/math/basic_types_yaml.h>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>

namespace hippoLBM
{

	using namespace onika::math;

	struct hippoLBMGridDetails
	{
		static constexpr int dim = 3;
		static constexpr bool ghost_layer = 2;
	};

	struct GridDetails
	{
		IJK dims;
		AABB bounds;
		std::array<bool, hippoLBMGridDetails::dim> periodic;
	};

	struct SubGridDetails
	{
		double Dx;
		IJK cart_coordinate;
		IJK cart_dims;
    MPI_Comm cart_comm;
		Vec3d offset;
		GridBlock block;
		std::array<bool, hippoLBMGridDetails::dim> periodic;
	};

	inline SubGridDetails sub_grid_resolution(const SubGridDetails& coarse_grid, ssize_t resolution)
	{
		SubGridDetails refined_grid = coarse_grid;
		refined_grid.block.start =  refined_grid.block.start * resolution;
		refined_grid.block.end = refined_grid.block.end * resolution;
		return refined_grid;
	}

	template<typename A, typename B> inline A convert(const B& in) { return A(in); } 

	template<>
		inline int3d convert<int3d, IJK>(const IJK& b)
		{
			int3d a;
			a[0] = b.i;
			a[1] = b.j;
			a[2] = b.k;
			return a;
		}

	template<>
		inline IJK convert<IJK, int3d>(const int3d& b)
		{
			IJK a;
			a.i = b[0];
			a.j = b[1];
			a.k = b[2];
			return a;
		}

	template<>
		inline std::array<bool, hippoLBMGridDetails::dim> convert(const std::vector<bool>& b)
		{
			std::array<bool, hippoLBMGridDetails::dim> a;
			for(int dim = 0 ; dim < hippoLBMGridDetails::dim ; dim++)
			{
				a[dim] = b[dim];
			}
			return a;
		}

	template<int Q>
		LBMDomain<Q> make_domain(const GridDetails grid, const SubGridDetails& subgrid)
		{
			constexpr int ghost_layer = hippoLBMGridDetails::ghost_layer;
			auto periodic = subgrid.periodic;
			int3d domain_size = convert<int3d>(grid.dims);
			IJK MPI_coord = subgrid.cart_coordinate;
			IJK MPI_grid_size = subgrid.cart_dims;

			int3d coord = convert<int3d>(MPI_coord);
			int3d ndims = convert<int3d>(MPI_grid_size);

			int3d subdomain, relative_position, offset;
			for (int dim = 0; dim < hippoLBMGridDetails::dim; dim++) {
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
			for (int dim = 0; dim < hippoLBMGridDetails::dim; dim++) {
				if (periodic[dim] == 0) // not periodic
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
			g.set_dx(subgrid.Dx);

			LBMGhostManager<Q> manager;
			// fill ghost comm
			int default_tag = 100;
			for (int i = 0; i < 27; i++) 
      {
				int idx = i;
				for (int dim = 0; dim < hippoLBMGridDetails::dim; dim++) 
        {
					int d = idx % 3; // keep 3 first bits
					relative_position[dim] = d - 1;
					idx = (idx - d) / 3; // shift 3 bits
				}
				if (relative_position[0] == 0 && relative_position[1] == 0 && relative_position[2] == 0)
					continue;

				auto [not_exist, coord_neig] = build_comm(relative_position, coord, domain_size, periodic, ndims);
				if (not_exist == false)
					continue;
				int neig;
				const MPI_Comm& MPI_COMM_CART = subgrid.cart_comm;
				int mpi_rank, mpi_size;
				MPI_Comm_rank(MPI_COMM_CART, &mpi_rank);
				MPI_Comm_size(MPI_COMM_CART, &mpi_size);
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

      auto bounds_cpy = grid.bounds;

			LBMDomain<Q> domain(manager, local_box, g, bounds_cpy, domain_size, MPI_coord, MPI_grid_size);
			return domain;
		} 



	SubGridDetails load_balancing(
			const GridDetails& grid, 
			MPI_Comm& comm)
	{
		auto& [ grid_size, bounds, periodic ]  = grid;
		double GridDx = double(bounds.bmax.x - bounds.bmin.x) / double(grid_size.i);
		if( GridDx != double(bounds.bmax.y - bounds.bmin.y) / double(grid_size.j))
		{
			lout << "GridDim and bounds are not dedfined correclty, GridDx should be equal for every direction" << std::endl;
		}
		if( GridDx != double(bounds.bmax.z - bounds.bmin.z) / double(grid_size.k))
		{
			lout << "coarseGridDim and bounds are not dedfined correclty, coarseGridDx should be equal for every direction" << std::endl;
		}


		// init periodic conditions 
		int * periods = new int[hippoLBMGridDetails::dim];
		for(size_t dim = 0 ; dim < periodic.size() ; dim++)
		{
			if(periodic[dim]) periods[dim] = 1;
			else periods[dim] = 0;
		}
		int3d ndims;
		MPI_Comm MPI_COMM_CART;
		int mpi_rank, mpi_size;
		MPI_Comm_rank(comm, &mpi_rank);
		MPI_Comm_size(comm, &mpi_size);

		for (int dim = 0; dim < hippoLBMGridDetails::dim; dim++) 
		{
			ndims[dim] = 0; // do not remove it
		}
		MPI_Dims_create(mpi_size, hippoLBMGridDetails::dim, ndims.data());
		MPI_Cart_create(comm, hippoLBMGridDetails::dim, ndims.data(), periods, true, &MPI_COMM_CART);
		int3d coord;
		MPI_Cart_coords(MPI_COMM_CART, mpi_rank, hippoLBMGridDetails::dim, coord.data());


		IJK block_size;

		auto set_block_size = [&coord, &ndims] (int dim, const int coarseGridDim) -> int
		{
			if(coord[dim] != ndims[dim] -1 )
			{
				return coarseGridDim / ndims[dim];
			}
			else
			{
				return coarseGridDim - (coarseGridDim / ndims[dim]) * coord[dim];
			}
		}; 


		block_size.i = set_block_size(0, grid_size.i);
		block_size.j = set_block_size(1, grid_size.j);
		block_size.k = set_block_size(2, grid_size.k);

		IJK inf = { 
			(grid_size.i / ndims[0]) * coord[0],
			(grid_size.j / ndims[1]) * coord[1],
			(grid_size.k / ndims[2]) * coord[2]}; 
		Vec3d offset = bounds.bmin + inf * GridDx;
		IJK sup = inf + block_size;

		SubGridDetails res;
		res.Dx = GridDx;
		res.cart_coordinate = convert<IJK>(coord);
		res.cart_dims = convert<IJK>(ndims);
		res.offset = offset;
		res.block = {inf, sup};
		res.periodic = periodic;
    res.cart_comm = MPI_COMM_CART;
		return res;
	}
}
