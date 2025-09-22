#include <mpi.h>
#include <onika/math/basic_types_yaml.h>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/comm.hpp>

namespace hippoLBM
{

	using namespace onika::math;

	/**
	 * @brief Static configuration for HippoLBM grid.
	 */
	struct hippoLBMGridConfig
	{
		static constexpr int dim = 3;
		static constexpr int ghost_layer = 2;
	};

	/**
	 * @brief Describes the global simulation grid.
	 */
	struct GridConfig
	{
		IJK dims; ///< Number of nodes in each direction (i,j,k).
		AABB bounds; ///< Physical bounding box of the grid domain.
		std::array<bool, hippoLBMGridConfig::dim> periodic; ///< Periodicity flags along each axis.

    void display() const
    {
      std::cout << "=================================" <<std::endl;
      std::cout << "GridConfig: " << std::endl;
      std::cout << "Grid dim size: [" << dims << "]" << std::endl; 
      std::cout << "Bounds inf:    [" << bounds.bmin << "]" << std::endl; 
      std::cout << "Bounds sup:    [" << bounds.bmax << "]" << std::endl; 
      std::cout << "=================================" <<std::endl;
    }
	};

	/**
	 * @brief Describes a subgrid, e.g. a partition of the global grid for parallel computations.
	 */
	struct SubGridConfig
	{
		double dx; ///< Grid node size.
		IJK cart_coordinate; ///< Cartesian coordinates of the subgrid in the process grid.
		IJK cart_dims; ///< Dimensions of the Cartesian process grid.
		MPI_Comm cart_comm; ///< MPI Cartesian communicator.
		Vec3d offset; ///< Offset of the subgrid relative to the global origin.
		GridBlock block; ///< Local block of cells owned by this subgrid.
		std::array<bool, hippoLBMGridConfig::dim> periodic; ///< Periodicity flags along each axis.

    void display() const
    {
      std::cout << "=================================" <<std::endl;
      std::cout << "SubGridConfig: " << std::endl;
      std::cout << "SubGrid node size:   " << dx << std::endl;
      std::cout << "SubGrid offset:      ["<< offset <<"]" << std::endl;
      std::cout << "SubGrid block start: ["<< block.start <<"]" << std::endl;
      std::cout << "SubGrid block end:   ["<< block.end <<"]" << std::endl;
      std::cout << "Periodicity:         ["<< periodic[0] << "," << periodic[1] << "," << periodic[2] <<"]" << std::endl;
      std::cout << "=================================" <<std::endl;
    }
	};

	/**
	 * @brief Refines a subgrid resolution by a given factor.
	 *
	 * @param coarse_grid Input coarse subgrid.
	 * @param resolution Refinement factor.
	 * @return SubGridDetails Refined grid with updated block indices.
	 */
	inline SubGridConfig sub_grid_resolution(const SubGridConfig& coarse_grid, ssize_t resolution)
	{
		SubGridConfig refined_grid = coarse_grid;
		refined_grid.block.start =  refined_grid.block.start * resolution;
		refined_grid.block.end = refined_grid.block.end * resolution;
    refined_grid.dx /= double(resolution);
		return refined_grid;
	}

	/**
	 * @brief Refines a grid resolution by a given factor.
	 *
	 * @param coarse_grid Input coarse grid.
	 * @param resolution Refinement factor.
	 * @return GridDetails Refined grid with updated block indices.
	 */
	inline GridConfig grid_resolution(const GridConfig& coarse_grid, ssize_t resolution)
	{
		GridConfig refined_grid = coarse_grid;
		refined_grid.dims = refined_grid.dims * resolution;
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
		inline std::array<bool, hippoLBMGridConfig::dim> convert(const std::vector<bool>& b)
		{
			std::array<bool, hippoLBMGridConfig::dim> a;
			for(int dim = 0 ; dim < hippoLBMGridConfig::dim ; dim++)
			{
				a[dim] = b[dim];
			}
			return a;
		}


	/**
	 * @brief Build a distributed LBM domain from a global and subgrid configuration.
	 *
	 * This function partitions the global domain into subdomains assigned to MPI ranks,
	 * sets up local boxes including ghost layers (2), applies periodicity rules,
	 * and initializes ghost communication patterns between neighboring subdomains.
	 *
	 * @tparam Q Number of discrete velocities in the LBM stencil (e.g. 19 for D3Q19).
	 * @param grid Global grid configuration (domain size, bounds).
	 * @param subgrid Subgrid configuration for this MPI rank (coordinates, communicator, periodicity).
	 * @return LBMDomain<Q> Fully constructed distributed LBM domain.
	 */
	template<int Q>
		LBMDomain<Q> make_domain(const GridConfig grid, const SubGridConfig& subgrid)
		{
      grid.display();
      subgrid.display();
			constexpr int ghost_layer = hippoLBMGridConfig::ghost_layer;

			auto periodic = subgrid.periodic;
			int3d domain_size = convert<int3d>(grid.dims);

			IJK mpi_coords = subgrid.cart_coordinate;
			IJK mpi_grid_dims = subgrid.cart_dims;

			int3d coord = convert<int3d>(mpi_coords);
			int3d ndims = convert<int3d>(mpi_grid_dims);

			int3d relative_position;

      int3d subdomain_with_ghost_size = convert<int3d>(subgrid.block.end - subgrid.block.start +  2 * ghost_layer - 1);
      int3d offset = convert<int3d>(subgrid.block.start - ghost_layer);

			/// Local domain including ghost cells
			Box3D local_box = { {0, 0, 0}, subdomain_with_ghost_size };

      /// Extended real domain (skip invalid points at non-periodic boundaries)
			Box3D ext = local_box;
			for (int dim = 0; dim < hippoLBMGridConfig::dim; dim++) {
				if (!periodic[dim]) // not periodic
				{
					if (coord[dim] == 0)
						ext.inf[dim] += ghost_layer;
					if (coord[dim] == ndims[dim] - 1)
						ext.sup[dim] -= ghost_layer;
				}
			}

			// ----------------------------
			// Initialize grid object
			// ----------------------------
			LBMGrid g;

			g.set_box(local_box);
			g.set_ext(ext);
			g.set_offset({offset[0], offset[1], offset[2]});
			g.set_ghost_layer(ghost_layer);
			g.set_dx(subgrid.dx);

			// ----------------------------
			// Setup ghost communication manager
			// ----------------------------
			LBMGhostManager<Q> manager;

			int neig;
			const MPI_Comm& MPI_COMM_CART = subgrid.cart_comm;
			int mpi_rank;
			MPI_Comm_rank(MPI_COMM_CART, &mpi_rank);

			// Build ghost communications with 26 neighbors (3^3 - 1)
			for (int i = 0; i < 27; i++) 
			{
				int idx = i;
				for (int dim = 0; dim < hippoLBMGridConfig::dim; dim++) 
				{
					int d = idx % 3; // keep 3 first bits
					relative_position[dim] = d - 1;
					idx = (idx - d) / 3; // shift 3 bits
				}

				if (relative_position == int3d{0,0,0})
					continue;

				// Find neighbor coordinates in MPI grid
				auto [exists, coord_neig] = build_comm(relative_position, coord, periodic, ndims);

				if (!exists) continue;

				MPI_Cart_rank(MPI_COMM_CART, coord_neig.data(), &neig);

        // Build send/receive boxes for this neighbor
				auto [send_box, recv_box] = build_boxes(relative_position, g);

        int rtag;
				int stag = relative_position[0] + 1
					+ (relative_position[1] + 1) * 3 
					+ (relative_position[2] + 1) * 9;
				if(neig == mpi_rank)  
				{
					// ----------------------------
					// Case 1: periodic self-communication
					// ----------------------------
					send_box = fix_box_with_periodicity(relative_position, g);
          rtag = stag;
				}
        else
        {
				  rtag = 2 - relative_position[0] - 1
					     + (2 - relative_position[1] - 1) * 3 
					     + (2 - relative_position[2] - 1) * 9;
          if(rtag > 27) std::cout << "rtag: " << rtag << " " << relative_position[0] << " " << relative_position[1] << " " << relative_position[2]<< std::endl;
        }

				hippoLBM::LBMComm<Q> send(neig, stag, send_box);
				hippoLBM::LBMComm<Q> recv(neig, rtag, recv_box);

				assert(recv.get_size() == send.get_size());
				if (recv.get_size() != 0)
				{
					manager.add_comm(send, recv);
				}
			}

			//manager.debug_print_comm();
      //write_comm(manager);

			auto bounds_cpy = grid.bounds;

			LBMDomain<Q> domain(manager, local_box, g, bounds_cpy, domain_size, mpi_coords, mpi_grid_dims);
			return domain;
		} 

	/**
	 * @brief Partition the global grid across MPI ranks using load balancing.
	 *
	 * This function uses MPI Cartesian topology to distribute the global grid
	 * into subdomains, ensuring that each MPI rank receives a block of cells.
	 * Handles periodic boundaries, computes local block size, offset, and
	 * constructs a SubGridConfig for the calling rank.
	 *
	 * @param grid   Global grid configuration (size, bounds, periodicity).
	 * @param comm   MPI communicator (will be converted into a Cartesian communicator).
	 * @return SubGridConfig Subdomain configuration for the current MPI rank.
	 */
	SubGridConfig load_balancing(
			const GridConfig& grid, 
			MPI_Comm& comm)
	{
		auto& [ grid_size, bounds, periodic ]  = grid;
		double GridDx = double(bounds.bmax.x - bounds.bmin.x) / double(grid_size.i);

		// Consistency check: ensure dx is the same in all directions
		if( GridDx != double(bounds.bmax.y - bounds.bmin.y) / double(grid_size.j))
		{
			onika::lout << "Grid dimensions and bounds are inconsistent: dx must be equal in X and Y." << std::endl;
		}
		if( GridDx != double(bounds.bmax.z - bounds.bmin.z) / double(grid_size.k))
		{
			onika::lout << "Grid dimensions and bounds are inconsistent: dx must be equal in Z." << std::endl;
		}

		// ----------------------------
		// Setup periodic boundary conditions
		// ----------------------------
		std::array<int, hippoLBMGridConfig::dim> periods;
		for(size_t dim = 0 ; dim < hippoLBMGridConfig::dim ; dim++)
		{
			periods[dim] = periodic[dim] ? 1 : 0;
		}

		int3d mpi_dims;
		MPI_Comm MPI_COMM_CART;
		int mpi_rank, mpi_size;
		MPI_Comm_rank(comm, &mpi_rank);
		MPI_Comm_size(comm, &mpi_size);

		for (int dim = 0; dim < hippoLBMGridConfig::dim; dim++) 
		{
			mpi_dims[dim] = 0; // do not remove it
		}
		MPI_Dims_create(mpi_size, hippoLBMGridConfig::dim, mpi_dims.data());
		MPI_Cart_create(comm, hippoLBMGridConfig::dim, mpi_dims.data(), periods.data(), true, &MPI_COMM_CART);
		int3d mpi_coords;
		MPI_Cart_coords(MPI_COMM_CART, mpi_rank, hippoLBMGridConfig::dim, mpi_coords.data());


		IJK block_size;

		auto set_block_size = [&mpi_coords, &mpi_dims] (int dim, const int global_dim_size) -> int
		{
			if(mpi_coords[dim] != mpi_dims[dim] -1 )
			{
				// Regular block (all but last rank in this dimension)
				return global_dim_size / mpi_dims[dim];
			}
			else
			{
				// Last rank in this dimension: handle remainder
				return global_dim_size - (global_dim_size / mpi_dims[dim]) * mpi_coords[dim];
			}
		}; 

		// ----------------------------
		// Compute block indices in the global grid
		// ----------------------------
		block_size.i = set_block_size(0, grid_size.i);
		block_size.j = set_block_size(1, grid_size.j);
		block_size.k = set_block_size(2, grid_size.k);

		IJK inf = { 
			(grid_size.i / mpi_dims[0]) * mpi_coords[0],
			(grid_size.j / mpi_dims[1]) * mpi_coords[1],
			(grid_size.k / mpi_dims[2]) * mpi_coords[2]}; 
		Vec3d offset = bounds.bmin + inf * GridDx;
		IJK sup = inf + block_size;

		// ----------------------------
		// Build subgrid configuration
		// ----------------------------
		SubGridConfig res;
		res.dx = GridDx;
		res.cart_coordinate = convert<IJK>(mpi_coords);
		res.cart_dims = convert<IJK>(mpi_dims);
		res.offset = offset;
		res.block = {inf, sup};
		res.periodic = periodic;
		res.cart_comm = MPI_COMM_CART;

		return res;
	}
}
