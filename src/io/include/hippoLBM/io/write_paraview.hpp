#pragma once

#include <hippoLBM/io/writer.hpp>
#include <onika/string_utils.h>
#include <grid/parallel_for_core.cu>


namespace hippoLBM
{
  constexpr Traversal PARAVIEW_TR = Traversal::All;

  struct paraview_buffer
  {
    /** Buffers */
    onika::memory::CudaMMVector<float> u; // Vec3d
    onika::memory::CudaMMVector<float> p;
    onika::memory::CudaMMVector<int> obst;

    /** streams */
    std::stringstream i;
    std::stringstream j;
    std::stringstream k;

    void resize(const int size)
    {
      u.resize( 3 * size); // Vec3d
      p.resize(size);
      obst.resize(size);
    }
 
    void sim_data_to_stream(box<3>& Box, double dx)
    {  
      // todo
    }

    void sim_header_to_stream(box<3>& Box, double dx)
    {
      for(int x = Box.start(0) ; x <= Box.end(0) ; x++) i << (double)(x*dx) << " ";
      for(int y = Box.start(1) ; y <= Box.end(1) ; y++) j << (double)(y*dx) << " ";
      for(int z = Box.start(2) ; z <= Box.end(2) ; z++) k << (double)(z*dx) << " ";
    }

    //void buffer_to_stream(
  };


  template<typename DomainQ>
    inline void write_pvtr( std::string basedir,  std::string basename, size_t number_of_files, DomainQ& domain, bool print_distributions)
    {
      grid<3>& Grid = domain.m_grid;
      auto [lx, ly, lz] = domain.domain_size;
      // I could be smarter here
      int box_size = sizeof(box<3>);
      auto global = Grid.build_box<Area::Global, PARAVIEW_TR>();// Traversal::Extend>(); //Traversal::Real>();
      std::vector<box<3>> recv;
      recv.resize(number_of_files);
      MPI_Gather(&global, box_size, MPI_CHAR, recv.data(), box_size, MPI_CHAR, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if(rank == 0)
      {
        std::string name = basedir + "/" + basename + ".pvtr";
        std::ofstream outFile(name);
        if (!outFile) {
          std::cerr << "Erreur : impossible de créer le fichier de sortie suivant: " << name << std::endl;
          return;
        }

        outFile << " <VTKFile type=\"PRectilinearGrid\"> " << std::endl;
        outFile << "   <PRectilinearGrid WholeExtent=\"0 " << lx - 1 << " 0 " << ly - 1 << " 0 " << lz - 1<< "\"" << std::endl;;
        outFile << "                     GhostLevel=\"#\">" << std::endl;
        //outFile << " GhostLevel=\"#\">" << std::endl;
        //  outFile << "      <Piece Extent=\"0 " << lx << " 0 " << ly << " 0 " << lz<< "\"" << std::endl;
        for(size_t i = 0 ; i < number_of_files ; i++ )
        {
          std::string subfile = basename + "/%06d.vtr" ;
          subfile = onika::format_string(subfile, i);
          outFile << "     <Piece Extent=\" " << recv[i].start(0) << " " <<  recv[i].end(0)  << " " << recv[i].start(1) << " " <<  recv[i].end(1)  << " " << recv[i].start(2) << " " <<  recv[i].end(2)  << "\" Source=\"" << subfile << "\"/>" << std::endl;
        }
        outFile << "    <PCoordinates>" << std::endl;
        outFile << "      <PDataArray type=\"Float32\" Name=\"X\"/>" << std::endl;
        outFile << "      <PDataArray type=\"Float32\" Name=\"Y\"/>" << std::endl;
        outFile << "      <PDataArray type=\"Float32\" Name=\"Z\"/>" << std::endl;
        outFile << "    </PCoordinates>" << std::endl;
        outFile << "     <PPointData Scalars=\"P OBST\"  Vectors=\"U\" >" << std::endl;
        outFile << "       <PDataArray Name=\"P\" type=\"Float32\" NumberOfComponents=\"1\"/>" << std::endl;
        outFile << "       <PDataArray Name=\"OBST\" type=\"Float32\" NumberOfComponents=\"1\"/>" << std::endl;
        outFile << "       <PDataArray Name=\"U\" type=\"Float32\" NumberOfComponents=\"3\"/>" << std::endl;
        if(print_distributions) 
        {
          outFile << "       <PDataArray Name=\"Fi\" type=\"Float32\" NumberOfComponents=\"19\"/>" << std::endl;
        }
        outFile << "     </PPointData> " << std::endl;
        outFile << "   </PRectilinearGrid>" << std::endl;
        outFile << " </VTKFile>" << std::endl;
      }
    }


  template<typename DomainQ, typename GridDataQ>
    inline void write_vtr(std::string name, DomainQ& domain, GridDataQ& data, traversal_lbm& traversals, LBMParameters params, bool print_distributions)
    {
      grid<3>& Grid = domain.m_grid;
      auto [lx, ly, lz] = domain.domain_size;
      const double dx = Grid.dx;
      name = name + ".vtr";
      std::ofstream outFile(name);
      if (!outFile) {
        std::cerr << "Erreur : impossible de créer le fichier de sortie suivant: " << name << std::endl;
        return;
      }
      // only real point  
      constexpr Area L = Area::Local;
      constexpr Area G = Area::Global;
      auto local = Grid.build_box<L,PARAVIEW_TR>();
      auto global = Grid.build_box<G,PARAVIEW_TR>();

      auto [traversal_ptr, traversal_size] = traversals.get_data<PARAVIEW_TR>();

      const int * const obst = data.obstacles();

      NullFuncWriter nullop;
      write_file writer_obst = {nullop};
      write_distributions<19> writer_Q;

      double ratio_dx_dtLB = dx / params.dtLB;
      UWriter u = {obst, ratio_dx_dtLB};
      write_vec3d writer_vec3d = {u, local};

      double c_c_avg_rho_div_three = 1./3. * params.celerity * params.celerity * params.avg_rho;
      PressionWriter pression = {obst, c_c_avg_rho_div_three};
      write_file writer_double = {pression};

			assert( local.get_length(0) == global.get_length(0) );
			assert( local.get_length(1) == global.get_length(1) );
			assert( local.get_length(2) == global.get_length(2) );

			paraview_buffer paraview_streams;
			paraview_streams.sim_header_to_stream(global, dx);

			outFile << "<VTKFile type=\"RectilinearGrid\">"  << std::endl;
			outFile << " <RectilinearGrid WholeExtent=\" 0 " << lx - 1 << " 0 " << ly - 1 << " 0 " << lz - 1<< "\">"  << std::endl;
			outFile << "      <Piece Extent=\""<< global.start(0) << " " << global.end(0) << " " << global.start(1) << " " << global.end(1) << " " << global.start(2) << " " << global.end(2) << " \">" << std::endl;
			outFile << "      <Coordinates>" << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"X\" format=\"ascii\">" <<std::endl;
			outFile << paraview_streams.i.rdbuf();
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"Y\" format=\"ascii\">" <<std::endl;
			outFile << paraview_streams.j.rdbuf();
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"Z\" format=\"ascii\">" <<std::endl;
			outFile << paraview_streams.k.rdbuf();
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "      </Coordinates>" << std::endl;
			outFile << "      <PointData>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"P\" format=\"ascii\">" << std::endl;
			{ 
				std::stringstream paraview_stream_buffer;
				for_all(traversal_ptr, traversal_size, writer_double, paraview_stream_buffer, onika::cuda::vector_data(data.m0));
				outFile << paraview_stream_buffer.rdbuf();
			} 
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"U\" format=\"ascii\" NumberOfComponents=\"3\">" << std::endl;
			{ 
				std::stringstream paraview_stream_buffer;
				for_all<L, PARAVIEW_TR>(Grid, writer_vec3d, paraview_stream_buffer, data.flux());
				outFile << paraview_stream_buffer.rdbuf();
			} 
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"OBST\" format=\"ascii\">" << std::endl;
			{
				std::stringstream paraview_stream_buffer;
				for_all(traversal_ptr, traversal_size, writer_obst, paraview_stream_buffer, onika::cuda::vector_data(data.obst));
				outFile << paraview_stream_buffer.rdbuf();
			}
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;

			if(print_distributions)
			{
				outFile << "          <DataArray type=\"Float32\" Name=\"Fi\" format=\"ascii\" NumberOfComponents=\"19\">" << std::endl;
				{
					std::stringstream paraview_stream_buffer;
					for_all(traversal_ptr, traversal_size, writer_Q, paraview_stream_buffer, data.distributions());
					outFile << paraview_stream_buffer.rdbuf();
				}
				outFile << std::endl;
				outFile << "          </DataArray>"  << std::endl;
			}

			outFile << "      </PointData>"  << std::endl;
			outFile << "      </Piece>" << std::endl;
			outFile << " </RectilinearGrid>"  << std::endl;
			outFile << "</VTKFile>"  << std::endl;
		}
}
