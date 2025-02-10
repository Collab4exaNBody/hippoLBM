#pragma once

#include <hipoLBM/io/writer.hpp>
#include <onika/string_utils.h>


namespace hipoLBM
{
	template<typename DomainQ>
		inline void write_pvtr( std::string basedir,  std::string basename, size_t number_of_files, DomainQ& domain)
		{
			grid<3>& Grid = domain.m_grid;
			auto [lx, ly, lz] = domain.domain_size;
			// I could be smarter here
			int box_size = sizeof(box<3>);
			auto global = Grid.build_box<Area::Global, Traversal::Extend>(); //Traversal::Real>();
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
				outFile << "   <PRectilinearGrid WholeExtent=\"0 " << lx - 1 << " 0 " << ly - 1 << " 0 " << lz - 1<< "\"";
				outFile << " GhostLevel=\"#\">" << std::endl;
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
				outFile << "     </PPointData> " << std::endl;
				outFile << "   </PRectilinearGrid>" << std::endl;
				outFile << " </VTKFile>" << std::endl;
			}
		}


	template<typename DomainQ, typename GridDataQ>
		inline void write_vtr(std::string name, DomainQ& domain, GridDataQ& data, traversal_lbm& traversals)
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
			constexpr Traversal Tr = Traversal::Extend;
			auto local = Grid.build_box<L,Tr>();
			auto global = Grid.build_box<G,Tr>();

			auto [traversal_ptr, traversal_size] = traversals.get_data<Tr>();

			write_file writter;
			write_vec3d writter_vec3d = {local};

			assert( local.get_length(0) == global.get_length(0) );
			assert( local.get_length(1) == global.get_length(1) );
			assert( local.get_length(2) == global.get_length(2) );

			outFile << "<VTKFile type=\"RectilinearGrid\">"  << std::endl;
			outFile << " <RectilinearGrid WholeExtent=\" 0 " << lx - 1 << " 0 " << ly - 1 << " 0 " << lz - 1<< "\">"  << std::endl;
			outFile << "      <Piece Extent=\""<< global.start(0) << " " << global.end(0) << " " << global.start(1) << " " << global.end(1) << " " << global.start(2) << " " << global.end(2) << " \">" << std::endl;
			outFile << "      <Coordinates>" << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"X\" format=\"ascii\">" <<std::endl;
			for(int x = global.start(0) ; x <= global.end(0) ; x++) outFile << (double)(x*dx) << " ";
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"Y\" format=\"ascii\">" <<std::endl;
			for(int y = global.start(1) ; y <= global.end(1) ; y++) outFile << (double)(y*dx) << " ";
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"Z\" format=\"ascii\">" <<std::endl;
			for(int z = global.start(2) ; z <= global.end(2) ; z++) outFile << (double)(z*dx) << " ";
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "      </Coordinates>" << std::endl;
			outFile << "      <PointData>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"P\" format=\"ascii\">" << std::endl;
			for_all(traversal_ptr, traversal_size, writter, outFile, onika::cuda::vector_data(data.m0));
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"U\" format=\"ascii\" NumberOfComponents=\"3\">" << std::endl;
			for_all<L, Tr>(Grid, writter_vec3d, outFile, onika::cuda::vector_data(data.m1));
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "          <DataArray type=\"Float32\" Name=\"OBST\" format=\"ascii\">" << std::endl;
			for_all(traversal_ptr, traversal_size, writter, outFile, onika::cuda::vector_data(data.obst));
			outFile << std::endl;
			outFile << "          </DataArray>"  << std::endl;
			outFile << "      </PointData>"  << std::endl;
			outFile << "      </Piece>" << std::endl;
			outFile << " </RectilinearGrid>"  << std::endl;
			outFile << "</VTKFile>"  << std::endl;
		}
}
