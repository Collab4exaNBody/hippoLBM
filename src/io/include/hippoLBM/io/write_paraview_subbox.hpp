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

#include <onika/string_utils.h>

#include <filesystem>
<<<<<<< HEAD
#include <vector>

=======
>>>>>>> 49-paraview-dump-a-subbox-of-the-subdomain
#include <hippoLBM/compute/parallel_for_core.hpp>
#include <hippoLBM/core/enum.hpp>
#include <hippoLBM/grid/domain.hpp>
#include <hippoLBM/grid/fields.hpp>
#include <hippoLBM/grid/lbm_parameters.hpp>
#include <hippoLBM/io/write_paraview.hpp>
#include <hippoLBM/io/writer.hpp>
<<<<<<< HEAD

namespace hippoLBM {

/** @brief Writes a subpart of the LBM grid, restricted to `bounds` (a physical-space AABB), to a
 * Paraview file pair (.pvtr + one .vtr per overlapping rank), reusing write_pvtr/write_vtr so the
 * output is in the exact same P/U/OBST format as write_paraview. Ranks whose local subdomain does
 * not overlap `bounds` write nothing. */
=======
#include <vector>

namespace hippoLBM {

>>>>>>> 49-paraview-dump-a-subbox-of-the-subdomain
template <int Q>
void write_paraview_subbox(MPI_Comm comm, std::string filename, std::string basedir, long timestep,
                           LBMFields<Q>& fields, const LBMParameters& parameters, const LBMDomain<Q>& domain,
                           const onika::math::AABB& bounds, bool display_filename = false) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  const LBMGrid& grid = domain.grid();
  const double dx = grid.dx_;
  Point3D pmin = {int(std::floor(bounds.bmin.x / dx)), int(std::floor(bounds.bmin.y / dx)),
                  int(std::floor(bounds.bmin.z / dx))};
  Point3D pmax = {int(std::floor(bounds.bmax.x / dx)), int(std::floor(bounds.bmax.y / dx)),
                  int(std::floor(bounds.bmax.z / dx))};
  Box3D global_input_box = {pmin, pmax};

  // clip the requested box to the whole domain extent [0,lx-1]x[0,ly-1]x[0,lz-1] (NOT
  // grid.build_box<Global,All>(), which only covers this rank's own slice).
  auto [lx, ly, lz] = domain.domain_size_;
  Box3D full_domain = {Point3D{0, 0, 0}, Point3D{lx - 1, ly - 1, lz - 1}};
  Box3D whole_extent = global_input_box;
  bool has_domain_overlap = intersect(full_domain, whole_extent);
  if (has_domain_overlap) {
    for (int dim = 0; dim < 3; dim++) {
      whole_extent.inf_[dim] = std::max(whole_extent.inf_[dim], full_domain.inf_[dim]);
      whole_extent.sup_[dim] = std::min(whole_extent.sup_[dim], full_domain.sup_[dim]);
    }
  }
  if (!has_domain_overlap) {
    lout << "[write_paraview_subbox] requested bounds do not overlap the domain" << std::endl;
    return;
  }

  // this rank's intersection with the requested box, in local coordinates
  auto [is_local_valid, local_box] = grid.restrict_box_to_grid<Area::Local, Traversal::All>(global_input_box);

  std::string file_name = filename;
  file_name = onika::format_string(file_name, timestep);
  std::string fullname = basedir + file_name;

  if (rank == 0) {
    std::filesystem::create_directories(fullname);
  }

  if (display_filename) {
    lout << "writing paraview subbox file: " << fullname << std::endl;
  }

  std::string rank_file = fullname + "/%06d";
  rank_file = onika::format_string(rank_file, rank);

  MPI_Barrier(comm);

  Box3D piece_box_global = is_local_valid ? grid.convert<Area::Global>(local_box) : Box3D{};
  std::vector<Box3D> piece_boxes;
  std::vector<int> piece_valid;
  gather_piece_boxes(comm, piece_box_global, is_local_valid, piece_boxes, piece_valid);

  ExternalParaviewFields no_external_fields;  // subbox extraction: no distributions/external field support
  int size = int(piece_boxes.size());
  write_pvtr(basedir, file_name, size, whole_extent, piece_boxes, piece_valid, no_external_fields);

  if (is_local_valid) {
    write_vtr(rank_file, domain, fields, parameters, local_box, no_external_fields);
  }
}
}  // namespace hippoLBM
