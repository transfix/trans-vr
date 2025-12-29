/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Curation.

  Curation is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Curation is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef OP_H
#define OP_H

#include <Curation/datastruct.h>
#include <Curation/hfn_util.h>
#include <Curation/mds.h>
#include <boost/shared_ptr.hpp>
#include <cvcraw_geometry/cvcgeom.h>
#include <vector>

namespace Curation {

void draw_ray(const Ray &ray, const double &r, const double &g,
              const double &b, const double &a, ofstream &fout);

void draw_segment(const Segment &segment, const double &r, const double &g,
                  const double &b, const double &a, ofstream &fout);

void draw_poly(const vector<Point> &poly, const double &r, const double &g,
               const double &b, const double &a, ofstream &fout);

void draw_VF(const Triangulation &triang, const Edge &dual_e, const double &r,
             const double &g, const double &b, const double &a,
             ofstream &fout);

void draw_tetra(const Cell_handle &cell, const double &r, const double &g,
                const double &b, const double &a, ofstream &fout);

void write_wt(const Triangulation &triang, const char *file_prefix);
void write_smax(const Triangulation &triang,
                map<int, cell_cluster> &cluster_set,
                const vector<int> &sorted_cluster_index_vector,
                const int output_seg_count, const char *file_prefix);

void write_mesh(const Mesh &mesh, const char *op_prefix);

//  std::vector< CVCGEOM_NAMESPACE::cvcgeom_t >
CVCGEOM_NAMESPACE::cvcgeom_t
write_smax_to_geom(const Triangulation &triang,
                   map<int, cell_cluster> &cluster_set,
                   const vector<int> &sorted_cluster_index_vector); //,
//      const int output_seg_count);
} // namespace Curation

#endif // OP_H
