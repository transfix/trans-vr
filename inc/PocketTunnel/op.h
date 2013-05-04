/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PocketTunnel.

  PocketTunnel is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PocketTunnel is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef OP_H
#define OP_H

#include <PocketTunnel/datastruct.h>
#include <PocketTunnel/hfn_util.h>
#include <cvcraw_geometry/cvcgeom.h>

namespace PocketTunnel
{

void
draw_ray(const cgal_Ray& ray,
         const double& r, 
         const double& g, 
         const double& b, 
         const double& a, 
         ofstream& fout);

void
draw_segment(const Segment& segment, 
             const double& r, 
             const double& g, 
             const double& b, 
             const double& a,
             ofstream& fout);


void
draw_poly(const vector<Point>& poly,
          const double& r, 
          const double& g, 
          const double& b, 
          const double& a,
          ofstream& fout);


void
draw_VF(const Triangulation& triang,
        const Edge& dual_e, 
        const double& r, 
        const double& g, 
        const double& b, 
        const double& a,
        ofstream& fout);

void
draw_tetra(const Cell_handle& cell,
           const double& r, 
           const double& g, 
           const double& b, 
           const double& a,
           ofstream& fout);


void 
write_wt( const Triangulation &triang,
	  const char* file_prefix);
void
write_handle(const Triangulation &triang, 
	  map<int, cell_cluster> &cluster_set,
	  const vector<int> &sorted_cluster_index_vector,
	  const int output_seg_count,
	  const char* file_prefix);

void
convert_pocket_tunnel_to_rawc_geometry(CVCGEOM_NAMESPACE::cvcgeom_t** PTV,
                                       const Triangulation& triang, 
	                               map<int, cell_cluster> &cluster_set,
	                               const vector<int> &sorted_cluster_index_vector,
	                               const int, const int);

void
write_S2(const Triangulation& triang,
         const vector<vector<Triangle_3> > S2, const char* file_prefix);

};

#endif // OP_H

