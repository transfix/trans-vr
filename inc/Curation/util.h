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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef UTIL_H
#define UTIL_H

#include <Curation/datastruct.h>

namespace Curation
{

void
normalize(Vector& v);

double
length_of_seg(const Segment& s);

double
cell_volume( const Cell_handle& c);

bool          
is_inf_VF(const Triangulation& triang,
          const Cell_handle& c, const int uid, const int vid);

bool
is_outside_VF(const Triangulation& triang, 
              const Edge& e);

bool
find_0_volume_tetrahedron(Triangulation &triang);

bool
check_del_vor_property(Triangulation &triang);

bool 
identify_cospherical_neighbor(Triangulation &triang);

bool
is_obtuse(const Point& p0, const Point& p1, const Point& p2);

void
cluster_cospherical_tetrahedra(Triangulation &triang);


void
find_flow_direction(Triangulation &triang );

void 
identify_sink_and_saddle(Triangulation &triang);

double 
cosine( const Vector& v, const Vector& w);

int 
find_third_vertex_index( const Facet& f, Vertex_handle v, Vertex_handle w); 

int 
edge_index( const int facet_index, const int first_vertex_index, 
	    const int second_vertex_index);

void 
vertex_indices( const int facet_index, const int edge_index,
		int& first_vertex, int& second_vertex);

}

#endif // UTIL_H

