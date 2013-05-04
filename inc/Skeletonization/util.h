/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Skeletonization.

  Skeletonization is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Skeletonization is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __SKELETONIZATION__UTIL_H__
#define __SKELETONIZATION__UTIL_H__

#include <Skeletonization/datastruct.h>
#include <Skeletonization/graph.h>

namespace Skeletonization
{

double 
cosine( const Vector& v, const Vector& w);

void
normalize(Vector& v);

double
length_of_seg(const Segment& s);

bool
is_obtuse(const Point& p0, const Point& p1, const Point& p2);

int 
find_third_vertex_index( const Facet& f, Vertex_handle v, Vertex_handle w);

int 
edge_index( const int facet_index, const int first_vertex_index, 
	    const int second_vertex_index);

void 
vertex_indices( const int facet_index, const int edge_index,
		int& first_vertex, int& second_vertex);

bool
is_same_side_of_ray(const Point& p0, const Point& p1,
                    const Point& a, const Point& b);

void
is_contained_in_inf_tr(const Point& p0, const Point& p1, const Point& p2,
                       const Point& a, const Point& b,
                       const vector<int>& coincidence_vector,
                       bool* contained);
bool
is_outside_bounding_box(const Point& p, 
                        const vector<double>& bounding_box);

bool
is_outside_bounding_box(const vector<Point>& points, 
                        const vector<double>& bounding_box);

bool
is_VF_outside_bounding_box(const Triangulation& triang,
                           const Edge& e,
                           const vector<double>& bounding_box);
bool
is_VF_crossing_surface(const Triangulation& triang, 
                       const Edge& e);

bool
is_cospherical_pair(const Triangulation& triang, const Facet& f);

bool
identify_cospherical_neighbor(Triangulation &triang);

void
mark_VF_on_u1(Triangulation& triang,
              Cell_handle& c, int uid, int vid);

void
mark_VF_visited(Triangulation& triang,
                Cell_handle& c, int uid, int vid);

void
set_patch_id(Triangulation& triang,
             Cell_handle& c, int uid, int vid,
             const int& id);

bool
is_surf_VF(const Triangulation& triang, 
           const Cell_handle& c, const int uid, const int vid);

bool          
is_inf_VF(const Triangulation& triang,
          const Cell_handle& c, const int uid, const int vid);


bool
is_there_any_common_element(const vector<int>& vec1, const vector<int>& vec2);


}
#endif // UTIL_H

