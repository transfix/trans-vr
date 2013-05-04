/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of SecondaryStructures.

  SecondaryStructures is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  SecondaryStructures is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __UTIL_H__
#define __UTIL_H__

#include <SecondaryStructures/datastruct_ss.h>

double my_drand();

double cosine(const SecondaryStructures::Vector& v, const SecondaryStructures::Vector& w);
void normalize(SecondaryStructures::Vector& v);
double length_of_seg(const SecondaryStructures::Segment& s);
bool is_obtuse(const SecondaryStructures::Point& p0, const SecondaryStructures::Point& p1, const SecondaryStructures::Point& p2);
bool are_neighbors(const SecondaryStructures::Cell_handle c0, const SecondaryStructures::Cell_handle c1);
int find_third_vertex_index(const SecondaryStructures::Facet& f, SecondaryStructures::Vertex_handle v, SecondaryStructures::Vertex_handle w);
int edge_index(const int facet_index, const int first_vertex_index, const int second_vertex_index);
void vertex_indices(const int facet_index, const int edge_index, int& first_vertex, int& second_vertex);
void transform(vector<SecondaryStructures::Point>& target, const vector<SecondaryStructures::Vector>& source, const SecondaryStructures::Vector& t, const SecondaryStructures::Point& o, const SecondaryStructures::Vector& a, const double& c, const double& s);
bool is_same_side_of_ray(const SecondaryStructures::Point& p0, const SecondaryStructures::Point& p1, const SecondaryStructures::Point& a, const SecondaryStructures::Point& b);
void is_contained_in_inf_tr(const SecondaryStructures::Point& p0, const SecondaryStructures::Point& p1, const SecondaryStructures::Point& p2, const SecondaryStructures::Point& a, const SecondaryStructures::Point& b, const vector<int>& coincidence_vector, bool* contained);
bool is_outside_bounding_box(const SecondaryStructures::Point& p, const vector<double>& bounding_box);
bool is_outside_bounding_box(const vector<SecondaryStructures::Point>& points, const vector<double>& bounding_box);
bool is_VF_outside_bounding_box(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Edge& e, const vector<double>& bounding_box);
bool is_VF_crossing_surface(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Edge& e);
bool is_cospherical_pair(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Facet& f);
bool identify_cospherical_neighbor(SecondaryStructures::Triangulation& triang);
void mark_VF_on_u1(SecondaryStructures::Triangulation& triang, SecondaryStructures::Cell_handle& c, int uid, int vid);
void mark_VF_visited(SecondaryStructures::Triangulation& triang, SecondaryStructures::Cell_handle& c, int uid, int vid);
void set_patch_id(SecondaryStructures::Triangulation& triang, SecondaryStructures::Cell_handle& c, int uid, int vid, const int& id);
bool is_surf_VF(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Cell_handle& c, const int uid, const int vid);
bool is_inf_VF(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Cell_handle& c, const int uid, const int vid);
bool is_outside_VF(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Edge& e);
bool is_inside_VF(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Edge& e);
bool is_surf_VF(const SecondaryStructures::Triangulation& triang, const SecondaryStructures::Edge& e);
bool is_there_any_common_element(const vector<int>& vec1, const vector<int>& vec2);

#endif
