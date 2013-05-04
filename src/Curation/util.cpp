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

#include <Curation/util.h>

namespace Curation{

void
normalize(Vector& v)
{
   v = (1./(sqrt(CGAL::to_double(v*v))))*v;
   return;
}

double
length_of_seg(const Segment& s)
{
   return sqrt(CGAL::to_double((s.point(0) - s.point(1))*(s.point(0) - s.point(1))));
}

// check if the angle <p0,p2,p1 > 90 degree
bool
is_obtuse(const Point& p0, const Point& p1, const Point& p2)
{
   Vector v0 = p0 - p2;
   Vector v1 = p1 - p2;
   return (CGAL::to_double(v0 * v1) < 0);
}

double
cell_volume(const Cell_handle& c)
{
   Tetrahedron t = Tetrahedron(c->vertex(0)->point(),
                               c->vertex(1)->point(),
                               c->vertex(2)->point(),
                               c->vertex(3)->point());
   return ( CGAL::to_double(CGAL::abs(t.volume()) ) );
}

bool
is_p_inside_cell(const Point& p, const Cell_handle& c, bool& is_degenerate)
{
   is_degenerate = false;
   for(int id = 0; id < 4; id ++)
   {
       Tetrahedron t1 (c->vertex((id+1)%4)->point(),
                       c->vertex((id+2)%4)->point(),
                       c->vertex((id+3)%4)->point(),
                       p);
       Tetrahedron t2 (c->vertex((id+1)%4)->point(),
                       c->vertex((id+2)%4)->point(),
                       c->vertex((id+3)%4)->point(),
                       c->vertex(id)->point());
       if( CGAL::is_negative(t1.volume()*t2.volume()) ) return false;
       if( ! CGAL::is_positive(t1.volume()*t2.volume()) ) is_degenerate = true;
   }
   return true;
}

bool          
is_inf_VF(const Triangulation& triang,
          const Cell_handle& c, const int uid, const int vid)
{
   Facet_circulator fcirc = triang.incident_facets(Edge(c,uid,vid));
   Facet_circulator begin = fcirc;
   do{
      Cell_handle cur_c = (*fcirc).first;
      if( triang.is_infinite( cur_c ) ) return true;
      fcirc ++;
   } while(fcirc != begin);
   return false;
}

bool
is_outside_VF(const Triangulation& triang, 
              const Edge& e)
{
   Facet_circulator fcirc = triang.incident_facets(e);
   Facet_circulator begin = fcirc;
   do{
      if( ! (*fcirc).first->outside) return false;
      fcirc ++;
   } while(fcirc != begin);
   return true;
}

bool
is_inside_VF(const Triangulation& triang, 
             const Edge& e)
{
   Facet_circulator fcirc = triang.incident_facets(e);
   Facet_circulator begin = fcirc;
   do{
      if((*fcirc).first->outside) return false;
      fcirc ++;
   } while(fcirc != begin);
   return true;
}

bool
is_surf_VF(const Triangulation& triang, 
           const Edge& e)
{
   Facet_circulator fcirc = triang.incident_facets(Edge(e));
   Facet_circulator begin = fcirc;
   bool in = false, out = false;
   do{
      Cell_handle cur_c = (*fcirc).first;
      if( cur_c->outside ) out = true;
      else in = true;
      fcirc ++;
   } while(fcirc != begin);
   return (in && out);
}

bool
is_outside_bounding_box(const Point& p, 
                       const vector<double>& bounding_box)
{
   if(CGAL::to_double(p.x()) < bounding_box[0] ||
      CGAL::to_double(p.x()) > bounding_box[1] ||
      CGAL::to_double(p.y()) < bounding_box[2] ||
      CGAL::to_double(p.y()) > bounding_box[3] ||
      CGAL::to_double(p.z()) < bounding_box[4] ||
      CGAL::to_double(p.z()) > bounding_box[5] )
      return true;
   return false;
}

// Compute the cosine of the smaller of the two angles 
// made by the vectors v and w
double 
cosine( const Vector& v, const Vector& w) {
  return CGAL::to_double( v * w) / 
    sqrt( CGAL::to_double( v * v) * CGAL::to_double( w * w));
}

// Find the index of the third vertex for a facet where this vertex 
// is neither *v nor *w.
int 
find_third_vertex_index( const Facet& f, Vertex_handle v, Vertex_handle w) {
  int id = f.second;
  for ( int i =1; i <= 3; ++i) {
    if ( f.first->vertex((id+i)%4) != v && 
	 f.first->vertex((id+i)%4) != w) 
      return (id+i)%4;
  }
  return -1;
}

// Compute the index of an edge. The index of an edge in a facet
// is defined as the index of a facet in cell, i.e. it is the index
// of the opposite vertex. The arguments are the facet index of the
// facet witth respect to the cell and the indices of the vertices
// incident to the edge also with respect to the cell.
int 
edge_index( const int facet_index, const int first_vertex_index, 
	    const int second_vertex_index) {
  return 6 - facet_index - first_vertex_index - second_vertex_index;
}

// Compute the indices of the vertices incident to an edge.
void 
vertex_indices( const int facet_index, const int edge_index,
		int& first_vertex, int& second_vertex) {
  if ( (facet_index == 0 && edge_index == 1) ||
       (facet_index == 1 && edge_index == 0)) {
    first_vertex = 2; second_vertex = 3;
  } else if ( (facet_index == 0 && edge_index == 2) ||
	      (facet_index == 2 && edge_index == 0)) {
    first_vertex = 1; second_vertex = 3;
  } else if ( (facet_index == 0 && edge_index == 3) ||
	      (facet_index == 3 && edge_index == 0)) {
    first_vertex = 1; second_vertex = 2;
  } else if ( (facet_index == 1 && edge_index == 2) ||
	      (facet_index == 2 && edge_index == 1)) {
    first_vertex = 0; second_vertex = 3;
  } else if ( (facet_index == 1 && edge_index == 3) ||
	      (facet_index == 3 && edge_index == 1)) {
    first_vertex = 0; second_vertex = 2;
  } else if ( (facet_index == 2 && edge_index == 3) ||
	      (facet_index == 3 && edge_index == 2)) {
    first_vertex = 0; second_vertex = 1;
  }
}


}
