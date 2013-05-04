/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef DATASTRUCT_H
#define DATASTRUCT_H


#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <deque>
#include <string>
#include <list>
#include <vector>

// for compiler discrepancies
#include <CGAL/basic.h>

#include <CGAL/Cartesian.h>
#include <CGAL/Point_3.h>
#include <CGAL/Timer.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h> // implements Traits.
#include <CGAL/Triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Delaunay_triangulation_3.h>

using namespace std;

namespace TightCocone
{

// -----------------------------------------------------------------------
// Define a new Vertex_base with additional attributes.
// -----------------------------------------------------------------------

template < class Gt, class Vb = CGAL::Triangulation_vertex_base_3<Gt> >
class MEDAX_vertex : public Vb
{
  typedef Vb                                     Base;
public:
  typedef typename Vb::Point                     Point;
  typedef typename Point::R                      Rep;
  typedef typename Vb::Vertex_handle             Vertex_handle;
  typedef typename Vb::Cell_handle               Cell_handle;
  typedef CGAL::Vector_3<Rep>                    Vector;

  template < typename TDS2 >
  struct Rebind_TDS {
    typedef typename Vb::template Rebind_TDS<TDS2>::Other    Vb2;
    typedef MEDAX_vertex<Gt,Vb2>                            Other;
  };

  MEDAX_vertex()                          { init(); }
  MEDAX_vertex( const Point& p) : Base(p) { init(); }
  MEDAX_vertex( const Point& p, void* cell) 
    : Base(p, cell)                    { init(); }
  MEDAX_vertex( void* cell) : Base(cell)  { init(); }
  

  // Member functions for TightCocone

  void set_normal( const Vector& v)      { v_normal = v; }
  Vector normal() const                  { return v_normal; }

  void set_pole( const Point& p)         { v_pole = p; }
  Point pole() const                     { return v_pole; }

  void set_width( const double& w)       { v_width  = w; }
  double width() const                   { return v_width; }

  void set_diameter( const double& d)    { v_diameter = d; }
  double diameter() const                { return v_diameter; }

  void set_flat( const bool b)           { v_flat = b; } 
  bool is_flat() const                   { return v_flat; }

  void set_convex_hull( const bool b)    { v_chull = b; }
  bool on_convex_hull() const            { return v_chull; }

  void set_isolated( const bool b)       { v_isolated = b; }
  bool is_isolated() const               { return v_isolated; }

  void set_umbrella( const bool b)       { v_umbrella = b; }
  bool has_umbrella() const              { return v_umbrella; }

  void set_on_smooth_surface( const bool b)     { v_on_smooth_surface = b; }
  bool on_smooth_surface() const                { return v_on_smooth_surface; }

  // Public member needed for robust cocone
  vector<double> nnd_vector;

  // General purpose
  int id;  
  bool tag;   
  bool visited;

  // For TightCocone
  bool bad;
  bool bad_neighbor;
  vector<int > bad_v;

  // For Medial Axis
  vector<Vector> normal_stack; // Used to store the normals of the
                               // incident disk

private:

  inline void init() 
  {
    v_normal = CGAL::NULL_VECTOR;
    v_pole = this->point();
    v_width = 0.0;
    v_diameter = 0.0; 
    v_flat = true;
    v_chull = false;
    v_umbrella = false;
    v_isolated = true;

    v_on_smooth_surface = false;
    nnd_vector.clear();
    nnd_vector.resize(3, DBL_MAX);

    // init of general purpose fields.
    id = -1;
    tag = false;
    visited = false;
  }

  Vector v_normal;    // Estimated normal at the vertex.
  Point  v_pole;      // The negative pole.
  double v_width;     // Radius of the cocone.
  double v_diameter;  // Diamter of the the incident umbrella.
  bool   v_flat;      // Indicates whether the vertex is flat.
  bool   v_chull;     // Indicates wheter the vertex lies on the boundary
                      // of the convex hull.
  bool   v_umbrella;  // Indicates if v has a (non sharp) umbrella.
  bool   v_isolated;  // Indicates whether the vertex is isolated.
  bool   v_on_smooth_surface; // Indicates whether, after RobustCocone,
                              // the vertex is selected.

};


// -----------------------------------------------------------------------
// Define a new Cell_base storing associated Voronoi vertices 
// and cocone flags.
// -----------------------------------------------------------------------
  

template < class Gt, class Cb = CGAL::Triangulation_cell_base_3<Gt> >
class MEDAX_cell : public Cb
{
  typedef Cb                                       Base;
public:
  typedef typename Cb::Point                       Point;
  typedef typename Cb::Vertex_handle               Vertex_handle;
  typedef typename Cb::Cell_handle                 Cell_handle;

  MEDAX_cell()                                  { init(); }
  MEDAX_cell( Vertex_handle v0, 
           Vertex_handle v1,
           Vertex_handle v2,
           Vertex_handle v3 )
    : Base(v0,v1,v2,v3)                      { init(); }
  MEDAX_cell( Vertex_handle v0, 
           Vertex_handle v1,
           Vertex_handle v2,
           Vertex_handle v3,
           Cell_handle c0,
           Cell_handle c1,
           Cell_handle c2,
           Cell_handle c3 )
    : Base(v0,v1,v2,v3,c0,c1,c2,c3)          { init(); }

  template < typename TDS2 >
  struct Rebind_TDS {
    typedef typename Cb::template Rebind_TDS<TDS2>::Other    Cb2;
    typedef MEDAX_cell<Gt,Cb2>                                  Other;
  };


  // Member functions for TightCocone
  void  set_voronoi( const Point& p)          { c_voronoi = p; }
  Point voronoi() const                       { return c_voronoi; }

  void set_cocone_flag( int i, bool new_value) {
    CGAL_precondition( 0 <= i && i < 4);
    c_cocone_flag[i] = new_value; 
  }

  bool cocone_flag( int i) const { 
    CGAL_precondition( 0 <= i && i < 4);
    return c_cocone_flag[i]; 
  }

  void set_removable( int i, bool new_value) {
    CGAL_precondition( 0 <= i && i < 4);
    c_removable[i] = new_value; 
  }

  bool removable( int i) const { 
    CGAL_precondition( 0 <= i && i < 4);
    return c_removable[i]; 
  }
  // End

  // Member functions for Robust Cocone
  void set_cell_radius( const double r)           	  { c_cell_radius = r; }
  double cell_radius() const 			  	  { return c_cell_radius; }
 
  void set_bb( const bool flag)               { c_bb = flag; }
  bool bb() const                             { return c_bb; }

  void set_deep_int(const int i, const bool flag) { c_deep_int[i] = flag; }
  bool deep_int(int i) const 		          { return c_deep_int[i]; }
  // End

  // Functions for Medial Axis
  void set_VF_on_medax(const int i, const int j, const bool val)
  {
	  CGAL_precondition(0 <= i && i <= 3);
	  CGAL_precondition(0 <= j && j <= 3);
	  CGAL_precondition(i != j);
	  c_VF_on_medax[i][j] = val;
	  c_VF_on_medax[j][i] = val;
  }

  bool VF_on_medax(int i, int j) const
  {
	  CGAL_precondition(0 <= i && i <= 3);
	  CGAL_precondition(0 <= j && j <= 3);
	  CGAL_precondition(i != j);
	  return c_VF_on_medax[i][j];
  }

  void set_VV_on_medax(const bool& b)    { c_VV_on_medax = b; }
  bool VV_on_medax() const               { return c_VV_on_medax; }

  void set_cosph_pair( const int &i, const bool &b)       { c_cosph_pair[i] = b; }
  bool cosph_pair(int i) const                            { return c_cosph_pair[i]; }

  void set_dirty(const bool& b)   { c_dirty = b; }
  bool dirty() const              { return c_dirty; }

  // Members for TightCocone
  bool outside;
  bool bdy[4];
  bool opaque[4];
  bool transp;
  int umbrella_member[4][4];

  // General purpose members.
  bool tag[4];   
  int id; 
  bool flag;
  bool visited;
  bool f_visited[4];
  bool e_visited[4][4];
  bool e_tag[4][4];
  int medax_comp_id[4][4];

  //For Cospherical degeracy
  bool c_cosph;

  inline void init() 
  {
    id = -1;
    visited = false;
    c_dirty = false;
    c_cosph = false;
    outside = false;
    c_cell_radius = 0.0;
    c_bb = false;
    c_VV_on_medax = false;
    for ( int i = 0; i < 4; ++i) 
    {
       f_visited[i] = false;
       c_cocone_flag[i] = false; 
       c_removable[i] = true;
       c_deep_int[i] = false;
       c_cosph_pair[i] = false;
       for(int j = 0; j < 4; j ++)
       {
          e_visited[i][j] = false;
          e_tag[i][j] = false;
          medax_comp_id[i][j] = -1;
          c_VF_on_medax[i][j] = false;
       }
    }
  }

private:

  Point c_voronoi;
  bool  c_cocone_flag[4];
  bool  c_removable[4];
  bool c_VV_on_medax;
  bool c_VF_on_medax[4][4]; // dual VF of edge btn i<->j is in medial axis
  double c_cell_radius;
  bool c_deep_int[4];
  bool c_bb;
  bool c_dirty;
  bool c_cosph_pair[4];
};


struct K : CGAL::Exact_predicates_inexact_constructions_kernel {};

typedef MEDAX_vertex< K >                                       Vertex;
typedef MEDAX_cell< K >                                         Cell;
typedef CGAL::Triangulation_data_structure_3<Vertex, Cell> 	TDS;
typedef CGAL::Delaunay_triangulation_3<K,TDS>  	                Triangulation;

typedef Triangulation::Point                                    Point;
typedef Point::R                          			Rep;
typedef CGAL::Vector_3<Rep>			     		Vector;
typedef CGAL::Ray_3<Rep>                                        Ray;
typedef CGAL::Segment_3<Rep>                                    Segment;
typedef CGAL::Triangle_3<Rep>                                   Triangle_3;
typedef CGAL::Tetrahedron_3<Rep>                     		Tetrahedron;

typedef Triangulation::Vertex_handle              		Vertex_handle;
typedef Triangulation::Edge                       		Edge;
typedef Triangulation::Facet                       		Facet;
typedef Triangulation::Cell_handle 	           		Cell_handle;

typedef Triangulation::Finite_vertices_iterator   		FVI;
typedef Triangulation::Finite_edges_iterator   		        FEI;
typedef Triangulation::Finite_facets_iterator   		FFI;
typedef Triangulation::Finite_cells_iterator   		        FCI;
typedef Triangulation::All_vertices_iterator   		        AVI;
typedef Triangulation::All_edges_iterator   		        AEI;
typedef Triangulation::All_facets_iterator   		        AFI;
typedef Triangulation::All_cells_iterator   		        ACI;

typedef Triangulation::Facet_circulator                         Facet_circulator;
typedef Triangulation::Cell_circulator            		Cell_circulator;

};

#endif //DATASTRUCT_H
