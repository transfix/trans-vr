/*
  Copyright 2008 The University of Texas at Austin
  
	Authors: Jose Rivera <transfix@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

/* $Id: project_verts.h 1527 2010-03-12 22:10:16Z transfix $ */

/*
  Joe R. - transfix@ices.utexas.edu - project_verts.h - v0.1

  This code simply projects a collection of input vertices to a triangulated surface.
  
  ForwardIterator_vertex - a ForwardIterator that iterates across vertex elements.
                           Each vertex element type should provide operator[] access to
			   vertex components.
  ForwardIterator_tri    - a ForwardIterator that iterates across triangle elements
                           Each triangle element type should provide operator[] access to
			   triangle components.  Each triangle component should be an index
			   into the container pointed to by ForwardIterator_vertex
  InputIterator_vertex   - an InputIterator that iterates across vertex elements.
                           Each vertex element type should provide operator[] access to
			   vertex components.
  InputIterator_tri      - an InputIterator that iterates across triangle elements
                           Each triangle element type should provide operator[] access to
			   triangle components.  Each triangle component should be an index
			   into the container pointed to by InputIterator_vertex

  The first version of project_verts::project projects to the reference surface based on
  the closest point to the input vertices.  The second version projects input points toward
  the normal direction until they hit the reference surface.  The second version requires
  triangle information for the input mesh in order to calculate input normals.

  Example:

  {
    typedef boost::array<double,3> point_t;
    typedef boost::array<int,3>    triangle_t;
    std::vector<point_t> input_verts, reference_verts;
    std::vector<triangle_t> reference_tris;

    //... fill up containers

    project_verts::project(input_verts.begin(),
                           input_verts.end(),
			   reference_verts.begin(),
			   reference_verts.end(),
			   reference_tris.begin(),
			   reference_tris.end());
  }
*/

#ifndef __PROJECT_VERTS_H__
#define __PROJECT_VERTS_H__

#include <iostream>

#include <cmath>
#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <iterator>

#include <CGAL/basic.h>
#include <CGAL/exceptions.h>
#include <CGAL/Timer.h>
#include <CGAL/Simple_cartesian.h>
//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#include <CGAL/K_neighbor_search.h>
//#include <CGAL/Orthogonal_incremental_neighbor_search.h>
#include <CGAL/Incremental_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/squared_distance_3.h>

#include <boost/array.hpp>

namespace project_verts
{
  struct K : CGAL::Simple_cartesian<double> {};
  //struct K : CGAL::Exact_predicates_inexact_constructions_kernel {};
  //struct K : CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt {};

  template <class ForwardIterator_vertex,
            class InputIterator_vertex,
            class InputIterator_tri>
  inline void project(ForwardIterator_vertex input_verts_begin,
		      ForwardIterator_vertex input_verts_end,
		      InputIterator_vertex   reference_verts_begin,
		      InputIterator_vertex   reference_verts_end,
		      InputIterator_tri      reference_tris_begin,
		      InputIterator_tri      reference_tris_end)
    {
      using namespace std;

      //typedef CGAL::Simple_cartesian<double> K;
      typedef K::Point_3 Point_d;
      typedef K::Triangle_3 Triangle_3;
      typedef K::Plane_3 Plane_3;
      typedef K::Point_2 Point_2;
      typedef K::Segment_2  Segment_2;
      typedef CGAL::Search_traits_3<K> TreeTraits;
      typedef CGAL::K_neighbor_search<TreeTraits> Neighbor_search;
      typedef Neighbor_search::Tree Tree;

      typedef boost::array<unsigned int,3> triangle_t;

      Tree tree;

      vector<Point_d> reference_verts;
      vector<triangle_t> reference_tris;

      {
	for(InputIterator_vertex i = reference_verts_begin;
	    i != reference_verts_end;
	    i++)
	  reference_verts.push_back(Point_d((*i)[0],(*i)[1],(*i)[2]));
	tree = Tree(reference_verts.begin(), reference_verts.end());
      }

      //copy reference tris and find all the triangles that each point is a part of
      map<Point_d, list<unsigned int> > neighbor_tris;
      for(InputIterator_tri i = reference_tris_begin;
	  i != reference_tris_end;
	  i++)
	{
	  triangle_t tri;
	  copy(i->begin(),i->end(),tri.begin());
	  reference_tris.push_back(tri);
	  for(triangle_t::iterator j = tri.begin();
	      j != tri.end();
	      j++)
	    neighbor_tris[reference_verts[*j]]
	      .push_back(distance(reference_tris_begin,i));
	}

      //now iterate through all input vertices and project them to the closest face
      //of reference geom
      for(ForwardIterator_vertex i = input_verts_begin;
	  i != input_verts_end;
	  i++)
	{
	  Point_d query((*i)[0],(*i)[1],(*i)[2]);
	  Neighbor_search search(tree,query,10); //instead of 10, consider using reference_verts.size() if it's not too slow...

	  if(search.begin() == search.end())
	    throw runtime_error("Neighbor search yielded zero results. Kd-tree empty?");

	  Point_d closest_point_with_tris = search.begin()->first;
	  for(Neighbor_search::iterator ni = search.begin();
	      ni != search.end() && neighbor_tris[closest_point_with_tris].empty();
	      ni++)
	    closest_point_with_tris = ni->first;

	  list<unsigned int> &closest_tris = neighbor_tris[closest_point_with_tris];
	  if(closest_tris.empty()) continue; //this might happen if we have a lot of free points

	  //now find the closest tri in the list via the closest point on each tri to the query
	  unsigned int tri_idx = closest_tris.front();
	  Triangle_3 closest_tri(reference_verts[reference_tris[tri_idx][0]],
				 reference_verts[reference_tris[tri_idx][1]],
				 reference_verts[reference_tris[tri_idx][2]]);
	  Point_d closest_point = closest_tri.supporting_plane().projection(query);
	  if(!closest_tri.has_on(closest_point))
	    {
	      //if the point isn't within the triangle after projection, 
	      //move it to the closest point on the triangle
	      //which would be on one of the triangle's edges
	      Plane_3 closest_tri_plane = closest_tri.supporting_plane();
	      Segment_2 tri_lines[3] = { Segment_2(closest_tri_plane.to_2d(closest_tri[0]),
						   closest_tri_plane.to_2d(closest_tri[1])),
					 Segment_2(closest_tri_plane.to_2d(closest_tri[1]),
						   closest_tri_plane.to_2d(closest_tri[2])),
					 Segment_2(closest_tri_plane.to_2d(closest_tri[2]),
						   closest_tri_plane.to_2d(closest_tri[0])) };
	      
	      Point_2 closest_points_on_current_line[3];
	      for(int j = 0; j < 3; j++)
		{
		  Point_2 P0 = tri_lines[j].source();
		  Point_2 P1 = tri_lines[j].target();
		  Point_2 PS = closest_tri_plane.to_2d(query);
		  double r = CGAL::to_double(-(((P1.x() - P0.x())*(P0.x()+PS.x()) +
						(P1.y() - P0.y())*(P0.y()+PS.y())) /
					       CGAL::squared_distance(P1,P0)));
		  r = max(1.0,min(0.0,r)); //clamp r so our point is on the tri segment
		  closest_points_on_current_line[j] = P0 + r*(P1-P0);
		}

	      //calculate distances for the 3 closest points and grab the absolute closest
	      map<double, Point_2> distances;
	      for(int j = 0; j < 3; j++)
		distances[CGAL::to_double(CGAL::squared_distance(closest_points_on_current_line[j],
								 closest_tri_plane.to_2d(query)))]
		  = closest_points_on_current_line[j];
	      closest_point = closest_tri_plane.to_3d(distances.begin()->second);
	    }
	  
	  for(list<unsigned int>::iterator j = closest_tris.begin();
	      j != closest_tris.end();
	      j++)
	    {
	      Triangle_3 current_tri(reference_verts[reference_tris[*j][0]],
				     reference_verts[reference_tris[*j][1]],
				     reference_verts[reference_tris[*j][2]]);
	      Point_d new_closest_point = current_tri.supporting_plane().projection(query);
	      if(!current_tri.has_on(new_closest_point))
		{
		  Plane_3 current_tri_plane = current_tri.supporting_plane();
		  Segment_2 tri_lines[3] = { Segment_2(current_tri_plane.to_2d(current_tri[0]),
						       current_tri_plane.to_2d(current_tri[1])),
					     Segment_2(current_tri_plane.to_2d(current_tri[1]),
						       current_tri_plane.to_2d(current_tri[2])),
					     Segment_2(current_tri_plane.to_2d(current_tri[2]),
						       current_tri_plane.to_2d(current_tri[0])) };
		  Point_2 closest_points_on_current_line[3];
		  for(int j = 0; j < 3; j++)
		    {
		      Point_2 P0 = tri_lines[j].source();
		      Point_2 P1 = tri_lines[j].target();
		      Point_2 PS = current_tri_plane.to_2d(query);
		      double r = CGAL::to_double(-(((P1.x() - P0.x())*(P0.x()+PS.x()) +
						    (P1.y() - P0.y())*(P0.y()+PS.y()))  /
						   CGAL::squared_distance(P1,P0)));
		      r = max(1.0,min(0.0,r)); //clamp r so our point is on the tri segment
		      closest_points_on_current_line[j] = P0 + r*(P1-P0);
		    }

		  //calculate distances for the 3 closest points and grab the absolute closest
		  map<double, Point_2> distances;
		  for(int j = 0; j < 3; j++)
		    distances[CGAL::to_double(CGAL::squared_distance(closest_points_on_current_line[j],
								     current_tri_plane.to_2d(query)))]
		      = closest_points_on_current_line[j];
		  new_closest_point = current_tri_plane.to_3d(distances.begin()->second);
		}
	      
	      if(CGAL::squared_distance(new_closest_point,query) <
		 CGAL::squared_distance(closest_point,query))
		{
		  closest_point = new_closest_point;
		  closest_tri = current_tri;
		}
	    }
	  
	  (*i)[0] = CGAL::to_double(closest_point.x());
	  (*i)[1] = CGAL::to_double(closest_point.y());
	  (*i)[2] = CGAL::to_double(closest_point.z());
	}
    }

  struct on_negative_side
  {
    typedef CGAL::Search_traits_3<K> TreeTraits;
    typedef CGAL::Incremental_neighbor_search<TreeTraits> NN_incremental_search;
    typedef NN_incremental_search::iterator NN_iterator;
    typedef K::Plane_3 Plane_3;

    on_negative_side(const Plane_3& plane) : _plane(plane) {}
    bool operator()( NN_iterator& it)
    {
      return _plane.has_on_negative_side((*it).first);
    }

    private:
    Plane_3 _plane;
  };

  template <class ForwardIterator_vertex,
            class ForwardIterator_tri,
            class InputIterator_vertex,
            class InputIterator_tri>
  inline void project(ForwardIterator_vertex input_verts_begin,
		      ForwardIterator_vertex input_verts_end,
		      ForwardIterator_tri    input_tris_begin,
		      ForwardIterator_tri    input_tris_end,
		      InputIterator_vertex   reference_verts_begin,
		      InputIterator_vertex   reference_verts_end,
		      InputIterator_tri      reference_tris_begin,
		      InputIterator_tri      reference_tris_end)
    {
      using namespace std;

      //typedef CGAL::Simple_cartesian<double> K;
      typedef K::Point_3 Point_3;
      typedef K::Vector_3 Vector_3;
      typedef K::Triangle_3 Triangle_3;
      typedef K::Plane_3 Plane_3;
      typedef K::Point_2 Point_2;
      typedef K::Segment_2  Segment_2;
      typedef CGAL::Search_traits_3<K> TreeTraits;
      //typedef CGAL::K_neighbor_search<TreeTraits> Neighbor_search;
      //typedef Neighbor_search::Tree Tree;
      //typedef CGAL::Orthogonal_incremental_neighbor_search<TreeTraits> NN_incremental_search;
      typedef CGAL::Incremental_neighbor_search<TreeTraits> NN_incremental_search;
      typedef NN_incremental_search::iterator NN_iterator;
      typedef NN_incremental_search::Tree Tree;
      typedef CGAL::Filter_iterator<NN_iterator, on_negative_side> NN_positive_side_iterator;

      typedef boost::array<unsigned int,3> triangle_t;

      CGAL::Timer timer;
      //Tree reference_tree;

      vector<Point_3> reference_verts, input_verts;
      vector<triangle_t> reference_tris, input_tris;

      cerr << "Copying the reference verts and building kd-tree... ";
      timer.reset(); timer.start();
      //copy the reference verts and build a kd-tree
      for(InputIterator_vertex i = reference_verts_begin;
	  i != reference_verts_end;
	  i++)
	reference_verts.push_back(Point_3((*i)[0],(*i)[1],(*i)[2]));
      Tree reference_tree(reference_verts.begin(), reference_verts.end());
      timer.stop();
      cerr << "Done. " << timer.time() << " seconds" << endl;
      
      cerr << "Copying the input verts... ";
      timer.reset(); timer.start();
      //copy the input verts
      for(ForwardIterator_vertex i = input_verts_begin;
	  i != input_verts_end;
	  i++)
	input_verts.push_back(Point_3((*i)[0],(*i)[1],(*i)[2]));
      timer.stop();
      cerr << "Done. " << timer.time() << " seconds" << endl;

      cerr << "Copying the reference tris and building neighbor directory... ";
      timer.reset(); timer.start();
      //copy reference tris and find all the triangles that each point is a part of
      map<Point_3, list<unsigned int> > reference_neighbor_tris;
      for(InputIterator_tri i = reference_tris_begin;
	  i != reference_tris_end;
	  i++)
	{
	  triangle_t tri;
	  copy(i->begin(),i->end(),tri.begin());
	  reference_tris.push_back(tri);
	  for(triangle_t::iterator j = tri.begin();
	      j != tri.end();
	      j++)
	    reference_neighbor_tris[reference_verts[*j]]
	      .push_back(distance(reference_tris_begin,i));
	}
      timer.stop();
      cerr << "Done. " << timer.time() << " seconds" << endl;

      cerr << "Copying the input tris and building neighbor directory... ";
      timer.reset(); timer.start();
      //copy input tris and find all the triangles that each point is a part of
      map<Point_3, list<unsigned int> > input_neighbor_tris;
      for(ForwardIterator_tri i = input_tris_begin;
	  i != input_tris_end;
	  i++)
	{
	  triangle_t tri;
	  copy(i->begin(),i->end(),tri.begin());
	  input_tris.push_back(tri);
	  for(triangle_t::iterator j = tri.begin();
	      j != tri.end();
	      j++)
	    input_neighbor_tris[input_verts[*j]]
	      .push_back(distance(input_tris_begin,i));
	}
      timer.stop();
      cerr << "Done. " << timer.time() << " seconds" << endl;

      //For each input vertex, calculate a normal and find the intersection point
      //of that normal with the reference surface.  Then move the input vertex to
      //that intersection point.  Note: we are assuming the triangle vertex ordering
      //is already such that neighboring normals make sense.  If your mesh doesn't
      //guarantee that, fix it!
      cerr << "Projecting: \r";
      for(vector<Point_3>::iterator i = input_verts.begin();
	  i != input_verts.end();
	  i++)
	{
	  Vector_3 normal;
	  list<unsigned int> &tri_indices = input_neighbor_tris[*i];
	  for(list<unsigned int>::iterator j = tri_indices.begin();
	      j != tri_indices.end();
	      j++)
	    normal = normal + Triangle_3(input_verts[input_tris[*j][0]],
					 input_verts[input_tris[*j][1]],
					 input_verts[input_tris[*j][2]])
	      .supporting_plane().orthogonal_vector();
	  normal = normal / sqrt(CGAL::to_double(normal.squared_length()));

	  //search for intersecting triangles from nearest to farthest
	  try
	    {
	      NN_incremental_search search(reference_tree,*i);

	      for(NN_iterator j = search.begin();
		  j != search.end();
		  j++)
#if 0
	      //Here we are skipping all verts on the negative side of the plane
	      //resulting from our input vertex and normal.
	      for(NN_positive_side_iterator
		    j(search.end(),
		      on_negative_side(Plane_3(*i,normal)),
		      search.begin()),
		    end(search.end(), on_negative_side(Plane_3(*i,normal)));
		  j != end;
		  j++)
#endif
		{
		  list<unsigned int> &reference_tri_indices = reference_neighbor_tris[(*j).first];
		  for(list<unsigned int>::iterator k = reference_tri_indices.begin();
		      k != reference_tri_indices.end();
		      k++)
		    {
		      //http://ozviz.wasp.uwa.edu.au/~pbourke/geometry/planeline/
		      
		      Triangle_3 cur_tri(reference_verts[reference_tris[*k][0]],
					 reference_verts[reference_tris[*k][1]],
					 reference_verts[reference_tris[*k][2]]);
		      Plane_3 cur_plane = cur_tri.supporting_plane();
#if 0
		      Point_3 P = *i;
		      K::RT u = 
			-(cur_plane.a()*P.x() + cur_plane.b()*P.y() +
			  cur_plane.c()*P.z() + cur_plane.d()) /
			(cur_plane.a()*normal.x() + 
			 cur_plane.b()*normal.y() + 
			 cur_plane.c()*normal.z());
		      Point_3 P_proj = P + u * normal;
		      //To ensure cur_tri.has_on() works, lets explicitly project P_proj to the tri's plane.
		      //We should be very close to the plane before projection, but not necessarily exactly
		      //on it due to precision error...
		      P_proj = cur_plane.projection(P_proj);
		      if(cur_tri.has_on(P_proj))
			{
			  throw P_proj;
			}
#endif

		      Point_3 P1 = *i;
		      Point_3 P2 = *i + normal;
		      K::RT u =
			(cur_plane.a()*P1.x() + cur_plane.b()*P1.y() +
			 cur_plane.c()*P1.z() + cur_plane.d()) /
			(cur_plane.a()*(P1.x() - P2.x()) +
			 cur_plane.b()*(P1.y() - P2.y()) +
			 cur_plane.c()*(P1.z() - P2.z()));
		      Point_3 P = P1 + u*(P2 - P1);
		      Point_3 P_proj = cur_plane.projection(P);
		      if(cur_tri.has_on(P_proj))
			{
			  throw P_proj;
			}
		    }
		}
	    }
	  catch(Point_3& P_Proj) //we intersected with some tri!
	    {
	      *i = P_Proj;
	    }
	  
	  cerr << "Projecting: " << ((i-input_verts.begin())/(input_verts.size()-1))*100.0 << "\r";
	}
      cerr << endl;

      //copy the new input verts back into the original container
      vector<Point_3>::iterator cur_pt = input_verts.begin();
      for(ForwardIterator_vertex i = input_verts_begin;
	  i != input_verts_end;
	  i++)
	{
	  (*i)[0] = CGAL::to_double(cur_pt->x());
	  (*i)[1] = CGAL::to_double(cur_pt->y());
	  (*i)[2] = CGAL::to_double(cur_pt->z());
	  cur_pt++;
	}
    }
}

#endif
