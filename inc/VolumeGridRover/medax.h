/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: medax.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef __MEDAX_H__
#define __MEDAX_H__

#include <CGAL/basic.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <string>
#include <set>
#include <vector>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Filtered_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Segment_Delaunay_graph_traits_2.h>
#include <CGAL/Segment_Delaunay_graph_vertex_base_2.h>
//#include <CGAL/Segment_Delaunay_graph_2.h>
#include <CGAL/Segment_Delaunay_graph_hierarchy_2.h>
#include <CGAL/Segment_Delaunay_graph_hierarchy_vertex_base_2.h>
#include <CGAL/Segment_Delaunay_graph_adaptation_traits_2.h>
//#include <CGAL/Segment_Delaunay_graph_adaptation_policies_2.h>
#include <CGAL/Voronoi_diagram_2.h>

#include <CGAL/intersections.h>

#include <boost/tuple/tuple.hpp>

//This code crashes gcc v3.4.6!

namespace medax
{
  //this following vertex class isn't really used because I found out the Delaunay vertex object is different
  //than the Voronoi vertex object :/  However I'm going to keep this here because it was a PITA to get template
  //typedefs right.  TODO: Email CGAL people and tell them their documentation needs to be updated to reflect
  //the Site storage traits parameter, among other things!
  template < class Gt, class Vbb, class Vb = CGAL::Segment_Delaunay_graph_hierarchy_vertex_base_2<Vbb> >
  class VC_vertex : public Vb
  {
    typedef Vb Base;
    mutable bool _keep; //flag to determine if a halfedge from this vertex should be dropped
   public:
    typedef typename Gt::Point_2  Point;
    typedef typename Vb::Face_handle Face_handle;

    VC_vertex() : _keep(false) {}
    VC_vertex(const Point& p) : Base(p), _keep(false) {}
    VC_vertex(const Point& p, Face_handle f) : Base(p,f), _keep(false) {}
    
    void keep(bool k) const { _keep = k; }
    bool keep() const { return _keep; }
  };

  typedef CGAL::Simple_cartesian<double>    CK;
  typedef CGAL::Filtered_kernel<CK>         Kernel;
  typedef CGAL::Segment_Delaunay_graph_traits_2<Kernel>  Gt;
  typedef CGAL::Segment_Delaunay_graph_storage_traits_2<Gt> SDGST2; //undocumented?????
  typedef VC_vertex<Gt,CGAL::Segment_Delaunay_graph_vertex_base_2<SDGST2> >                 Vertex;
  typedef CGAL::Triangulation_data_structure_2<Vertex,CGAL::Triangulation_face_base_2<Gt> > TDS2;
  //typedef CGAL::Segment_Delaunay_graph_2<Gt,TDS2>             Delaunay_graph_2;
  typedef CGAL::Segment_Delaunay_graph_hierarchy_2<Gt,SDGST2,CGAL::Tag_false,TDS2>  Delaunay_graph_2;
  typedef CGAL::Segment_Delaunay_graph_adaptation_traits_2<Delaunay_graph_2> At;
  //typedef CGAL::Segment_Delaunay_graph_caching_degeneracy_removal_policy_2<Delaunay_graph_2> Ap;
  typedef CGAL::Voronoi_diagram_2<Delaunay_graph_2,At/*,Ap*/> Voronoi_diagram_2;

  typedef At::Site_2 Site_2;
  typedef At::Point_2 Point_2;
  typedef Site_2::Segment_2 Segment_2;
  typedef Voronoi_diagram_2::Vertex_handle Vertex_handle;
  typedef Voronoi_diagram_2::Halfedge_handle Halfedge_handle;
  typedef Voronoi_diagram_2::Face_handle Face_handle;
  typedef Voronoi_diagram_2::Ccb_halfedge_circulator   Ccb_halfedge_circulator;

  //typedef std::set<Halfedge_handle> Halfedges;
  typedef boost::tuple<Point_2,Point_2> Simple_edge;
  typedef std::set<Simple_edge> Edges;

  //the medial axis computation returns the voronoi diagram computed from
  //the points of the input set of edges, and the subset of voronoi edges
  //that do not intersect the input set of edges (the inner and outer medial axes)
  //typedef boost::tuple<Voronoi_diagram_2,Halfedges> Medax_result;

  typedef CGAL::Polygon_2<Kernel> Polygon_2;
  typedef Polygon_2::Point_2 Poly_point_2;
  typedef std::vector<Polygon_2> Polygons;

  Edges computeDelaunay(const Polygons& p)
  {
    Delaunay_graph_2 dg;
    Edges edges;

    return edges;
  }

  Edges computeVoronoi(const Polygons& p)
  {
    using namespace std;
    Voronoi_diagram_2 vd;

    fprintf(stderr,"Adding polygon vertices to the voronoi diagram...\n");
    //add all poly vertices to the diagram
    for(Polygons::const_iterator polygon = p.begin();
	polygon != p.end();
	polygon++)
      {
	fprintf(stderr,"%5.2f %%\r",(float(polygon-p.begin())/float(p.size()-1))*100.0);

#ifndef MEDAX_INSERT_EDGES
	for(Polygon_2::Vertex_iterator vert = polygon->vertices_begin();
	    vert != polygon->vertices_end();
	    vert++)
	  vd.insert(Site_2::construct_site_2(Point_2(vert->x(),vert->y())));
#else
	for(Polygon_2::Edge_const_iterator edge = polygon->edges_begin();
	    edge != polygon->edges_end();
	    edge++)
	  vd.insert(Site_2::construct_site_2(Point_2(edge->source().x(),
						     edge->source().y()),
					     Point_2(edge->target().x(),
						     edge->target().y())));
#endif
      }
    fprintf(stderr,"\nDone\n");

    assert(vd.is_valid());

    fprintf(stderr,"Building edge set from voronoi faces...\n");
    Edges edges;
    for(Voronoi_diagram_2::Face_iterator face = vd.faces_begin();
	face != vd.faces_end();
	face++)
      {
	Ccb_halfedge_circulator halfedge_cir = face->outer_ccb();
	do
	  {
	    Halfedge_handle halfedge(halfedge_cir);
	    if(halfedge->is_segment() && halfedge->is_valid()) //only consider bounded edges
	      edges.insert(Simple_edge(halfedge->source()->point(),
				       halfedge->target()->point()));
	    halfedge_cir++;
	  }
	while(halfedge_cir != face->outer_ccb());
      }
    fprintf(stderr,"\nDone\n");

    return edges;
  }

  Edges compute(const Polygons& p)
  {
    using namespace std;

    Voronoi_diagram_2 vd;

    fprintf(stderr,"Adding polygon vertices to the voronoi diagram...\n");
    //add all poly vertices to the diagram
    for(Polygons::const_iterator polygon = p.begin();
	polygon != p.end();
	polygon++)
      {
	fprintf(stderr,"%5.2f %%\r",(float(polygon-p.begin())/float(p.size()-1))*100.0);

#ifndef MEDAX_INSERT_EDGES
	for(Polygon_2::Vertex_iterator vert = polygon->vertices_begin();
	    vert != polygon->vertices_end();
	    vert++)
	  vd.insert(Site_2::construct_site_2(Point_2(vert->x(),vert->y())));
#else
	for(Polygon_2::Edge_const_iterator edge = polygon->edges_begin();
	    edge != polygon->edges_end();
	    edge++)
	  vd.insert(Site_2::construct_site_2(Point_2(edge->source().x(),
						     edge->source().y()),
					     Point_2(edge->target().x(),
						     edge->target().y())));
#endif
      }
    fprintf(stderr,"\nDone\n");

    assert(vd.is_valid());

    //Halfedges halfedges;
    Edges edges;

#ifdef MEDAX_INSERT_EDGES
    //debugging
    int vert_cnt = 0;
    int halfedge_cnt = 0;
    int face_cnt = 0;
#endif

    fprintf(stderr,"Pruning voronoi diagram to obtain medial axis...\n");
    //insert edges into the list for each voronoi halfedge that doesn't intersect any poly edges
    for(Polygons::const_iterator polygon = p.begin();
	polygon != p.end();
	polygon++)
      {
	fprintf(stderr,"%5.2f %%\r",(float(polygon-p.begin())/float(p.size()-1))*100.0);

	for(Polygon_2::Edge_const_iterator edge = polygon->edges_begin();
	    edge != polygon->edges_end();
	    edge++)
	  {
	    //find the voronoi face of each contour/polygon vertex
	    Voronoi_diagram_2::Locate_result result = 
	      vd.locate(Point_2(edge->source().x(),edge->source().y()));

#ifndef MEDAX_INSERT_EDGES
	    //locate should always return a face in this case
	    //since we are trying to grab the voronoi face of each input site (point)
	    assert(boost::get<Face_handle>(&result));

	    Face_handle *f = boost::get<Face_handle>(&result);
	    Ccb_halfedge_circulator halfedge_cir = (*f)->outer_ccb();
	    do
	      {
		Halfedge_handle halfedge(halfedge_cir);

		if(halfedge->is_segment()) //only consider bounded edges
		  {
#ifdef MEDAX_ALWAYS_INSERT
		    edges.insert(Simple_edge(halfedge->source()->point(),
						halfedge->target()->point()));
#else
		    bool intersected = false;

		    //check intersection with all edges of contour
		    for(Polygon_2::Edge_const_iterator inner_edge = polygon->edges_begin();
			inner_edge != polygon->edges_end();
			inner_edge++)
		      {
			CGAL::Object result = CGAL::intersection(Segment_2(halfedge->source()->point(),
									   halfedge->target()->point()),
								 Segment_2(Point_2(inner_edge->source().x(),inner_edge->source().y()),
									   Point_2(inner_edge->target().x(),inner_edge->target().y())));
			
			{
			  Segment_2 tmp_seg;
			  assert(!CGAL::assign(tmp_seg, result));
			}

			Point_2 tmp_point;
			if(CGAL::assign(tmp_point, result))
			  {
			    intersected = true;
			    break;
			  }
		      }

		    if(!intersected)
		      edges.insert(Simple_edge(halfedge->source()->point(),
						  halfedge->target()->point()));
#endif
		  }

		halfedge_cir++;
	      }
	    while(halfedge_cir != (*f)->outer_ccb());
#else
	    {
	      Vertex_handle *v = boost::get<Vertex_handle>(&result);
	      Halfedge_handle *e = boost::get<Halfedge_handle>(&result);
	      Face_handle *f = boost::get<Face_handle>(&result);
	      
	      if(e)
		{
		  Halfedge_handle halfedge(*e);
		  
		  if(halfedge->is_segment()) //only consider bounded edges
		    {
#ifdef MEDAX_ALWAYS_INSERT
		      edges.insert(Simple_edge(halfedge->source()->point(),
						  halfedge->target()->point()));
#else
		      bool intersected = false;
		      
		      //check intersection with all edges of contour
		      for(Polygon_2::Edge_const_iterator inner_edge = polygon->edges_begin();
			  inner_edge != polygon->edges_end();
			  inner_edge++)
			{
			  CGAL::Object result = CGAL::intersection(Segment_2(halfedge->source()->point(),
									     halfedge->target()->point()),
								   Segment_2(Point_2(inner_edge->source().x(),inner_edge->source().y()),
									     Point_2(inner_edge->target().x(),inner_edge->target().y())));
			  
			  {
			    Segment_2 tmp_seg;
			    assert(!CGAL::assign(tmp_seg, result));
			  }
			  
			  Point_2 tmp_point;
			  if(CGAL::assign(tmp_point, result))
			    {
			      intersected = true;
			      break;
			    }
			}
		      
		      if(!intersected)
			edges.insert(Simple_edge(halfedge->source()->point(),
						    halfedge->target()->point()));
#endif
		    }

		  halfedge_cnt++;
		}
	      else if(f)
		{
		  Ccb_halfedge_circulator halfedge_cir = (*f)->outer_ccb();
		  do
		    {
		      Halfedge_handle halfedge(halfedge_cir);

		      if(halfedge->is_segment()) //only consider bounded edges
			{
#ifdef MEDAX_ALWAYS_INSERT
			  edges.insert(Simple_edge(halfedge->source()->point(),
						      halfedge->target()->point()));
#else
			  bool intersected = false;

			  //check intersection with all edges of contour
			  for(Polygon_2::Edge_const_iterator inner_edge = polygon->edges_begin();
			      inner_edge != polygon->edges_end();
			      inner_edge++)
			    {
			      CGAL::Object result = CGAL::intersection(Segment_2(halfedge->source()->point(),
										 halfedge->target()->point()),
								       Segment_2(Point_2(inner_edge->source().x(),inner_edge->source().y()),
										 Point_2(inner_edge->target().x(),inner_edge->target().y())));
			
			      {
				Segment_2 tmp_seg;
				assert(!CGAL::assign(tmp_seg, result));
			      }

			      Point_2 tmp_point;
			      if(CGAL::assign(tmp_point, result))
				{
				  intersected = true;
				  break;
				}
			    }

			  if(!intersected)
			    edges.insert(Simple_edge(halfedge->source()->point(),
							halfedge->target()->point()));
#endif
			}

		      halfedge_cir++;
		    }
		  while(halfedge_cir != (*f)->outer_ccb());

		  face_cnt++;
		}
	      else if(v)
		{
		  vert_cnt++;
		}
	    }
#endif
	  }
      }
#ifdef MEDAX_INSERT_EDGES
    fprintf(stderr,"\nDone (vert:%d,halfedge:%d,face:%d)\n",vert_cnt,halfedge_cnt,face_cnt);
#endif

    //return Medax_result(vd,halfedges);
    //return vd;
    return edges;
  }
}

#endif
