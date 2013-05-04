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

#ifndef __SKELETONIZATION__GRAPH_H__
#define __SKELETONIZATION__GRAPH_H__

#include <Skeletonization/datastruct.h>
#include <Skeletonization/robust_cc.h>

namespace Skeletonization
{
  class GVertex
  {
  public:
    GVertex()
      {
	init();
      }
	
    GVertex(Point p)
      {
	coord = p;
	init();
      }

    void set_point(const Point& p)			{ coord = p; }
    Point point() const				{ return coord; }

    void set_end_of_chain(const bool& b)            { v_end_of_chain = b; }
    bool end_of_chain() const                       { return v_end_of_chain; }

    void set_start_of_chain(const bool& b)          { v_start_of_chain = b; }
    bool start_of_chain() const                     { return v_start_of_chain; }

    void set_on_um_i1(const bool& b)                { v_on_um_i1 = b; }
    bool on_um_i1() const                           { return v_on_um_i1; }

    void add_inc_vert(const int i)			{ inc_vert_list.push_back(i);
      num_inc_vert ++; }
    int inc_vert(int i) const			{ return inc_vert_list[i];}
    bool is_inc_vert(const int v)			
    {
      for(int i = 0; i < num_inc_vert; i ++)
	if(inc_vert_list[i] == v)
	  return true;
      return false;
    }

    void add_inc_edge(const int i)			{ inc_edge_list.push_back(i);
      num_inc_edge ++; }
    int inc_edge(int i) const			{ return inc_edge_list[i];}
    bool get_eid(const int v, int &eid)			
    {
      eid = -1;
      assert(num_inc_vert == num_inc_edge);
      for(int i = 0; i < num_inc_vert; i ++)
	if(inc_vert_list[i] == v)
	  {
	    eid = inc_edge_list[i];
	    return true;
	  }
      return false;
    }

    void set_out( const bool& b )                   { v_out = b; }
    bool out() const                                { return v_out; }

    int id;
    bool visited;
    Cell_handle c;

    int num_inc_vert;
    int num_inc_edge;

    vector<int> cluster_membership;

    inline void init()
    {
      id = -1;
      visited = false;
      v_end_of_chain = false;
      v_start_of_chain = false; 
      v_on_um_i1 = false;
      v_out = false;

      num_inc_vert = 0;
      num_inc_edge = 0;

      cluster_membership.clear();

      inc_vert_list.clear();
      inc_edge_list.clear();
    }

  private:

    Point coord;
    bool v_end_of_chain;
    bool v_start_of_chain; 
    bool v_on_um_i1;

    bool v_out;

    vector<int> inc_vert_list;
    vector<int> inc_edge_list;
  };



  class GEdge 
  {

  public:

    GEdge()
      {
	init();
      }

    GEdge(const int v1, const int v2)
      {
	init();
	endpoint[0] = v1; endpoint[1] = v2;
      }

    void set_endpoint(const int i, const int val)	{ endpoint[i] = val; }
    int get_endpoint(int i) const			{ return endpoint[i]; }

    void set_color(const COLOR& c)                  { e_color = c;}
    COLOR get_color() const                         { return e_color;}

    void set_status(const COMPUTATION_STATUS& s)    { status=s;}
    COMPUTATION_STATUS get_status() const           { return status;}

    bool operator == (GEdge &A)  {
      return (  ((endpoint[0] == A.get_endpoint(0)) && (endpoint[1] == A.get_endpoint(1))) || 
		((endpoint[1] == A.get_endpoint(0)) && (endpoint[0] == A.get_endpoint(1))) );
    }
    int id;

    inline void init()
    {
      id = -1;
    }


  private:
    int endpoint[2];
    COMPUTATION_STATUS status;
    COLOR e_color;
  
  };

  class Graph {

  public:
    Graph()
      {
	init();
      }
    Graph(int v, int c)
      {
	init();
	// initialize the number of vertices and faces.
	nv = v;
	ne = c;
      }
    ~Graph() { vert_list.clear(); edge_list.clear(); }

    void set_nv(const int n)  	{nv = n;}
    int get_nv() const		{return nv;}

    void set_ne(const int n)	{ne = n;}
    int get_ne() const 		{return ne;}

    void add_vertex(GVertex v)      {vert_list.push_back(v); }
    GVertex vertex(int i) const	{return vert_list[i];}

    void add_edge(GEdge e)          {edge_list.push_back(e); }
    GEdge edge(int i) const	        {return edge_list[i];}


    void create(const vector< vector<Cell_handle> >& chains,
		const vector<COMPUTATION_STATUS>& chains_property,
		const vector< Facet >& start_of_chains);
    void erase() {vert_list.clear(); edge_list.clear();}

    vector<GVertex> vert_list;
    vector<GEdge> edge_list;

    inline void init()
    {
      nv = 0;
      ne = 0;

      vert_list.clear();
      edge_list.clear();
    }

  private:
    // number of vertices.
    int nv;
    // number of edges.
    int ne;
  };


}
#endif
