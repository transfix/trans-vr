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

#ifndef __SKELETONIZATION__SKEL_H__
#define __SKELETONIZATION__SKEL_H__

#include <Skeletonization/datastruct.h>
#include <Skeletonization/graph.h>
#include <Skeletonization/util.h>
#include <Skeletonization/hfn_util.h>

namespace Skeletonization
{

  class Polygon
  {
  public:
    Polygon() { init(); }
    ~Polygon() { ordered_v_list.clear();}

    vector<Point> ordered_v_list;
    vector<Cell_handle> cell_list;
    double width;

    Point centroid()
    {
      Vector c = CGAL::NULL_VECTOR;
      for(int i = 0; i < (int)ordered_v_list.size(); i ++)
	c = c + (ordered_v_list[i] - CGAL::ORIGIN);
      return (CGAL::ORIGIN + (1./(int) ordered_v_list.size())*c);
    }

    double area()
    {
      Point c = centroid();
      double A = 0;
      for(int i = 0; i < (int)ordered_v_list.size(); i ++)
	{
	  Triangle_3 t = Triangle_3(c, ordered_v_list[i], 
				    ordered_v_list[(i+1)%((int)ordered_v_list.size())]);
	  A += sqrt(CGAL::to_double(t.squared_area()));
	}
      return A;
    }

    inline void init()
      {
	ordered_v_list.clear();
      }
  private:
  };

  class Polyline
  {
  public:
    Polyline() { init(); }
    ~Polyline() 
      { 
	ordered_v_list.clear();
	patch_id_list.clear();
      }

    vector<Point> ordered_v_list;
    vector< vector<int> > patch_id_list; // stores the list of patches the vertex falls into.
    // the list is empty if the vertex is free.
    vector<int> cell_id_list;
    vector<Cell_handle> cell_list;
    vector<bool> is_max;

    int knot_ids[2];
    inline void init()
      {
	ordered_v_list.clear();
	patch_id_list.clear();
      }
  private:
  };

  class Knot
  {
  public:
    Knot();
    Knot(const Point& p) { pos = p; }
 
    Point pos;
    vector<int> inc_l;
  };

  class Skel {

  public:
    Skel() { init();}
    ~Skel() 
      { 
	L.clear();

	pl.clear();
	pl_C.clear();
	pl_A.clear();
	sorted_pl_id.clear();
	is_big_pl.clear();
	active_bdy.clear();
      }

    double max_pgn_width;

    vector<Knot> knot_vector;
    vector<bool> knot_invalid;

    vector< Polyline > L;
    vector<bool> L_invalid;

    vector< vector<Polygon> > pl;
    vector< Point > pl_C;
    vector< double > pl_A;
    vector< int > sorted_pl_id;
    vector<bool> is_big_pl;
    vector< vector<Point> > active_bdy;
    vector<int> pl_C_knotid;

    inline void init()
      {
	max_pgn_width = -DBL_MAX;
	L.clear();

	pl.clear();
	pl_C.clear();
	pl_A.clear();
	sorted_pl_id.clear();
	is_big_pl.clear();
	active_bdy.clear();
	pl_C_knotid.clear();
      }
  private:
  };


  void
    cluster_planar_patches(Triangulation& triang,
			   Skel& skel);

  void
    star(const Triangulation& triang,
	 const Graph& graph,
	 Skel& skel);

  void
    sort_cluster_by_area(Skel& skel);

  void
    filter_small_clusters(Skel& skel, 
			  int pl_cl_cnt = 2,
			  double threshold = 0.1, 
			  bool discard_by_threshold = false);

  void
    port_graph_to_skel(Graph& graph, Skel& skel);

  void
    identify_pl_cyl_vertices(Triangulation& triang,
			     const Skel& skel);

  void
    update_L(Skel& skel);

  void
    process_L(Skel& skel);

  void
    create_knots(Skel& skel);

  void
    prune_L(Skel& skel);

}

#endif // SKEL_H
