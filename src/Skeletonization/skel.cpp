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

#include <Skeletonization/skel.h>

namespace Skeletonization
{

  extern vector<double> bounding_box;

  void
  cluster_planar_patches(Triangulation& triang, 
			 Skel& skel)
  {
    double max_pgn_width = -DBL_MAX;
    for(FEI eit = triang.finite_edges_begin();
	eit != triang.finite_edges_end(); eit ++)
      {
	Cell_handle c = (*eit).first;
	int uid = (*eit).second, vid = (*eit).third;
	// if already visited, continue.
	if(c->e_visited[uid][vid]) continue;
	// start walk from a VF containing i1 saddle.
	if( ! c->i1_saddle(uid, vid) ) continue;
	CGAL_assertion(c->VF_on_um_i1(uid,vid) );

#ifndef __OUTSIDE__
	// for the purpose of protein structure,
	// we are not including the outside patches into the skeleton.
	if( c->outside ) continue;
#endif

	// mark the VF (dual to e) visited.
	mark_VF_visited(triang, c, uid, vid);

	// push this DE (dual VF) into a stack and start walk.
	vector<Edge> walk;
	walk.push_back((*eit));
	vector<Polygon> pgn_list;

	while( ! walk.empty() )
	  {
	    Cell_handle cur_c = walk.back().first;
	    int cur_uid = walk.back().second;
	    int cur_vid = walk.back().third;
	    walk.pop_back();

	    CGAL_assertion(cur_c->e_visited[cur_uid][cur_vid]);
	    CGAL_assertion(cur_c->VF_on_um_i1(cur_uid, cur_vid));

	    // assign cluster id to this VF.
	    set_patch_id(triang, cur_c, cur_uid, cur_vid, (int)skel.pl.size());

	    // collect the VFs which are on u1 and are connected to 
	    // the current VF (dual e) via a common VE.
	    Facet_circulator fcirc = triang.incident_facets(Edge(cur_c, cur_uid, cur_vid) );
	    Facet_circulator begin = fcirc;
	    // create a polygon and store it in the cluster.
	    Polygon pgn = Polygon();
	    do{
	      Cell_handle new_c = (*fcirc).first;
	      int new_fid = (*fcirc).second;
	      pgn.ordered_v_list.push_back(new_c->voronoi());
	      pgn.cell_list.push_back(new_c);

	      // if this VF has all three incident VFs on um_i1
	      // it is a nonmanifold edge .. do not go through
	      // this edge.
	      if(new_c->VF_on_um_i1((new_fid+1)%4, (new_fid+2)%4) &&
		 new_c->VF_on_um_i1((new_fid+2)%4, (new_fid+3)%4) &&
		 new_c->VF_on_um_i1((new_fid+3)%4, (new_fid+1)%4) )
		{
		  fcirc ++; continue;
		}

	      int new_uid = -1, new_vid = -1;
	      for(int k = 1; k < 4; k ++)
		{
		  if(new_c->vertex((new_fid+k)%4)->id != cur_c->vertex(cur_uid)->id &&
		     new_c->vertex((new_fid+k)%4)->id != cur_c->vertex(cur_vid)->id )
		    {
		      vertex_indices(new_fid, (new_fid+k)%4, new_uid, new_vid);
		      break;
		    }
		}
	      CGAL_assertion(new_uid != -1 && new_vid != -1 && new_uid != new_vid);
	      CGAL_assertion((cur_c->vertex(cur_uid)->id == new_c->vertex(new_uid)->id &&
			      cur_c->vertex(cur_vid)->id == new_c->vertex(new_vid)->id) ||
			     (cur_c->vertex(cur_uid)->id == new_c->vertex(new_vid)->id &&
			      cur_c->vertex(cur_vid)->id == new_c->vertex(new_uid)->id) );
	      int new_wid = 6 - new_fid - new_uid - new_vid;

	      // look for new ones to add to it.
	      if(new_c->VF_on_um_i1(new_uid, new_wid) && !new_c->e_visited[new_uid][new_wid])
		{
		  // take this DE (dual VF).
		  walk.push_back(Edge(new_c, new_uid, new_wid));
		  // mark this DE (dual VF) visited.
		  mark_VF_visited(triang, new_c, new_uid, new_wid);
		}
	      if(new_c->VF_on_um_i1(new_vid, new_wid) && !new_c->e_visited[new_vid][new_wid])
		{
		  // take this DE (dual VF).
		  walk.push_back(Edge(new_c, new_vid, new_wid));
		  // mark it visited.
		  mark_VF_visited(triang, new_c, new_vid, new_wid);
		}
	      fcirc++;
	    } while(fcirc != begin);
	    pgn_list.push_back(pgn);

	    // set the width of the polygon = length of the dual Del edge.
	    double width = sqrt(CGAL::to_double((cur_c->vertex(cur_uid)->point() - cur_c->vertex(cur_vid)->point())*
						(cur_c->vertex(cur_uid)->point() - cur_c->vertex(cur_vid)->point())));
	    pgn_list[(int)pgn_list.size()-1].width = width;

	    if( width > max_pgn_width ) max_pgn_width = width;
	  }
	skel.pl.push_back(pgn_list);
	// compute a weighted centroid of the cluster.
	double total_area = 0;
	Vector centroid = CGAL::NULL_VECTOR;
	for(int i = 0; i < (int)pgn_list.size(); i ++)
	  {
	    total_area += pgn_list[i].area();
	    centroid = centroid + pgn_list[i].area()*(pgn_list[i].centroid() - CGAL::ORIGIN);
	  }
	skel.pl_C.push_back(CGAL::ORIGIN + (1./total_area)*centroid);
	skel.pl_A.push_back(total_area);
	skel.pl_C_knotid.push_back(-1);
      }

    skel.max_pgn_width = max_pgn_width;
  }


  void
  star(const Triangulation& triang,
       const Graph& graph,
       Skel& skel)
  {
    skel.active_bdy.resize((int)skel.pl.size());

    // iterate over the VVs to identify the ones which sit on a planar patch.
    // on the boundary of a planar skeletal VF cluster
    for(FCI cit = triang.finite_cells_begin();
	cit != triang.finite_cells_end(); cit ++)
      {
	if( ! cit->VV_on_um_i1()) continue;

	set<int> inc_clusters;
	inc_clusters.clear();
	for(int i = 0; i < 4; i ++)
	  for(int j = i+1; j < 4; j ++)
            if( cit->VF_on_um_i1(i,j) )
	      {
		if( cit->patch_id[i][j] == -1 )
                  continue;
		inc_clusters.insert(cit->patch_id[i][j]);
	      }
	if( inc_clusters.empty() ) continue;
      
	// VV is in the boundary of two clusters.
	// note this VV in all the incident clusters.
	if((int)inc_clusters.size() > 1) 
	  {
	    for(set<int>::iterator it = inc_clusters.begin();
		it != inc_clusters.end(); it ++)
	      skel.active_bdy[(*it)].push_back(cit->voronoi());
	    continue;
	  }

	// the point on the patch is incident on only one cluster.
	// find out if there is an edge in the graph associated with this vertex
	// which is on a different cluster.
	// if so, this point is active, else it is passive.
	if( cit->g_vid == -1 ) continue;
	GVertex gv = graph.vert_list[cit->g_vid];
	vector<int> C1 = gv.cluster_membership;
	bool is_active = false;
	for(int i = 0; i < gv.num_inc_vert; i ++)
	  {
	    vector<int> C2 = graph.vert_list[gv.inc_vert(i)].cluster_membership;
	    if( ! is_there_any_common_element(C1, C2) )
	      is_active = true;
	  }
	if( is_active ) 
	  skel.active_bdy[(*inc_clusters.begin())].push_back(cit->voronoi());
      }
  }

  void
  sort_cluster_by_area(Skel& skel)
  {
    vector<bool> visited;
    visited.clear(); visited.resize((int)skel.pl_A.size(), false);
    for(int i = 0; i < (int)skel.pl_A.size(); i ++)
      {
	int ind = -1; double max = 0;
	for(int j = 0; j < (int)skel.pl_A.size(); j ++)
	  {
	    if(visited[j]) continue;
	    if(skel.pl_A[j] > max)
	      {
		max = skel.pl_A[j];
		ind = j;
	      }
	  }
	CGAL_assertion(ind != -1); CGAL_assertion(max != -DBL_MAX);
	visited[ind] = true;
	skel.sorted_pl_id.push_back(ind);
      }
  }


  void
  filter_small_clusters(Skel& skel, 
			int pl_cl_cnt,
			double threshold,
			bool discard_by_threshold)
  {
    skel.is_big_pl.clear();
    skel.is_big_pl.resize((int)skel.sorted_pl_id.size(), true);
    if (skel.sorted_pl_id.size() > 0) {
      double max_area = skel.pl_A[skel.sorted_pl_id[0]];
      for(int i = 0; i < (int)skel.sorted_pl_id.size(); i ++)
	{
	  int cl_id = skel.sorted_pl_id[i];
	  if( discard_by_threshold )
	    {
	      if(skel.pl_A[cl_id] < threshold*max_area)
	      skel.is_big_pl[cl_id] = false;
	    }
	  else
	    {
	      if(i >= pl_cl_cnt) 
		{
		  skel.is_big_pl[cl_id] = false;
		}
	      else
		{
		  for(int j = 0; j < (int)skel.pl[cl_id].size(); j ++)
		    for(int k = 0; k < (int)skel.pl[cl_id][j].cell_list.size(); k ++)
		      skel.pl[cl_id][j].cell_list[k]->set_big_pl(true);
		}
	    }
	}
    }
  }

  // Given a graph, create a network of polylines and store it 
  // in the linear portion of skel.
  void
  port_graph_to_skel(Graph& graph, Skel& skel)
  {
    // start with a vertex with degree higher than 2 and walk.
    vector<int> joints_and_leaves;
    for(int i = 0; i < graph.get_nv(); i ++)
      {
#ifndef __OUTSIDE__
	if(graph.vert_list[i].out())
	  continue;
#endif
	if(graph.vert_list[i].num_inc_vert == 1 ||
	   graph.vert_list[i].num_inc_vert > 2 )
	  joints_and_leaves.push_back(i);
      }
 
    // start a walk in all possible directions via the graph edges 
    // from the joints and leaves
    // each walk will terminate in a joint vertex or a terminal vertex
    // and each walk will create a polyline.

    vector< vector<int> > polylines;

    while( ! joints_and_leaves.empty())
      {
	int g_vid = joints_and_leaves.back();
	graph.vert_list[g_vid].visited = true;
	joints_and_leaves.pop_back();

	for(int i = 0; i < graph.vert_list[g_vid].num_inc_vert; i ++)
	  {
	    // polyline will store all the vertices visited in the current walk.
	    vector<int> polyline;

	    GVertex v = graph.vert_list[graph.vert_list[g_vid].inc_vert(i)];

	    // cases.
	    // 0. v is already visited.
	    if(v.visited) continue;
	    // 1. v is another joint.
	    if(v.num_inc_vert > 2)
	      {
		CGAL_assertion(g_vid > v.id);
            
		polyline.push_back(g_vid);
		polyline.push_back(v.id);
		polylines.push_back(polyline);
		continue;
	      }
	    // 2. v is a terminus.
	    if(v.num_inc_vert == 1)
	      {
		polyline.push_back(g_vid);
		polyline.push_back(v.id);
		polylines.push_back(polyline);
		continue;
	      }
	    // therefore v has two incident vertices.
	    polyline.push_back(g_vid);

	    vector<int> walk;
	    walk.push_back(v.id);
	    graph.vert_list[v.id].visited = true;


	    int p_vid = g_vid;
	    while( ! walk.empty() )
	      {
		GVertex cur_v = graph.vert_list[walk.back()];
		walk.pop_back();
		polyline.push_back(cur_v.id);

		CGAL_assertion(cur_v.visited);

		if(cur_v.num_inc_vert != 2) continue;
		int next_vid = cur_v.inc_vert(0) == p_vid?
		  cur_v.inc_vert(1) :
		  cur_v.inc_vert(0);
		walk.push_back(next_vid);
		graph.vert_list[next_vid].visited = true;
		p_vid = cur_v.id;
	      }
	    polylines.push_back(polyline);
	  }
      }

    // if a polyline has a leaf vertex, mark it.
    vector<bool> has_leaf;
    has_leaf.resize((int)polylines.size(), false);
    for(int i = 0; i < (int)polylines.size(); i ++)
      for(int j = 0; j < (int)polylines[i].size(); j ++)
	if(graph.vert_list[polylines[i][j]].num_inc_vert == 1)
	  has_leaf[i] = true;

    // sort the polyines by their lengths.
    vector<double> lengths;
    for(int i = 0; i < (int)polylines.size(); i ++)
      {
	double length = 0;
	for(int j = 0; j < (int)polylines[i].size() - 1; j ++)
	  {
	    GVertex v1 = graph.vert_list[polylines[i][j]];
	    GVertex v2 = graph.vert_list[polylines[i][j+1]];

	    double l = CGAL::to_double((v1.point() - v2.point()) *
				       (v1.point() - v2.point()) );
	    length += l;
	  }
	lengths.push_back(length);
      }
    vector<int> sorted_length_chain_ids;
    vector<bool> visited;
    visited.resize((int)polylines.size(), false);

    for(int i = 0; i < (int)lengths.size(); i ++)
      {
	int ind = -1; double min_length = DBL_MAX;
	for(int j = 0; j < (int)lengths.size(); j ++)
	  {
	    if(visited[j]) continue;
	    if(lengths[j] < min_length) 
	      {
		ind = j;
		min_length = lengths[j]; 
	      }
	  }
	if(ind == -1) { cerr << "error in sorting lengths." << endl; continue; }
	sorted_length_chain_ids.push_back(ind);
	visited[ind] = true;
      }

    // store the sorted polylines in skel.
    skel.L.clear();
    for(int i = 0; i < (int)sorted_length_chain_ids.size(); i ++)
      {
	Polyline l = Polyline();
	int ch_id = sorted_length_chain_ids[i];
	l.patch_id_list.resize( (int)polylines[ch_id].size() );
	for(int j = 0; j < (int)polylines[ch_id].size(); j ++)
	  {
	    GVertex gv = graph.vert_list[polylines[ch_id][j]];
	    l.ordered_v_list.push_back(gv.point());
	    l.cell_list.push_back(gv.c);
	    for(int k = 0; k < (int)gv.cluster_membership.size(); k ++)
	      l.patch_id_list[j].push_back(gv.cluster_membership[k]);
	  }
	skel.L.push_back(l);
	skel.L_invalid.push_back(false);
      }
  }


}
