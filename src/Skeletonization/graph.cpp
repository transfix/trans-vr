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

#include <Skeletonization/graph.h>

namespace Skeletonization
{

void Graph::create(const vector< vector<Cell_handle> >& chains,
		   const vector<COMPUTATION_STATUS>& chains_property,
                   const vector< Facet >& start_of_chains)
{
    // `chains' is a collection of chains each of which is given by a list
    // of ordered vertices on it. the edges are therefore implicitly defined
    // between two consecutive vertices in the list. It has impurity because
    // some edges are not correct. So, we collect only the correct edges
    // and remove any duplication. We call it pure_chains.
    vector< vector<int> > pure_chains;
    pure_chains.resize((int)chains.size());
    int current_pure_chain = -1;

    // we first create a set of vertices. we omit duplication at the start
    // and end of chains by consulting the vector start_end_of_chains.
    for(int i = 0; i < (int)chains.size(); i ++)
    {
       if(chains_property[i] != SUCCESS) continue;
       if((int)chains[i].size() == 0) continue;

       current_pure_chain++;

       Facet i2f = start_of_chains[i];
       if( i2f.first->saddle_g_vid[i2f.second] == -1)
       {
          vert_list.push_back( GVertex(circumcenter(start_of_chains[i])) );
          vert_list[(int)vert_list.size()-1].id = (int)vert_list.size()-1;
          vert_list[(int)vert_list.size()-1].c = i2f.first;
          pure_chains[current_pure_chain].push_back((int)vert_list.size()-1);

          Cell_handle c[2]; int id[2];
          c[0] = i2f.first; id[0] = i2f.second;
          c[1] = c[0]->neighbor(id[0]); id[1] = c[1]->index(c[0]);

          c[0]->saddle_g_vid[id[0]] = (int)vert_list.size()-1;
          c[1]->saddle_g_vid[id[1]] = (int)vert_list.size()-1;

          vert_list[(int)vert_list.size()-1].set_out(c[0]->outside && c[1]->outside );

          // if either of the three VFs incident on the VE (dual to Face(c[0], id[0]))
          // is on_um_i1, this graph vertex is also on um_i1.
          int u = (id[0]+1)%4, v = (id[0]+2)%4, w = (id[0]+3)%4;
          if(c[0]->VF_on_um_i1(u,v) || 
             c[0]->VF_on_um_i1(v,w) || 
             c[0]->VF_on_um_i1(w,u) )
                vert_list[(int)vert_list.size()-1].set_on_um_i1(true);
          // collect the clusters the incident VFs fall into.
          if(c[0]->patch_id[u][v] != -1)
             vert_list[(int)vert_list.size()-1].cluster_membership.push_back(c[0]->patch_id[u][v]);
          if(c[0]->patch_id[v][w] != -1 &&
             c[0]->patch_id[v][w] !=  c[0]->patch_id[u][v])
             vert_list[(int)vert_list.size()-1].cluster_membership.push_back(c[0]->patch_id[v][w]);
          if(c[0]->patch_id[w][u] != -1 &&
             c[0]->patch_id[w][u] !=  c[0]->patch_id[u][v] &&
             c[0]->patch_id[w][u] !=  c[0]->patch_id[v][w] )
             vert_list[(int)vert_list.size()-1].cluster_membership.push_back(c[0]->patch_id[w][u]);
          if((int)vert_list[(int)vert_list.size()-1].cluster_membership.size() >= 2) cerr << " >= 2 ";
       }
       else
       {
          pure_chains[current_pure_chain].push_back(i2f.first->saddle_g_vid[i2f.second]);
       }

       for(int j = 0; j < (int)chains[i].size(); j ++)
       {
          // if the cell is already included by another chain
          if(chains[i][j]->g_vid != -1)
             pure_chains[current_pure_chain].push_back(chains[i][j]->g_vid);
          else // add its voronoi as a vertex in the graph
          {
             vert_list.push_back(GVertex(chains[i][j]->voronoi()));
             vert_list[(int)vert_list.size()-1].id = (int)vert_list.size()-1;
             vert_list[(int)vert_list.size()-1].c = chains[i][j];

             pure_chains[current_pure_chain].push_back((int)vert_list.size()-1);
             chains[i][j]->g_vid = (int)vert_list.size()-1;

             vert_list[(int)vert_list.size()-1].set_out(chains[i][j]->outside);

             // keep the info if this cell also lies on um(i1).
             if(chains[i][j]->VV_on_um_i1()) 
             {
                vert_list[(int)vert_list.size()-1].set_on_um_i1(true);
                for(int u = 0; u < 4; u ++)
                {
                   for(int v = u+1; v < 4; v ++)
                   {
                      if(chains[i][j]->patch_id[u][v] == -1) continue;
                      bool found = false;
                      for(int k = 0; k < (int)vert_list[(int)vert_list.size()-1].cluster_membership.size(); k ++)
                         if(vert_list[(int)vert_list.size()-1].cluster_membership[k] == 
                            chains[i][j]->patch_id[u][v])
                            found = true;
                      if(found) continue;
                      vert_list[(int)vert_list.size()-1].cluster_membership.push_back(
                                                  chains[i][j]->patch_id[u][v]);
                   }
                }
             }
          }
       }
    }
    set_nv((int)vert_list.size());

    for(int i = 0; i < (int)pure_chains.size(); i ++)
    {
       if((int)pure_chains[i].size() == 0) continue;
       for(int j = 0; j < (int)pure_chains[i].size() - 1; j ++)
       {
          edge_list.push_back(GEdge(pure_chains[i][j], pure_chains[i][j+1]));
          edge_list[(int)edge_list.size()-1].id = (int)edge_list.size()-1;

          // we have considered only the chains with status == SUCCESS.
          edge_list[(int)edge_list.size()-1].set_status(SUCCESS);

          // update adjacency information.
          // vertex
          vert_list[pure_chains[i][j]].add_inc_vert(pure_chains[i][j+1]);
          vert_list[pure_chains[i][j+1]].add_inc_vert(pure_chains[i][j]);
          // edge
          vert_list[pure_chains[i][j]].add_inc_edge((int)edge_list.size()-1);
          vert_list[pure_chains[i][j+1]].add_inc_edge((int)edge_list.size()-1);
       }
    }
    set_ne((int)edge_list.size());
}

}



