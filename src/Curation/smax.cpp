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

#include <Curation/smax.h>

#define EPSILON 3

namespace Curation{

void add_cell_to_cluster(map<int, cell_cluster> &cluster_set, int rep1, int rep2)
{
   cell_cluster *c_rep1 = &cluster_set[rep1]; // pointer to the head of 1st cluster.
   cell_cluster *c = &cluster_set[c_rep1->tail]; // pointer to the end of 1st cluster.
   CGAL_assertion(c != 0);
   CGAL_assertion(c->nxt == NULL);

   cell_cluster *c_rep2 = &cluster_set[rep2]; // head of 2nd cluster.
   c_rep1->tail = c_rep2->tail;
   c->nxt = c_rep2;
   while(c->nxt != 0) 
   { 
      c->nxt->rep = rep1;
      c = c->nxt; 
   }
}

void merge_cluster(map<int, cell_cluster> &cluster_set, int rep1, int rep2)
{
   cell_cluster *c_rep1 = &cluster_set[rep1]; // head of cluster 1
   cell_cluster *c = &cluster_set[c_rep1->tail]; // tail of cluster 1
   CGAL_assertion(c != 0);
   CGAL_assertion(c->nxt == NULL);
   cell_cluster *c_rep2 = &cluster_set[rep2];
   c_rep1->tail = c_rep2->tail;

   c->nxt = c_rep2;
   while(c->nxt != 0) 
   { 
      c->nxt->rep = rep1;
      c = c->nxt; 
   }
}


void
grow_maximum(Cell_handle c_max, 
             Triangulation& triang, 
             map<int, cell_cluster> &cluster_set)
{
   // mark it visited.
   c_max->visited = true;
   cluster_set[c_max->id].in_cluster = true;
   cluster_set[c_max->id].outside = c_max->outside;

   // Now grow the maximum through the other tetrahedra.
   vector<Facet> bdy_stack;
   for(int i = 0; i < 4; i ++)
      bdy_stack.push_back(Facet(c_max, i));
   while(! bdy_stack.empty())
   {
      Cell_handle old_c = bdy_stack.back().first;
      int old_id = bdy_stack.back().second;
      bdy_stack.pop_back();
      CGAL_assertion(old_c->visited);
      CGAL_assertion(cluster_set[old_c->id].in_cluster);

      #ifndef __INSIDE__
      CGAL_assertion( old_c->outside );
      #endif

      #ifndef __OUTSIDE__
      CGAL_assertion( ! old_c->outside );
      #endif

      Cell_handle new_c = old_c->neighbor(old_id);
      int new_id = new_c->index(old_c);

      // If the new_c is infinite then no point in checking
      // the flow.
      if(triang.is_infinite(new_c))
         continue;
      // if the flow hits the surface continue.
      // if( old_c->outside != new_c->outside)
         // continue;
      // If new_c is already visited continue.
      if(new_c->visited) 
         continue;

      // collect new_c only if new_c flows into old_c via the
      // facet in between.
      if( is_outflow(Facet(new_c, new_id)) )
      {
         // CGAL_assertion( !is_outflow(Facet(old_c, old_id)));
         if(is_outflow(Facet(old_c, old_id))) continue;
         
         // new_c has to satisfy the following.
         CGAL_assertion( !is_maxima(new_c) && !new_c->visited &&
                         !triang.is_infinite(new_c));
         
         new_c->visited = true;
         cluster_set[new_c->id].in_cluster = true;
         cluster_set[new_c->id].outside = c_max->outside;

         // put the cells accessible via new_c into bdy_stack.
         for(int i = 1; i < 4; i ++)
         {
            if(new_c->neighbor((new_id+i)%4)->visited) 
	       continue;
	    bdy_stack.push_back(Facet(new_c, (new_id+i)%4));
         }
         // put new_c into the current cluster.
         // In other words merge the current cluster and the cluster owned by
         // new_c (as it is initialized).
         add_cell_to_cluster(cluster_set, c_max->id, new_c->id);
      }
   }
}

void
club_segment(Triangulation &triang,
	     map<int, cell_cluster> &cluster_set,
	     double mr)
{
   for(FFI fit = triang.finite_facets_begin();
      fit != triang.finite_facets_end(); fit ++)
   {
      Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
      int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};
      // each finite facet has atleast one finite cell incident.
      if(triang.is_infinite(c[0]) || triang.is_infinite(c[1]))
         continue;
      // if any one of them is not in any cluster, continue.
      if( ! cluster_set[c[0]->id].in_cluster || 
          ! cluster_set[c[1]->id].in_cluster )
         continue;
      // if any of the clusters is inside, continue.
      if( ! cluster_set[c[0]->id].outside ||
          ! cluster_set[c[1]->id].outside )
         continue;
      // if the two adjacent cells belong to the same cluster, continue.
      if(cluster_set[c[0]->id].find() == 
	 cluster_set[c[1]->id].find()) continue;

      // find the circumradius of the facet.
      double r = circumradius(Facet(c[0],id[0]));
      double sq_fr = r*r;
      // collect the squared_radius of the two maxima defining two clusters.
      double sq_c1r = cluster_set[cluster_set[c[0]->id].find()].sq_r;
      double sq_c2r = cluster_set[cluster_set[c[1]->id].find()].sq_r;
      if( sq_c1r/sq_fr < mr && sq_c2r/sq_fr < mr )
      {
         // we can merge the two clusters.
         // in other words, the cluster with bigger radius owns the cluster
         // with smaller radius.
         // the merge routine assumes, rep1-th cluster owns the union.
         // so set rep1 as the maximum of the bigger radius.
         int rep1 = cluster_set[c[0]->id].find(); // gives the id of the sink c is flowing into.
	 int rep2 = cluster_set[c[1]->id].find(); // - do -
	 CGAL_assertion(rep1 != rep2);
         if( sq_c1r < sq_c2r )
         {
            rep1 = cluster_set[c[1]->id].find();
            rep2 = cluster_set[c[0]->id].find();
         }
         CGAL_assertion( cluster_set[rep1].sq_r >= cluster_set[rep2].sq_r );
         merge_cluster(cluster_set, rep1, rep2);
      }
   }
}


void
club_contiguous_segment(Triangulation &triang,
	                map<int, cell_cluster> &cluster_set )
{
   for(FFI fit = triang.finite_facets_begin();
      fit != triang.finite_facets_end(); fit ++)
   {
      Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
      // if the two adjacent cells belong to the same cluster, continue.
      if(cluster_set[c[0]->id].find() == 
	 cluster_set[c[1]->id].find()) 
         continue;
      // if any one of them is not in any cluster, continue.
      if( ! cluster_set[c[0]->id].in_cluster || 
          ! cluster_set[c[1]->id].in_cluster )
         continue;

      // if any of the clusters is inside, continue.
      if( ! cluster_set[c[0]->id].outside ||
          ! cluster_set[c[1]->id].outside )
         continue;

      // merge the two clusters.
      merge_cluster(cluster_set, cluster_set[c[0]->id].find(), cluster_set[c[1]->id].find());
   }
}

void
calc_cluster_volume_and_store_with_cluster_rep(
		Triangulation &triang,
	        map<int, cell_cluster> &cluster_set,
		vector<double> &cluster_volume_vector,
		vector<int> &cluster_rep_vector )
{
    for(FCI cit = triang.finite_cells_begin();
       cit != triang.finite_cells_end(); cit ++)
    {
       if( ! cluster_set[cit->id].in_cluster )
          continue;
       if( ! cluster_set[cit->id].outside )
          continue;
       double volume = cell_volume(cit);
       // see if we already computed some of the cluster volume.
       // in other words see if the 'rep' is there in the cluster_rep_vector.
       int rep = cluster_set[cit->id].find();
       bool found = false;
       int pos = -1;
       for(int i = 0; i < (int)cluster_rep_vector.size(); i ++)
          if(cluster_rep_vector[i] == rep) 
	  {
	     found = true;
	     pos = i;
	     break;
	  }
       if(found)
          cluster_volume_vector[pos] += volume;
       else
       {
          cluster_volume_vector.push_back(volume);
	  cluster_rep_vector.push_back(rep);
       }
    }
    CGAL_assertion(cluster_volume_vector.size() == cluster_rep_vector.size());
}

void
sort_cluster_wrt_volume(const vector<double>& cluster_volume_vector,
		        const vector<int>& cluster_rep_vector,
		        vector<int>& sorted_cluster_index_vector)
{
    vector<bool> f;
    f.clear(); f.resize(cluster_volume_vector.size(), false);
    for(int i = 0; i < (int)cluster_volume_vector.size(); i ++)
    {
	    int ind = -1; double max = -HUGE;
	    for(int j = 0; j < (int)cluster_volume_vector.size(); j ++)
	    {
		    if(f[j]) continue;
		    if(cluster_volume_vector[j] > max)
		    {
			    max = cluster_volume_vector[j];
			    ind = j;
		    }
	    }
	    CGAL_assertion(ind != -1); CGAL_assertion(max != -HUGE);
	    f[ind] = true;
	    sorted_cluster_index_vector.push_back(cluster_rep_vector[ind]);
    }
}

vector<int>
compute_smax(Triangulation& triang, 
             map<int, cell_cluster> &cluster_set,
             const double& mr)
{
   for(ACI cit = triang.all_cells_begin();
      cit != triang.all_cells_end(); cit ++)
   {
      cluster_set[cit->id] = cell_cluster(cit->id, cit->cell_radius());
      cit->visited = false;
   }

   int max_cnt = 0;
   for(FCI cit = triang.finite_cells_begin();
      cit != triang.finite_cells_end(); cit ++)
   {
      if( ! is_maxima(cit) ) continue;

      #ifndef __OUTSIDE__
      if( cit->outside ) continue;
      #endif

      #ifndef __INSIDE__
      if( ! cit->outside ) continue;
      #endif
      
      if(max_cnt++%1000 == 0) cerr << "+";
      grow_maximum(cit, triang, cluster_set);
   }
   cerr << ".";

   club_segment(triang, cluster_set, mr );
   // club_contiguous_segment(triang, cluster_set );
   cerr << ".";

   // Compute the volume of each cluster. Remember after merging the 
   // 'rep' field is more useful than cluster_id.
   vector<int> cluster_ids;
   vector<double> cluster_vol;
   cluster_ids.clear();
   cluster_vol.clear();
   calc_cluster_volume_and_store_with_cluster_rep(triang, 
	  					  cluster_set,
			                          cluster_vol, 
						  cluster_ids);
   cerr << ".";

   // Sort the clusters with respect to the volumes.
   vector<int> sorted_indices;
   sorted_indices.clear();
   sort_cluster_wrt_volume(cluster_vol, cluster_ids, sorted_indices);
   cerr << ".";

   /*
   cout << "Statistics ";
   for(int i = 0; i < (int)sorted_indices.size(); i ++)
   {
      cout << sorted_indices[i] << " ";
      int cl_id = sorted_indices[i];
      bool found = false;
      int pos = -1;
      for(int i = 0; i < (int)cluster_ids.size(); i ++)
      {
          if(cluster_ids[i] == cl_id) 
	  {
	     found = true;
	     pos = i;
	     break;
	  }
      }
      if(found)
         cout << cluster_vol[pos] << endl;
      else
         cerr << "error." << endl;
   }
   */
   return sorted_indices;
}

}
