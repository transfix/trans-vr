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

#include <SecondaryStructures/u1.h>

using namespace SecondaryStructures;

// Identify the index-1 saddles.
// These are the intersection points of VF with their dual DEs. Compute their unstable manifolds.
void compute_u1(Triangulation& triang, char* op_file_prefix)
{
	// reset the f_visited field to false in every cell.
	for (FFI fit = triang.finite_facets_begin();
			fit != triang.finite_facets_end(); fit ++)
	{
		(*fit).first->f_visited[(*fit).second] = false;
		(*fit).first->neighbor((*fit).second)->f_visited[
			(*fit).first->neighbor((*fit).second)->index((*fit).first)] = false;
	}
	cerr << "\ti1 saddles ";
	vector<Facet> dual_VE_list;
	for (FEI eit = triang.finite_edges_begin();
			eit != triang.finite_edges_end(); eit ++)
	{
		Cell_handle c = (*eit).first;
		int uid = (*eit).second, vid = (*eit).third;
		// if VF is crossing the surface, continue.
		if (is_surf_VF(triang, c, uid, vid))
		{
			continue;
		}
		// if infinite VF continue.
		if (is_inf_VF(triang, c, uid, vid))
		{
			continue;
		}
		// if not on medial axis, conitnue.
		if (! c->VF_on_medax(uid, vid))
		{
			continue;
		}
		// if it's not i1saddle continue.
		if (! is_i1_saddle((*eit), triang))
		{
			continue;
		}
		// collect its VEs into dual_VE_list.
		// markings: 1. VF contains an i1 saddle.
		//           2. VF is on um(i1).
		Facet_circulator fcirc = triang.incident_facets((*eit));
		Facet_circulator begin = fcirc;

		do
		{
			Cell_handle cur_c = (*fcirc).first;
			int cur_fid = (*fcirc).second;
			int cur_uid = -1, cur_vid = -1, cur_wid = -1;
			// assign cur_uid, cur_vid, cur_wid to access the edge (c,uid,vid)
			// via cur_c. convention e = (c,uid,vid) = (cur_c, cur_uid, cur_vid).
			for (int i = 1; i < 4; i ++)
				if (cur_c->vertex((cur_fid+i)%4)->id != c->vertex(uid)->id &&
						cur_c->vertex((cur_fid+i)%4)->id != c->vertex(vid)->id)
				{
					cur_wid = (cur_fid+i)%4;
					vertex_indices(cur_fid, cur_wid, cur_uid, cur_vid);
					break;
				}

			// to make sure that the two cells are accessing the same edge.
			CGAL_assertion((c->vertex(uid)->id == cur_c->vertex(cur_uid)->id &&
							c->vertex(vid)->id == cur_c->vertex(cur_vid)->id) ||
						   (c->vertex(uid)->id == cur_c->vertex(cur_vid)->id &&
							c->vertex(vid)->id == cur_c->vertex(cur_uid)->id));


			// markings are done first.
			cur_c->set_i1_saddle(cur_uid, cur_vid, true);
			cur_c->set_VF_on_um_i1(cur_uid, cur_vid, true);
			CGAL_assertion(cur_c->VV_on_um_i1());


			// if the current VE (= dual(fcirc)) is already visited continue.
			if (cur_c->f_visited[cur_fid])
			{
				fcirc++;
				continue;
			}
			// if it's not transversal, continue.
			if (! is_transversal_flow((*fcirc)))
			{
				fcirc++;
				continue;
			}
			// otherwise collect the VE
			dual_VE_list.push_back(*fcirc);
			// mark visited.
			cur_c->f_visited[cur_fid] = true;
			cur_c->neighbor(cur_fid)->f_visited[cur_c->neighbor(cur_fid)->index(cur_c)] = true;
			fcirc ++;
		}
		while (fcirc != begin);
	}
	cerr << " collected. ";
	// at this stage we have a list of VFs (which contain i1 saddles)
	// which is what currently um(i1) is. It has a set of edges among
	// which through the transversal ones the um(i1) will grow.
	int progress = 0;
	while (! dual_VE_list.empty())
	{
		if (++progress%1000 == 0)
		{
			progress = 0;
			cerr << ".";
		}
		Facet f = dual_VE_list.back();
		dual_VE_list.pop_back();
		CGAL_assertion(f.first->f_visited[f.second]);
		CGAL_assertion(is_transversal_flow(f));
		Cell_handle c = f.first;
		int fid = f.second;
		// convention : uid,vid corresponds to the ACCEPTOR DE.
		//              uid,wid and vid,wid correspond to the DONOR DE.
		//           => <u,w,v > 90 degree.
		int uid = -1, vid = -1, wid = -1;
		CGAL_assertion(find_acceptor(c, fid, uid, vid, wid));
		CGAL_assertion(uid + vid + wid == 6 - fid);
		// dual(f = Facet(c,fid)) is transversal and dual(e = Edge(c,uid,vid)) is
		// acceptor of dual(f).
		// growth of um(i1) -> include this acceptor VF into the um(i1).
		// before we need to pass the VF through the following tests.
		// 1. if it is an infinite VF or not.
		// 2. if it is crossing the surface or not.
		// in both cases the VF should be rejected.
		if (is_surf_VF(triang, c, uid, vid) ||
				is_inf_VF(triang, c, uid, vid))
		{
			continue;
		}
		// mark the VF to be on um(i1).
		Facet_circulator fcirc = triang.incident_facets(Edge(c,uid,vid));
		Facet_circulator begin = fcirc;
		do
		{
			Cell_handle _c = (*fcirc).first;
			int _id = (*fcirc).second;
			int _uid = -1, _vid = -1;
			triang.has_vertex(_c, _id, c->vertex(uid), _uid);
			triang.has_vertex(_c, _id, c->vertex(vid), _vid);
			_c->set_VF_on_um_i1(_uid,_vid,true);
			fcirc ++;
		}
		while (fcirc != begin);
		// push its unvisited transversal VEs into dual_VE_list.
		do
		{
			Cell_handle c0 = (*fcirc).first;
			int fid0 = (*fcirc).second;
			if (c0->f_visited[fid0])
			{
				fcirc++;
				continue;
			}
			Cell_handle c1 = c0->neighbor(fid0);
			int fid1 = c1->index(c0);
			// collect the VE if it's transversal.
			if (! is_transversal_flow(Facet(c0, fid0)))
			{
				fcirc++;
				continue;
			}
			dual_VE_list.push_back((*fcirc));
			// mark this VE (dual DF) visited.
			c0->f_visited[fid0] = true;
			c1->f_visited[fid1] = true;
			fcirc ++;
		}
		while (fcirc != begin);
	}
}
