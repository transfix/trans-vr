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

#include <cstdlib>
#include <ctime>
#include <SecondaryStructures/op.h>
#include <SecondaryStructures/skel.h>

using namespace SecondaryStructures;

//Moved these here because of trouble with -frounding-math on gcc.
//We need it because building on SUSE linux gives an assertion error with CGAL without that option!
// - Joe Rivera - 4/20/2010
static const double DEFAULT_SHARP = 2 * M_PI / 3.0; // Angle of sharp edges.
static const double DEFAULT_RATIO = 1.2 * 1.2; // Squared thinness factor.
static const double DEFAULT_FLAT  = M_PI / 3.0; // Angle for flatness Test
static const double DEFAULT_MED_THETA = M_PI*22.5/180.0; // original: M_PI*22.5/180.0;

// arand, 5-27-2011: moved this initialization from header since mac was giving linking errors
// -- flatness marking ---
const double Skel::DEFAULT_ANGLE = M_PI / 8.0; // Half of the co-cone angle.

// -- robust cocone ---
const double Skel::DEFAULT_BIGBALL_RATIO  = (1./4.)*(1./4.); // parameter to choose big balls.
const double Skel::DEFAULT_THETA_IF_d  = 5.0; // parameter for infinite-finite deep intersection.
const double Skel::DEFAULT_THETA_FF_d  = 10.0; // parameter for finite-finite deep intersection.

// -- medial axis ---
//const double DEFAULT_MED_THETA = M_PI*22.5/180.0; // original: M_PI*22.5/180.0;
const double Skel::DEFAULT_MED_RATIO = 8.0*8.0; // original: 8.0*8.0;

const double Skel::BB_SCALE=0;

Skel::Skel()
{
	nv = 0;
	nf = 0;
	ne = 0;
	vert_list.clear();
	face_list.clear();
	edge_list.clear();
	comps.clear();
	A.clear();
	C.clear();
	star.clear();
	comp_pl.clear();
	comp_cnt = 0;
	bbox_diagonal = 0;
	/* interative options */
	helix_cnt = 0;
	beta_cnt = 0;
	_max_sheets = 0;
	bw = 3.;
	b_tol = 0.5;
	_alphaMinWidth = 2.5 - .5;
	_alphaMaxWidth = 2.5 + .5;
	_betaMinWidth = 2.5 - .5;
	_betaMaxWidth = 2.5 + .5;
}

void Skel::add_u1_to_skel(Triangulation& triang)
{
	for (FCI cit = triang.finite_cells_begin();
			cit != triang.finite_cells_end(); cit ++)
	{
		add_vertex(SVertex(cit->voronoi()));
		cit->set_skel_id((int)vert_list.size()-1);
	}
	// Skel is a planar graph with polygons and edges.
	// Add voronoi faces as polygons into skel.
	for (FEI eit = triang.finite_edges_begin();
			eit != triang.finite_edges_end(); eit ++)
	{
		Cell_handle c = (*eit).first;
		int uid = (*eit).second, vid = (*eit).third;
		if (! is_inside_VF(triang, (*eit)))
		{
			continue;
		}
		if (! c->VF_on_um_i1(uid, vid))
		{
			continue;
		}
		// Build a polygon with the Voronoi vertices.
		// Add the vertices and the polygon with
		// all the adjacency information.
		// Mark the VF visited in every Cell.
		Facet_circulator fcirc = triang.incident_facets((*eit));
		Facet_circulator begin = fcirc;
		vector<int> vlist;
		do
		{
			Cell_handle _c = (*fcirc).first;
			int _id = (*fcirc).second;
			int _uid = -1, _vid = -1;
			triang.has_vertex(_c, _id, c->vertex(uid), _uid);
			triang.has_vertex(_c, _id, c->vertex(vid), _vid);
			_c->e_visited[_uid][_vid] = true;
			_c->e_visited[_vid][_uid] = true;
			vlist.push_back(_c->skel_id());
			vert_list[_c->skel_id()].set_iso(false);
			vert_list[_c->skel_id()].on_u1 = true;
			fcirc ++;
		}
		while (fcirc != begin);
		// We have a new polygon to be inserted into skel.
		add_face(SFace(vlist));
		int fid = (int)face_list.size() - 1;
		face_list[fid].width = length_of_seg(Segment(c->vertex(uid)->point(), c->vertex(vid)->point()));
		for (int i = 0; i < (int)vlist.size(); i ++)
		{
			// add face to the vertex.
			vert_list[vlist[i]].add_inc_face(fid);
			int uid = vlist[i], vid = vlist[(i+1)%((int)vlist.size())];
			if (! vert_list[uid].is_inc_vert(vid))
			{
				CGAL_assertion(! vert_list[vid].is_inc_vert(uid));
				// inc_vert needs to be updated.
				vert_list[uid].add_inc_vert(vid);
				vert_list[vid].add_inc_vert(uid);
				// create an edge between uid and vid.
				add_edge(SEdge(uid,vid));
				// add face to the edge
				int eid = (int)edge_list.size()-1;
				edge_list[eid].add_inc_face(fid);
				edge_list[eid].on_u1 = true;
				// add edge to the vertices.
				vert_list[vid].add_inc_edge(edge_list.size() - 1);
				vert_list[uid].add_inc_edge(edge_list.size() - 1);
				// add edge to the face.
				face_list[fid].add_edge((int)edge_list.size() - 1);
			}
			else
			{
				// the edge is already there.
				// find the edge id using the vertex indices.
				SVertex sv = vert_list[vid];
				int eid = -1;
				CGAL_assertion(sv.get_eid(uid, eid));
				CGAL_assertion(eid != -1);
				// add the face to the edge.
				edge_list[eid].add_inc_face(fid);
				// add edge to the face.
				face_list[fid].add_edge(eid);
				// set non-manifold mark.
				if (edge_list[eid].num_inc_face > 2)
				{
					edge_list[eid].set_enm(true);
				}
			}
		}
	}
	set_nf((int)face_list.size());
	set_ne((int)edge_list.size());
	set_nv((int)vert_list.size());
	// walk on the skeleton to identify which polygons are vertex/edge-connected.
	// store the connected component id in every polygon.
	vector<bool> f_vis;
	f_vis.resize((int)face_list.size(), false);
	int comp_id = -1;
	for (int i = 0; i < (int)face_list.size(); i ++)
	{
		if (f_vis[i])
		{
			continue;
		}
		vector<int> walk;
		walk.push_back(i);
		f_vis[i] = true;
		comp_id++;
		face_list[i].comp_id = comp_id;
		vector<int> cur_comp;
		cur_comp.push_back(i);
		while (! walk.empty())
		{
			int fid = walk.back();
			walk.pop_back();
			CGAL_assertion(face_list[fid].comp_id == comp_id);
			CGAL_assertion(f_vis[fid]);
			// for every edge of this face, collect all the unvisited edge-adjacent faces.
			for (int j = 0; j < face_list[fid].edge_cnt; j ++)
			{
				int eid = face_list[fid].edge(j);
				// if non-manifold edge, continue.
				if (edge_list[eid].enm())
				{
					continue;
				}
				// collect all the incident faces and see which ones can be added.
				for (int k = 0; k < edge_list[eid].num_inc_face; k ++)
				{
					int _fid = edge_list[eid].inc_face(k);
					if (f_vis[_fid])
					{
						continue;
					}
					walk.push_back(_fid);
					f_vis[_fid] = true;
					// set the comp_id into the face.
					face_list[_fid].comp_id = comp_id;
					cur_comp.push_back(_fid);
				}
			}
		}
		comps.push_back(cur_comp);
		comp_cnt++;
	}
	comp_pl.resize(comp_cnt, false);
}

void Skel::add_u2_to_skel(const pair< vector< vector<Cell_handle> >, vector<Facet> >& u2)
{
	vector< vector<Cell_handle> > path = u2.first;
	vector<Facet> i2 = u2.second;
	CGAL_assertion((int)i2.size() == (int)path.size());
	int cnt = (int)i2.size();
	for (int i = 0; i < cnt; i ++)
	{
		// every chain starts with the circumcenter of an i2 facet.
		Facet i2f = i2[i];
		vector<Cell_handle> chain = path[i];
		int vid = i2f.first->skel_id(i2f.second);
		if (vid == -1)
		{
			// add this point.
			Point p = circumcenter(i2f);
			add_vertex(SVertex(p));
			vid = (int)vert_list.size()-1;
			vert_list[vid].set_iso(false);
			// store this id in the corresponding facet.
			Cell_handle c[2] = { i2f.first, i2f.first->neighbor(i2f.second) };
			int id[2] = { c[0]->index(c[1]), c[1]->index(c[0]) };
			c[0]->set_skel_id(id[0], vid);
			c[1]->set_skel_id(id[1], vid);
		}
		int uid = chain[0]->skel_id();
		CGAL_assertion(uid != -1);
		// add an edge between uid and vid.
		if (! in_same_comp(i2f.first->skel_id(),
						   i2f.first->neighbor(i2f.second)->skel_id()))
		{
			if (! vert_list[uid].is_inc_vert(vid))
			{
				CGAL_assertion(! vert_list[vid].is_inc_vert(uid));
				// inc_vert needs to be updated.
				vert_list[uid].add_inc_vert(vid);
				vert_list[vid].add_inc_vert(uid);
				// create an edge between uid and vid.
				add_edge(SEdge(uid,vid));
				int eid = (int)edge_list.size()-1;
				// set width.
				edge_list[eid].width = circumradius(i2f);
				// mark the vertices and edge on_u2.
				edge_list[eid].on_u2 = true;
				vert_list[uid].on_u2 = true;
				vert_list[vid].on_u2 = true;
				// add edge to the vertices.
				vert_list[vid].add_inc_edge(edge_list.size() - 1);
				vert_list[uid].add_inc_edge(edge_list.size() - 1);
			}
			else
			{
				// the edge is already there.
				// find the edge id using the vertex indices.
				SVertex sv = vert_list[vid];
				int eid = -1;
				CGAL_assertion(sv.get_eid(uid, eid));
				CGAL_assertion(eid != -1);
				edge_list[eid].on_u2 = true;
			}
		}
		// start the rest of the chain.
		for (int j = 0; j < (int)chain.size()-1; j ++)
		{
			uid = chain[j]->skel_id();
			CGAL_assertion(uid != -1);
			vid = chain[j+1]->skel_id();
			CGAL_assertion(vid != -1);
			// add an edge between uid and vid.
			if (! vert_list[uid].is_inc_vert(vid))
			{
				CGAL_assertion(! vert_list[vid].is_inc_vert(uid));
				// inc_vert needs to be updated.
				vert_list[uid].add_inc_vert(vid);
				vert_list[vid].add_inc_vert(uid);
				// create an edge between uid and vid.
				add_edge(SEdge(uid,vid));
				int eid = (int)edge_list.size()-1;
				// set width.
				double width = chain[j]->cell_radius() < chain[j+1]->cell_radius()?
							   chain[j]->cell_radius() : chain[j+1]->cell_radius();
				if (are_neighbors(chain[j], chain[j+1]))
				{
					width = circumradius(Facet(chain[j], chain[j]->index(chain[j+1])));
				}
				edge_list[eid].width = width;
				// mark the vertices and edges on_u2.
				edge_list[eid].on_u2 = true;
				vert_list[uid].on_u2 = true;
				vert_list[vid].on_u2 = true;
				// add edge to the vertices.
				vert_list[vid].add_inc_edge(edge_list.size() - 1);
				vert_list[uid].add_inc_edge(edge_list.size() - 1);
			}
			else
			{
				// the edge is already there.
				// find the edge id using the vertex indices.
				SVertex sv = vert_list[vid];
				int eid = -1;
				CGAL_assertion(sv.get_eid(uid, eid));
				CGAL_assertion(eid != -1);
				edge_list[eid].on_u2 = true;
			}
		}
	}
	set_ne((int)edge_list.size());
	set_nv((int)vert_list.size());
}

bool Skel::in_same_comp(const int& v1, const int& v2) const
{
	set<int> c1;
	for (int i = 0; i < vert_list[v1].num_inc_face; i++)
	{
		int fid = vert_list[v1].inc_face(i);
		if (face_list[fid].comp_id != -1)
		{
			c1.insert(face_list[fid].comp_id);
		}
	}
	set<int> c2;
	for (int i = 0; i < vert_list[v2].num_inc_face; i++)
	{
		int fid = vert_list[v2].inc_face(i);
		if (face_list[fid].comp_id != -1)
		{
			c2.insert(face_list[fid].comp_id);
		}
	}
	if (vert_list[v1].on_u1)
	{
		CGAL_assertion(! c1.empty());
	}
	if (vert_list[v2].on_u1)
	{
		CGAL_assertion(! c2.empty());
	}
	if (c1.empty() || c2.empty())
	{
		return false;
	}
	for (set<int>::iterator it1 = c1.begin();
			it1 != c1.end(); it1++)
		for (set<int>::iterator it2 = c2.begin();
				it2 != c2.end(); it2 ++)
			if ((*it1) == (*it2))
			{
				return true;
			}
	return false;
}

void Skel::do_star()
{
	// compute the area, center and star for each component.
	for (int i = 0; i < comp_cnt; i ++)
	{
		Vector comp_center = CGAL::NULL_VECTOR;
		double comp_area = 0;
		for (int j = 0; j < (int)comps[i].size(); j ++)
		{
			int fid = comps[i][j];
			Vector c = CGAL::NULL_VECTOR;
			for (int k = 0; k < face_list[fid].v_cnt; k ++)
			{
				int vid = face_list[fid].get_vertex(k);
				c = c + (vert_list[vid].point() - CGAL::ORIGIN);
			}
			c = (1./face_list[fid].v_cnt)*c;
			for (int k = 0; k < face_list[fid].v_cnt; k ++)
			{
				int vid = face_list[fid].get_vertex(k);
				int uid = face_list[fid].get_vertex((k+1)%face_list[fid].v_cnt);
				Triangle_3 t = Triangle_3((CGAL::ORIGIN + c), vert_list[vid].point(),
										  vert_list[uid].point());
				double a = sqrt(CGAL::to_double(t.squared_area()));
				comp_area += a;
				comp_center = comp_center + a*c;
			}
		}
		A.push_back(comp_area);
		comp_center = (1./comp_area)*comp_center;
		// project the center to the nearest face of the component.
		double min_d = HUGE;
		Vector p_cc = comp_center;
		for (int j = 0; j < (int)comps[i].size(); j ++)
		{
			int fid = comps[i][j];
			for (int k = 0; k < face_list[fid].v_cnt; k ++)
			{
				Vector _p = vert_list[face_list[fid].get_vertex(k)].point() - CGAL::ORIGIN;
				double d = CGAL::to_double((comp_center - _p)*(comp_center - _p));
				if (d < min_d)
				{
					p_cc = _p;
					min_d = d;
				}
			}
		}
		C.push_back(CGAL::ORIGIN + p_cc);
	}
	// identify the active vertices and store them in 'star' of
	vector<int> temp;
	star.resize(comp_cnt, temp);
	for (int i = 0; i < get_nv(); i ++)
	{
		set<int> c1;
		for (int j = 0; j < vert_list[i].num_inc_face; j++)
		{
			int fid = vert_list[i].inc_face(j);
			if (face_list[fid].comp_id != -1)
			{
				c1.insert(face_list[fid].comp_id);
			}
		}
		if (c1.empty())
		{
			continue;
		}
		if ((int)c1.size() > 1)
		{
			for (set<int>::iterator it = c1.begin();
					it != c1.end(); it ++)
			{
				star[(*it)].push_back(i);
			}
		}
		else
		{
			// if it has an edge incident on it, whose both endpoints
			// are not from the same component, make it active.
			for (int j = 0; j < vert_list[i].num_inc_edge; j++)
			{
				int eid = vert_list[i].inc_edge(j);
				if (! in_same_comp(edge_list[eid].get_endpoint(0),
								   edge_list[eid].get_endpoint(1)))
				{
					set<int>::iterator it = c1.begin();
					star[(*it)].push_back(i);
				}
			}
		}
	}
}

void Skel::filter(const int& pc)
{
	// remove the planar status of the components which are smaller than
	// pc biggest planar patches.
	// sort the components according to their areas.
	// vector<int> comp_id;
	comp_id.clear();
	vector<bool> b;
	b.resize(comp_cnt, false);
	int max_sheets = 0;
	for (int i = 0; i < comp_cnt; i ++)
	{
		double max_a = -HUGE;
		int max_id = -1;
		for (int j = 0; j < comp_cnt; j ++)
		{
			if (b[j])
			{
				continue;
			}
			if (A[j] > max_a)
			{
				max_a = A[j];
				max_id = j;
			}
		}
		b[max_id] = true;
		comp_id.push_back(max_id);
		if (max_a >= SHEET_AREA_THRESHOLD)
		{
			max_sheets++;
		}
	}
	// set the max helices (when culling by area)
	_max_sheets = max_sheets;
	// this is now done in the interactive section
	// for(int i = 0; i < comp_cnt; i ++)
	//    if( i < pc ) comp_pl[comp_id[i]] = true;
	// new color on each surface
	comp_colors.resize(comp_cnt);
	// first 6 are straight pallete
	comp_colors[0] = Color(0,0,1);
	comp_colors[1] = Color(0,1,0);
	comp_colors[2] = Color(0,1,1);
	comp_colors[3] = Color(1,0,0);
	comp_colors[4] = Color(1,0,1);
	comp_colors[5] = Color(1,1,0);
	// rest are random
	for (int i=6; i<comp_cnt; i++)
	{
	  float red = double(rand())/double(RAND_MAX); //drand48();
	  float green = double(rand())/double(RAND_MAX);//drand48();
	  float blue = double(rand())/double(RAND_MAX);//drand48();
		comp_colors[i] = Color(red,green,blue);
	}
	return;
}

void Skel::refine_skel(const double& eps)
{
	// first load the original edges from u2 which are in linear skeleton.
	vector<int> lid;
	lid.resize(get_nv(), -1);
	for (int i = 0; i < get_ne(); i ++)
	{
		if (edge_list[i].num_inc_face >= 1)
		{
			continue;
		}
		else
		{
			if (in_same_comp(edge_list[i].get_endpoint(0),
							 edge_list[i].get_endpoint(1)))
			{
				continue;
			}
		}
		int vid[2] = { edge_list[i].get_endpoint(0),
					   edge_list[i].get_endpoint(1)
					 };
		Point p[2] = { vert_list[vid[0]].point(),
					   vert_list[vid[1]].point()
					 };
		if (lid[vid[0]] == -1)
		{
			L.vlist.push_back(p[0]);
			lid[vid[0]] = (int)L.vlist.size()-1;
		}
		if (lid[vid[1]] == -1)
		{
			L.vlist.push_back(p[1]);
			lid[vid[1]] = (int)L.vlist.size()-1;
		}
		pair<int,int> e(lid[vid[0]], lid[vid[1]]);
		L.elist.push_back(e);
	}
	// now add the edges which are generated by starring.
	for (int i = 0; i < comp_cnt; i ++)
	{
		if (comp_pl[i])
		{
			continue;
		}
		// insert the center.
		Vector c = C[i] - CGAL::ORIGIN;
		L.vlist.push_back(CGAL::ORIGIN + c);
		int cid = (int)L.vlist.size()-1;
		for (int j = 0; j < (int)star[i].size(); j ++)
		{
			Vector p = vert_list[star[i][j]].point() - CGAL::ORIGIN;
			int uid = cid;
			// divide the segment into 10 parts and project any intermediate
			// point to the closest point on the component.
			for (int k = 1; k < REFINE_FACTOR-1; k ++)
			{
				Vector pi = (double)(REFINE_FACTOR - k)/REFINE_FACTOR*c +
							(double)(k)/REFINE_FACTOR*p;
				// if pi is further than epsilon away from the component, add it.
				double min_d = HUGE;
				Vector _pi = pi;
				for (int f = 0; f < (int)comps[i].size(); f ++)
				{
					int fid = comps[i][f];
					for (int v = 0; v < face_list[fid].v_cnt; v ++)
					{
						Vector temp = vert_list[face_list[fid].get_vertex(v)].point() - CGAL::ORIGIN;
						double d = CGAL::to_double((pi - temp)*(pi - temp));
						if (d < min_d)
						{
							_pi = temp;
							min_d = d;
						}
					}
				}
				if (min_d < eps)
				{
					continue;
				}
				// add _pi to L.vlist and make an edge between uid and the new point.
				// update uid.
				L.vlist.push_back(CGAL::ORIGIN + _pi);
				int vid = (int)L.vlist.size()-1;
				pair<int,int> e(uid,vid);
				L.elist.push_back(e);
				uid = vid;
			}
			// add an edge between uid and the endpoint p.
			if (lid[star[i][j]] == -1)
			{
				L.vlist.push_back(vert_list[star[i][j]].point());
				lid[star[i][j]] = (int)L.vlist.size()-1;
			}
			pair<int,int> e(uid,lid[star[i][j]]);
			L.elist.push_back(e);
		}
	}
	L.nv = (int)L.vlist.size();
	L.ne = (int)L.elist.size();
}

// convert to raw geometry.
cvcraw_geometry::cvcgeom_t* Skel::buildSheetGeometry()
{
	// seed
	srand((unsigned int)time(0));
	// output triangulated polygons
	{
		// no specular component
//		float specularColor[3] = {0,0,0};
//		memcpy(geom->m_SpecularColor, specularColor, sizeof(float)*3);
		// half diffuse
//		float diffuseColor[3] = {0.5,0.5,0.5};
//		memcpy(geom->m_DiffuseColor, diffuseColor, sizeof(float)*3);
		// .2 ambient
//		float ambientColor[3] = {0.2,0.2,0.2};
//		memcpy(geom->m_AmbientColor, ambientColor, sizeof(float)*3);
	}
	{
        cout<<"band width = " << _betaMinWidth <<" " << _betaMaxWidth << " nf " << get_nf() << endl;
		std::vector<float> vertices;
		std::vector<unsigned int> indices;
		std::vector<float> colors;
        
        
		for (int i=0; i<get_nf(); i++)
		{
			const SFace curface = face(i);
         
			// prune by width
			if (curface.width < _betaMinWidth ||
					curface.width > _betaMaxWidth)
			{
				continue;
			}
			// comp_pl is a bitmask over which faces we want to draw
			if (!comp_pl[curface.comp_id])
			{
				continue;
			}
			const int nverts = curface.v_cnt;
			Point midpt(0,0,0);
			// sum all vertices in the face
			for (int vi = 0; vi < nverts; vi++)
			{
				Point curpt = vertex(curface.get_vertex(vi)).point();
				midpt = midpt + Vector(curpt.x(), curpt.y(), curpt.z());
			}
			// find midpt
			midpt= Point(midpt.x()/nverts,
						 midpt.y()/nverts,
						 midpt.z()/nverts);
			//
			// create vectors
			//
			// first vertex is midpoint
			assert((vertices.size() % 3) == 0);
			const int midptIndex = vertices.size()/3;
			vertices.push_back(midpt.x());
			vertices.push_back(midpt.y());
			vertices.push_back(midpt.z());
			// push vertex color onto vector
			colors.push_back(comp_colors[curface.comp_id].r);
			colors.push_back(comp_colors[curface.comp_id].g);
			colors.push_back(comp_colors[curface.comp_id].b);
		//	colors.push_back(comp_colors[curface.comp_id].a);
			// keep track of the first index for future use
			assert((vertices.size() % 3) == 0);
			const int firstIndex = vertices.size()/3;
			for (int vi = 0; vi < nverts; vi++)
			{
				assert((vertices.size() % 3) == 0);
				const int currentIndex = vertices.size()/3;
				// add triangle
				if (vi == (nverts-1))
				{
					// end case is exception-- loop around to front
					indices.push_back(currentIndex);
					indices.push_back(firstIndex);
					indices.push_back(midptIndex);
				}
				else
				{
					indices.push_back(currentIndex);
					indices.push_back(currentIndex+1);
					indices.push_back(midptIndex);
				}
				// push current vertex onto vector
				Point addpt = vertex(curface.get_vertex(vi)).point();
				vertices.push_back(addpt.x());
				vertices.push_back(addpt.y());
				vertices.push_back(addpt.z());
				// push vertex color onto vector
				colors.push_back(comp_colors[curface.comp_id].r);
				colors.push_back(comp_colors[curface.comp_id].g);
				colors.push_back(comp_colors[curface.comp_id].b);
//				colors.push_back(comp_colors[curface.comp_id].a);
			}
		}
	
		vectors_to_tri_geometry(vertices,indices,colors, sheetGeom);
	
/*	FILE* fp =fopen("sheet.rawc", "w");
	fprintf(fp, "%d %d\n", geom->m_NumTriVerts, geom->m_NumTris);
	for(int i=0; i< geom->m_NumTriVerts; i++)
	{
		fprintf(fp, "%f %f %f %f %f %f\n", geom->m_TriVerts[3*i +0], geom->m_TriVerts[3*i +1], geom->m_TriVerts[3*i +2], geom->m_TriVertColors[3*i+0],
		 geom->m_TriVertColors[3*i+1], geom->m_TriVertColors[3*i+2]); 
	}

	for(int i=0; i<geom->m_NumTris; i++)
	{
		fprintf(fp, "%d %d %d\n", geom->m_Tris[3*i+0], geom->m_Tris[3*i+1], 
		geom->m_Tris[3*i+2]); 
	}
	fclose(fp);  */
 
	}
	return sheetGeom;
}

int Skel::compute_secondary_structures(cvcraw_geometry::cvcgeom_t* inputGeom)
{
	int argc = 3;
	char* argv[] = { (char*)"./secstruct", (char*)"inputfile.txt", (char*)"outputTexMol" };
	char ifname[90];
	char ofprefix[90];
	// robust cocone parameters.
	bool b_robust = false;
	double bb_ratio = DEFAULT_BIGBALL_RATIO;
	double theta_ff = M_PI/180.0*DEFAULT_THETA_FF_d;
	double theta_if = M_PI/180.0*DEFAULT_THETA_IF_d;
	// for flatness marking (in cocone)
	double flatness_ratio = DEFAULT_RATIO;
	double cocone_phi = DEFAULT_ANGLE;
	double flat_phi = DEFAULT_FLAT;
	//For medial axis
	double theta = DEFAULT_MED_THETA;
	double medial_ratio = DEFAULT_MED_RATIO;
	int biggest_medax_comp_id = -1;
	// Check commandline options.
	bool help = false;
	if (argc == 1)
	{
		help = true;
	}
	for (int i = 1; i < argc; i++)
	{
		if (argc < 3)
		{
			help = true;
			break;
		}
		if ((strcmp("-h", argv[i]) == 0) ||
				(strcmp("-help", argv[i]) == 0))
		{
			help = true;
			break;
		}
		else if (strcmp("-r", argv[i]) == 0)
		{
			b_robust = true;
		}
		else if (strcmp("-bbr", argv[i]) == 0)
		{
			++i;
			if (i >= argc)
			{
				cerr << "Error: option -bbr requires "
					 << "a second argument." << endl;
				help = true;
			}
			else
			{
				bb_ratio = atof(argv[i]) * atof(argv[i]);
			}
		}
		else if (strcmp("-thif", argv[i]) == 0)
		{
			++i;
			if (i >= argc)
			{
				cerr << "Error: option -thif requires "
					 << "a second argument." << endl;
				help = true;
			}
			else
			{
				theta_if = M_PI/180.*atof(argv[i]);
			}
		}
		else if (strcmp("-thff", argv[i]) == 0)
		{
			++i;
			if (i >= argc)
			{
				cerr << "Error: option -thff requires "
					 << "a second argument." << endl;
				help = true;
			}
			else
			{
				theta_ff = M_PI/180.*atof(argv[i]);
			}
		}
		else if (strcmp("-medr", argv[i]) == 0)
		{
			++i;
			if (i >= argc)
			{
				cerr << "Error: option -medr requires "
					 << "a second argument." << endl;
				help = true;
			}
			else
			{
				medial_ratio = atof(argv[i]) * atof(argv[i]);
			}
		}
		else if (strcmp("-medth", argv[i]) == 0)
		{
			++i;
			if (i >= argc)
			{
				cerr << "Error: option -medth requires "
					 << "a second argument." << endl;
				help = true;
			}
			else
			{
				theta = M_PI/180.*atof(argv[i]);
			}
		}
		else if (strcmp("-hc", argv[i]) == 0)
		{
			++i;
			if (i >= argc)
			{
				cerr << "Error: option -hc requires "
					 << "a second argument." << endl;
				help = true;
			}
			else
			{
				helix_cnt = atoi(argv[i]);
			}
		}
		else if (strcmp("-bc", argv[i]) == 0)
		{
			++i;
			if (i >= argc)
			{
				cerr << "Error: option -bc requires "
					 << "a second argument." << endl;
				help = true;
			}
			else
			{
				beta_cnt = atoi(argv[i]);
			}
		}
		else if (i+1 >= argc)
		{
			help = true;
		}
		else
		{
			strcpy(ifname, argv[i]);
			strcpy(ofprefix, argv[i+1]);
			i++;
		}
	}
	if (help)
	{
		cerr << "Usage: " << argv[0]
			 << " <infile> <outfile>" << endl;
		exit(1);
	}
	cerr << endl << "Shape name : " << ifname << endl << endl;
	CGAL::Timer timer;
	int cnt =0;
	Triangulation triang;
	timer.start();
	if (b_robust)
	{
		// Build the triangulation data structure.
		cerr << "DT 1 " << flush;
		for (int i = 0; i < inputGeom->points().size(); i++)
		{
			float x = inputGeom->points()[i][0];
			float y = inputGeom->points()[i][1];
			float z = inputGeom->points()[i][2];
			triang.insert(Point(x,y,z));
		}
		cerr << ". done." << endl;
		cerr << "Time : " << timer.time() <<" sec(s)."<< endl <<endl;
		timer.reset();
		// Initialization
		cerr << "Init 1 ";
		initialize(triang);
		cerr << ".";
		// Computation of voronoi vertex and cell radius.
		compute_voronoi_vertex_and_cell_radius(triang);
		cerr << ". done." << endl;
		cerr << "Time : " << timer.time() << endl << endl;
		timer.reset();
		cerr << "RC ";
		robust_cocone(bb_ratio, theta_ff, theta_if, triang, ofprefix);
		cerr << " done." << endl;
		cerr << "Time : " << timer.time() << endl << endl;
		timer.reset();
#ifdef DEBUG_OP
		write_iobdy(triang, ofprefix);
#endif
		// Create a new triangulation from the pointset taken
		// from "ofprefix.tcip".
		// Delete the earlier triangulation.
		triang.clear();
	}
	char robust_shape_filename[100];
	if (b_robust)
	{
		strcat(strcpy(robust_shape_filename, ofprefix),".tcip");
	}
	else
	{
		strcpy(robust_shape_filename, ifname);
	}
	cerr << "DT 2 ";
	// Maintain the min-max span of the pointset in 3 directions.
	double x_min = HUGE, x_max = -HUGE,
		   y_min = HUGE, y_max = -HUGE,
		   z_min = HUGE, z_max = -HUGE;
	int total_pt_cnt = 0;


		vector <Point> points;
	set <Point> points_used;

	for(int i = 0; i < inputGeom->points().size(); i++)
	{
		float x = inputGeom->points()[i][0];
		float y = inputGeom->points()[i][1];
		float z = inputGeom->points()[i][2];

		points.push_back(Point(x,y,z));
	}
	
	if(inputGeom->triangles().size() > 0)
	{
		for(int i=0; i< inputGeom->triangles().size(); i++)
		{
			int l1 = inputGeom->triangles()[i][0];
			int l2 = inputGeom->triangles()[i][1];
			int l3 = inputGeom->triangles()[i][2];
			points_used.insert(points[l1]);
			points_used.insert(points[l2]);
			points_used.insert(points[l3]);
		}
	}
	else 
	{
		for(int i=0; i<points.size(); i++)
		points_used.insert(points[i]);
	}


//	for (int i = 0; i < inputGeom->m_NumTriVerts*3;)
	for(set<Point>::iterator sit = points_used.begin(); sit!= points_used.end(); ++sit)
	{
		total_pt_cnt++;
//		float x = inputGeom->m_TriVerts[i++];
//		float y = inputGeom->m_TriVerts[i++];
//		float z = inputGeom->m_TriVerts[i++];
//		triang.insert(Point(x,y,z));
		triang.insert(*sit);

//		Vertex_handle new_vh = triang.insert(Point(x,y,z));
		Vertex_handle new_vh = triang.insert(*sit);
		// check x-span
		if (CGAL::to_double(new_vh->point().x()) < x_min)
		{
			x_min = CGAL::to_double(new_vh->point().x());
		}
		if (CGAL::to_double(new_vh->point().x()) > x_max)
		{
			x_max = CGAL::to_double(new_vh->point().x());
		}
		// check y-span
		if (CGAL::to_double(new_vh->point().y()) < y_min)
		{
			y_min = CGAL::to_double(new_vh->point().y());
		}
		if (CGAL::to_double(new_vh->point().y()) > y_max)
		{
			y_max = CGAL::to_double(new_vh->point().y());
		}
		// check z-span
		if (CGAL::to_double(new_vh->point().z()) < z_min)
		{
			z_min = CGAL::to_double(new_vh->point().z());
		}
		if (CGAL::to_double(new_vh->point().z()) > z_max)
		{
			z_max = CGAL::to_double(new_vh->point().z());
		}
	}
	cerr << " done." << endl;
	cerr << "Total point count: " << total_pt_cnt << endl;
	cerr << "Del Time : " << timer.time() << endl << endl;
	timer.reset();
	// Bounding box of the point set.
	bounding_box.push_back(x_min - BB_SCALE*(x_max-x_min));
	bounding_box.push_back(x_max + BB_SCALE*(x_max-x_min));
	bounding_box.push_back(y_min - BB_SCALE*(y_max-y_min));
	bounding_box.push_back(y_max + BB_SCALE*(y_max-y_min));
	bounding_box.push_back(z_min - BB_SCALE*(z_max-z_min));
	bounding_box.push_back(z_max + BB_SCALE*(z_max-z_min));
	bbox_diagonal = sqrt(CGAL::to_double((Vector(x_min,y_min,z_min) - Vector(x_max,y_max,z_max))*
										 (Vector(x_min,y_min,z_min) - Vector(x_max,y_max,z_max))));
	// init
	cerr << "Init 2 ";
	initialize(triang);
	cerr << ".";
	// compute voronoi vertex
	compute_voronoi_vertex_and_cell_radius(triang);
	cerr << ". done." << endl;
	cerr << "Time : " << timer.time() << endl << endl;
	timer.reset();
	// surface reconstruction using tightcocone.
	cerr << "TC ";
	tcocone(cocone_phi, DEFAULT_SHARP, flat_phi, flatness_ratio, triang);
	cerr << " done." << endl;
	cerr << "Time : " << timer.time() << endl << endl;
	timer.reset();
	// write the water-tight surface.
	write_wt(triang, ofprefix);
	// Medial Axis
	timer.reset();
	cerr << "Medial axis " << flush;
	compute_medial_axis(triang, theta, medial_ratio, biggest_medax_comp_id);
	cerr << " done." << endl;
	cerr << "TIME: "<<timer.time()<<" sec(s)." << endl << endl;
	timer.reset();
#ifdef DEBUG_OP
	// writing the medial axis in OFF format.
	write_axis(triang, biggest_medax_comp_id, ofprefix);
#endif
	// rewrite this part.
	cerr << "U1";
	compute_u1(triang, ofprefix);
	cerr << " done." << endl;
	cerr << "TIME: "<<timer.time()<<" sec(s)." << endl;
	timer.reset();
	cerr << "Building planar part ";
	add_u1_to_skel(triang);
	cerr << " done." << endl;
	cerr << "TIME: "<<timer.time()<<" sec(s)." << endl << endl;
	timer.reset();
#ifdef DEBUG_OP
	// writing the U1's.
	write_u1_skel(ofprefix);
#endif
	cerr << "U2";
	pair< vector< vector<Cell_handle> >, vector<Facet> > u2 = compute_u2(triang, ofprefix);
	cerr << " done." << endl;
	cerr << "TIME: "<<timer.time()<<" sec(s)." << endl;
	timer.reset();
	cerr << "Building linear part ";
	add_u2_to_skel(u2);
	cerr << " done." << endl;
	cerr << "TIME: "<<timer.time()<<" sec(s)." << endl << endl;
	timer.reset();
#ifdef DEBUG_OP
	// writing the U2's.
	write_u2_skel(ofprefix);
#endif
	cerr << "Filtering ";
	// star the patches.
	do_star();
	// filter small patches.
	filter(beta_cnt);
	cerr << "done.";
	// refine the linear part of the skeleton.
	// store the final skeleton as a polylinear graph in skel.
	double eps = 0.1*bbox_diagonal;
	refine_skel(eps);
	cerr << "Refinement " << timer.time() << endl;
	timer.reset();
	// write the full skel.
	write_skel(ofprefix);
	cerr << "Write skel " << timer.time() << endl;
	timer.reset();
	// compute sheets.
	// compute_sheets(skel, ofprefix);
	// currently, collect the top 'beta_cnt' patches
	// and color them differently.
	//
	// all remaining "interactive" computations are processed in Skel::update_display
	//
	return 0;
}

//void Skel::update_display(Geometry* helixGeom, Geometry* sheetGeom, Geometry* curveGeom,  int alphaCount, int betaCount,  float alphaMinWidth, float alphaMaxWidth,float betaMinWidth, float betaMaxWidth,bool alphaHistogramChanged)
void Skel::buildAllGeometry(int alphaCount, int betaCount,  float alphaMinWidth, float alphaMaxWidth,float betaMinWidth, float betaMaxWidth,bool alphaHistogramChanged, bool betaHistogramChanged)

{
    helixGeom = new cvcraw_geometry::cvcgeom_t();
	sheetGeom = new cvcraw_geometry::cvcgeom_t();
	skelGeom = new cvcraw_geometry::cvcgeom_t();
	curveGeom = new cvcraw_geometry::cvcgeom_t();

    _alphaMinWidth = alphaMinWidth;
	_alphaMaxWidth = alphaMaxWidth;
	_betaMinWidth = betaMinWidth;
	_betaMaxWidth = betaMaxWidth;
    
 	helix_cnt = alphaCount;
	beta_cnt = betaCount;
	// rebuild geometry
	const char* ofprefix = "outputTexMol";
	CGAL::Timer timer;
	timer.start();
    
    // recompute helices
    if (alphaHistogramChanged)
    {
        compute_helices(ofprefix);
    }

    
	// bound minimums
	if (helix_cnt < 0)
	{
		helix_cnt = 0;
	}
	if (beta_cnt < 0)
	{
		beta_cnt = 0;
	}
	// bound maximums
    
 
	if (helix_cnt > (helices.size()))
	{
		helix_cnt = (helices.size());
	}
	if (beta_cnt > _max_sheets)
	{
		beta_cnt = _max_sheets;
	}
	// if(beta_cnt > (comp_id.size()))
	//         beta_cnt = (comp_id.size());
	// fix bitmask to only cover the top beta_cnt elements
	for (int i = 0; i < comp_cnt; i ++)
	{
		comp_pl[comp_id[i]] = (i < beta_cnt);
	}


  // cerr << "Build geometry " << timer.time() << endl;
	timer.reset();
	// compute the helices and write helix geometry
	// recompute helices
        
        
   if (alphaHistogramChanged)
    { 
        helixGeom = buildHelixGeometry();
    }
    if (betaHistogramChanged)
    {
        sheetGeom = buildSheetGeometry();
    }
    
		

        curveGeom = buildCurveGeometry(_curve_comps, _curve);  //We don't build curve geometry first.
	


	// cerr << "Build helices " << timer.time() << endl;
	timer.reset();
	// build the point curve geometry
	  skelGeom = buildSkeletonGeometry();

}


cvcraw_geometry::cvcgeom_t* Skel::buildSkeletonGeometry()
{
	std::vector<float> vertices;
	std::vector<unsigned int> indices;
	int curIndex = 0;
	for (int i=0; i<get_ne(); i++)
	{
		// ensure that edge is attached to a visible face
		// comp_pl is a bitmask over which faces we want to draw
		// bool visible = false;
		// for(int faceItr = 0; faceItr < edge(i).get_n_inc_faces(); faceItr++) {
		//         int face = edge(i).inc_face(faceItr);
		//         if(comp_pl[face])
		//                 visible = true;
		// }
		// if(visible) {
		Point p0 = vertex(edge(i).get_endpoint(0)).point();
		Point p1 = vertex(edge(i).get_endpoint(1)).point();
		indices.push_back(curIndex++);
		vertices.push_back(p0.x());
		vertices.push_back(p0.y());
		vertices.push_back(p0.z());
		indices.push_back(curIndex++);
		vertices.push_back(p1.x());
		vertices.push_back(p1.y());
		vertices.push_back(p1.z());
		// }
	}
	vectors_to_line_geometry(vertices,indices,1.0,0,0,1.0,skelGeom);
	return skelGeom;



}
