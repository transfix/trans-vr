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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <Curation/curat.h>

namespace Curation {
extern int xdim, ydim, zdim;
extern double xmin, ymin, zmin, xmax, ymax, zmax, xspan, yspan, zspan;

void curate_tr(Triangulation &triang, map<int, cell_cluster> &cluster_set,
               const vector<int> &sorted_cluster_index_vector,
               const int output_pocket_count, const int output_tunnel_count) {
  int tun_void_cnt = 0;
  for (int i = 0; i < (int)sorted_cluster_index_vector.size(); i++) {
    if (i >= (int)sorted_cluster_index_vector.size())
      break;
    int cl_id = sorted_cluster_index_vector[i];
    if (cluster_set[cl_id].mouth_cnt < 1)
      tun_void_cnt++;
    else if (cluster_set[cl_id].mouth_cnt > 1)
      tun_void_cnt++;
    else
      continue;
    if (tun_void_cnt <= output_tunnel_count) {
      cerr << "Tunnel/Void number " << tun_void_cnt
           << " is not to be curated." << endl;
      continue;
    }
    // currently we do not curate more than 100 tunnel/void.
    if (tun_void_cnt > 100)
      break;
    // curate this tunnel/void.
    for (FCI cit = triang.finite_cells_begin();
         cit != triang.finite_cells_end(); cit++) {
      if (cluster_set[cit->id].find() != cl_id)
        continue;
      cit->outside = false;
      // tag this cell.
      cit->c_tag = true;
    }
  }

  int pocket_cnt = 0;
  for (int i = 0; i < (int)sorted_cluster_index_vector.size(); i++) {
    if (i >= (int)sorted_cluster_index_vector.size())
      break;
    int cl_id = sorted_cluster_index_vector[i];
    if (cluster_set[cl_id].mouth_cnt < 1)
      continue;
    else if (cluster_set[cl_id].mouth_cnt > 1)
      continue;
    else
      pocket_cnt++;

    if (pocket_cnt <= output_pocket_count) {
      cerr << "Pocket number " << pocket_cnt << " is not to be curated."
           << endl;
      continue;
    }
    // currently we do not curate more than 100 tunnel/void.
    if (pocket_cnt > 100)
      break;

    // curate this pocket.
    for (FCI cit = triang.finite_cells_begin();
         cit != triang.finite_cells_end(); cit++) {
      if (cluster_set[cit->id].find() != cl_id)
        continue;
      cit->outside = false;
      // tag this cell.
      cit->c_tag = true;
    }
  }
  return;
}

map<pair<int, int>, int> edge_map;

bool find_id(const int &uid, const int &vid, int &eid) {
  map<pair<int, int>, int>::iterator it =
      edge_map.find(pair<int, int>(uid, vid));
  if (it != edge_map.end()) {
    eid = it->second;
    return true;
  } else
    return false;
}

void set_id(const int &uid, const int &vid, const int &new_eid) {
  edge_map[pair<int, int>(uid, vid)] = new_eid;
  return;
}

Point get_intersection_point(const vector<double> &in_val,
                             const vector<double> &out_val,
                             const vector<Point> &inP,
                             const vector<Point> &outP, const int &i,
                             const int &j) {
  double v[2] = {in_val[i], out_val[j]};
  double isovalue = 0;
  Point res_p = CGAL::ORIGIN +
                (isovalue - v[1]) / (v[0] - v[1]) * (inP[i] - CGAL::ORIGIN) +
                (isovalue - v[0]) / (v[1] - v[0]) * (outP[j] - CGAL::ORIGIN);
  return res_p;
}

void build_bdy_tr_mesh(const vector<Point> &outP, const vector<Point> &inP,
                       const vector<int> &out_id, const vector<int> &in_id,
                       const vector<double> &out_val,
                       const vector<double> &in_val, TrMesh &trmesh) {
  static int newp_cnt = 0;
  if ((int)inP.size() == 1) {
    vector<int> vid;
    vid.resize(3, -1);
    // the three vertices are the intersection points between inP[0] and
    // outP[0,1,2].
    for (int i = 0; i < 3; i++) {
      int id = -1;
      if (find_id(in_id[0], out_id[i], id))
        vid[i] = id;
      else {
        Point p = get_intersection_point(in_val, out_val, inP, outP, 0, i);
        // add this point to the mesh.
        TrVertex *new_v = new TrVertex(p);
        trmesh.v_list.push_back(new_v);
        // record the id of this new point associated with in_id[0] and
        // out_id[i].
        id = newp_cnt;
        set_id(in_id[0], out_id[i], id);
        vid[i] = id;
        newp_cnt++;
      }
    }
    // add a facet between the three vertices vid[0,1,2]
    TrFace *new_f = new TrFace(vid[0], vid[1], vid[2]);
    trmesh.f_list.push_back(new_f);
  } else if ((int)inP.size() == 2) {
    // re-order the inP, outP etc accoding to their ids.
    vector<Point> o_in_P, o_out_P;
    vector<double> o_in_val, o_out_val;
    vector<int> o_in_id, o_out_id;

    if (in_id[0] > in_id[1]) {
      o_in_P.push_back(inP[1]);
      o_in_P.push_back(inP[0]);
      o_in_val.push_back(in_val[1]);
      o_in_val.push_back(in_val[0]);
      o_in_id.push_back(in_id[1]);
      o_in_id.push_back(in_id[0]);
    } else if (in_id[0] < in_id[1]) {
      o_in_P.push_back(inP[0]);
      o_in_P.push_back(inP[1]);
      o_in_val.push_back(in_val[0]);
      o_in_val.push_back(in_val[1]);
      o_in_id.push_back(in_id[0]);
      o_in_id.push_back(in_id[1]);
    } else
      assert(0);

    if (out_id[0] > out_id[1]) {
      o_out_P.push_back(outP[1]);
      o_out_P.push_back(outP[0]);
      o_out_val.push_back(out_val[1]);
      o_out_val.push_back(out_val[0]);
      o_out_id.push_back(out_id[1]);
      o_out_id.push_back(out_id[0]);
    } else if (out_id[0] < out_id[1]) {
      o_out_P.push_back(outP[0]);
      o_out_P.push_back(outP[1]);
      o_out_val.push_back(out_val[0]);
      o_out_val.push_back(out_val[1]);
      o_out_id.push_back(out_id[0]);
      o_out_id.push_back(out_id[1]);
    } else
      assert(0);

    vector<int> new_vid;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
        // intersection point between o_in_p[i] and o_out_p[j].
        int id = -1;
        if (find_id(o_in_id[i], o_out_id[j], id))
          new_vid.push_back(id);
        else {
          Point p = get_intersection_point(o_in_val, o_out_val, o_in_P,
                                           o_out_P, i, j);
          // add the new point to the mesh.
          TrVertex *new_v = new TrVertex(p);
          trmesh.v_list.push_back(new_v);
          // record the id of this new point associated with in_id[0] and
          // out_id[i].
          id = newp_cnt;
          set_id(o_in_id[i], o_out_id[j], id);
          new_vid.push_back(id);
          newp_cnt++;
        }
      }
    // add the facets.
    // facet 0: p0, p3, p1.
    TrFace *new_f_1 = new TrFace(new_vid[0], new_vid[3], new_vid[1]);
    trmesh.f_list.push_back(new_f_1);
    // facet 1: p0, p3, p2.
    TrFace *new_f_2 = new TrFace(new_vid[0], new_vid[3], new_vid[2]);
    trmesh.f_list.push_back(new_f_2);
  } else if ((int)inP.size() == 3) {
    vector<int> new_vid;
    new_vid.resize(3, -1);
    for (int i = 0; i < 3; i++) {
      // intersection point i = 0,1,2.
      int id = -1;
      if (find_id(in_id[i], out_id[0], id))
        new_vid[i] = id;
      else {
        Point p = get_intersection_point(in_val, out_val, inP, outP, i, 0);
        // add the new point to the mesh.
        TrVertex *new_v = new TrVertex(p);
        trmesh.v_list.push_back(new_v);
        // record the id of this new point associated with in_id[0] and
        // out_id[i].
        id = newp_cnt;
        set_id(in_id[i], out_id[0], id);
        new_vid[i] = id;
        newp_cnt++;
      }
    }
    // add the facet between p0, p1, p2.
    TrFace *new_f = new TrFace(new_vid[0], new_vid[1], new_vid[2]);
    trmesh.f_list.push_back(new_f);
  } else
    cerr << "x"; // no-op.
}

bool is_p_outside(const Point &q, const Triangulation &triang, Cell_handle &c,
                  int &u, int &v) {
  Triangulation::Locate_type lt;
  u = -1;
  v = -1;
  c = triang.locate(q, lt, u, v);
  if (lt == Triangulation::OUTSIDE_AFFINE_HULL) {
    cerr << "Point " << q << " is outside the affine hull." << endl;
    return true;
  } else if (lt == Triangulation::OUTSIDE_CONVEX_HULL)
    return true;
  else {
    if (lt == Triangulation::CELL) {
      if (c->outside)
        return true;
      else
        return false;
    } else if (lt == Triangulation::FACET) {
      Cell_handle _c = c->neighbor(u);
      if (c->outside && _c->outside)
        return true;
      else
        return false;
    } else if (lt == Triangulation::EDGE) {
      if (is_outside_VF(triang, Edge(c, u, v)))
        return true;
      else
        return false;
    } else {
      CGAL_assertion(lt == Triangulation::VERTEX);
      return false;
    }
  }
}

bool is_p_outside(const Point &q, const Triangulation &triang,
                  const Cell_handle &start_cell, Cell_handle &c, int &u,
                  int &v) {
  Triangulation::Locate_type lt;
  u = -1;
  v = -1;
  c = triang.locate(q, lt, u, v, start_cell);
  if (lt == Triangulation::OUTSIDE_AFFINE_HULL) {
    cerr << "Point " << q << " is outside the affine hull." << endl;
    return true;
  } else if (lt == Triangulation::OUTSIDE_CONVEX_HULL)
    return true;
  else {
    if (lt == Triangulation::CELL) {
      if (c->outside)
        return true;
      else
        return false;
    } else if (lt == Triangulation::FACET) {
      Cell_handle _c = c->neighbor(u);
      if (c->outside && _c->outside)
        return true;
      else
        return false;
    } else if (lt == Triangulation::EDGE) {
      if (is_outside_VF(triang, Edge(c, u, v)))
        return true;
      else
        return false;
    } else {
      CGAL_assertion(lt == Triangulation::VERTEX);
      return false;
    }
  }
}

} // namespace Curation

/*
bool
is_boundary_bcc_tet(const IJK_quad& Q,
                    vector<Point>& outP, vector<Point>& inP,
                    vector<int>& out_id, vector<int>& in_id,
                    vector<double>& out_val, vector<double>& in_val,
                    KdTree& kd_tree,
                    const Triangulation& triang,
                    const vector<Vertex_handle>& vertices,
                    const vector<Facet>& cur_facets,
                    vector<unsigned short>& in_out_vec,
                    vector<double>& val_vec )
{
   inP.clear(); outP.clear(); in_id.clear(); out_id.clear(); in_val.clear();
out_val.clear(); Cell_handle c[4]; bool is_out[4] = {false, false, false,
false}; Point pts[4]; int ids[4] = {-1,-1,-1,-1}; double values[4]; for(int i
= 0; i < 4; i ++)
   {
      int _i = Q.q[i].first,
          _j = Q.q[i].second,
          _k = Q.q[i].third;
      // for every point check if it's inside or outside
      // by looking at which Delaunay triangle it stays in.
      Point p;
      if( i != 3 ) p = Point(xmin + xspan*_i, ymin + yspan*_j, zmin +
zspan*_k); else p = Point(xmin + xspan*(_i+.5), ymin + yspan*(_j+.5), zmin +
zspan*(_k+.5)); pts[i] = p;

      int id = _k*xdim*ydim + _j*xdim + _i;
      if( i == 3 ) id = xdim*ydim*zdim + _k*(xdim-1)*(ydim-1) + _j*(xdim-1) +
_i; ids[i] = id;

      if( in_out_vec[ids[i]] != 2 )
      {
         values[i] = val_vec[ids[i]];
         if( in_out_vec[ids[i]] == 0 )
            is_out[i] = false;
         else if( in_out_vec[ids[i]] == 1 )
            is_out[i] = true;
         else assert(false);
         continue;
      }

      int u,v;
      is_out[i] = is_p_outside(p, triang, c[i], u, v);
      if( is_out[i] ) in_out_vec[ids[i]] = 1;
      else in_out_vec[ids[i]] = 0;
      val_vec[ids[i]] = values[i] = 1; // df_to_curated_surf(pts[i], kd_tree,
triang, vertices, cur_facets);
   }
   // all out.
   if( is_out[0] && is_out[1] && is_out[2] && is_out[3] ) return false;
   // all in.
   if( (!is_out[0]) && (!is_out[1]) && (!is_out[2]) && (!is_out[3]) ) return
false;

   // this is a boundary tetrahedron, so we need to place
   // the values carefully.
   for(int i = 0; i < 4; i ++)
   {
      if( !is_out[i] )
      {
         inP.push_back(pts[i]);
         in_id.push_back(ids[i]);
         in_val.push_back(-values[i]);
      }
      else
      {
         outP.push_back(pts[i]);
         out_id.push_back(ids[i]);
         out_val.push_back(values[i]);
      }
   }
   return true;
}

void
fill_bcc_tet(const IJK& ijk,
             vector<IJK_quad>& bcc_tet)
{
   int i = ijk.first, j = ijk.second, k = ijk.third;
   // remember fourth vertex is the center of the voxel - always.
   // face - x,y,z=0
   bcc_tet.push_back(IJK_quad(IJK(i,j,k), IJK(i+1,j,k), IJK(i,j+1,k),
IJK(i,j,k))); bcc_tet.push_back(IJK_quad(IJK(i,j+1,k), IJK(i+1,j,k),
IJK(i+1,j+1,k), IJK(i,j,k)));
   // face - x=0,y,z
   bcc_tet.push_back(IJK_quad(IJK(i,j,k), IJK(i,j+1,k), IJK(i,j,k+1),
IJK(i,j,k))); bcc_tet.push_back(IJK_quad(IJK(i,j+1,k), IJK(i,j,k+1),
IJK(i,j+1,k+1), IJK(i,j,k)));
   // face - x,y=0,z
   bcc_tet.push_back(IJK_quad(IJK(i,j,k), IJK(i+1,j,k), IJK(i,j,k+1),
IJK(i,j,k))); bcc_tet.push_back(IJK_quad(IJK(i+1,j,k), IJK(i,j,k+1),
IJK(i+1,j,k+1), IJK(i,j,k)));
   // face - x,y,z=1
   bcc_tet.push_back(IJK_quad(IJK(i,j,k+1), IJK(i+1,j,k+1), IJK(i,j+1,k+1),
IJK(i,j,k))); bcc_tet.push_back(IJK_quad(IJK(i,j+1,k+1), IJK(i+1,j,k+1),
IJK(i+1,j+1,k+1), IJK(i,j,k)));
   // face - x=1,y,z
   bcc_tet.push_back(IJK_quad(IJK(i+1,j,k), IJK(i+1,j+1,k), IJK(i+1,j,k+1),
IJK(i,j,k))); bcc_tet.push_back(IJK_quad(IJK(i+1,j+1,k), IJK(i+1,j,k+1),
IJK(i+1,j+1,k+1), IJK(i,j,k)));
   // face - x,y=1,z
   bcc_tet.push_back(IJK_quad(IJK(i,j+1,k), IJK(i+1,j+1,k), IJK(i,j+1,k+1),
IJK(i,j,k))); bcc_tet.push_back(IJK_quad(IJK(i+1,j+1,k), IJK(i,j+1,k+1),
IJK(i+1,j+1,k+1), IJK(i,j,k)));
}
}


void
curate_vol( KdTree& kd_tree,
         const Triangulation& triang,
         const vector<Vertex_handle>& vertices,
         const vector<Facet>& cur_facets,
         TrMesh& trmesh )
{
   vector<unsigned short> in_out_vec;
   vector<double> val_vec;
   in_out_vec.resize(xdim*ydim*zdim + (xdim-1)*(ydim-1)*(zdim-1),2);
   val_vec.resize(xdim*ydim*zdim + (xdim-1)*(ydim-1)*(zdim-1),0);
   for(unsigned int k = 0; k < zdim-1; k ++)
   {
      for(unsigned int j = 0; j < ydim-1; j ++)
      {
         for(unsigned int i = 0; i < xdim-1; i ++)
         {
            vector<IJK_quad> bcc_tet;
            fill_bcc_tet(IJK(i,j,k), bcc_tet);
            for(int t = 0; t < (int)bcc_tet.size(); t ++)
            {
               IJK_quad Q = bcc_tet[t];
               vector<Point> outP, inP;
               vector<int> out_id, in_id;
               vector<double> out_val, in_val;
               if( is_boundary_bcc_tet(Q, outP, inP, out_id, in_id, out_val,
in_val, kd_tree, triang, vertices, cur_facets, in_out_vec, val_vec) )
                  build_bdy_tr_mesh(outP, inP, out_id, in_id, out_val, in_val,
trmesh);
            }
         }
      }
      cerr << ".";
   }
}  */
