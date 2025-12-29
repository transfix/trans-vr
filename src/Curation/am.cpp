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

#include <Curation/am.h>

namespace Curation {

// ------------------------
// find_nonmanifold_region
// ------------------------
void find_nonmanifold_region(Mesh &mesh) {

  int nv = mesh.get_nv();

  bool is_there_any_edge_non_manifold = false;
  bool is_there_any_vertex_non_manifold = false;

  for (int i = 0; i < nv; i++) {
    // check if ith vertex has a good umbrella.

    // rotate around this vertex and check if
    // there is any non-manifold behaviour in
    // the umbrella incident on this vertex.

    MVertex mv = mesh.vert_list[i];
    if (mv.iso())
      continue;
    if (mv.num_inc_edge == 0)
      continue;

    // start with one edge incident on the vertex.
    int eid = mv.inc_edge(0);

    // in a local vector mark it visited.
    vector<int> vis_e_vector;
    vis_e_vector.push_back(eid);

    vector<int> vis_f_vector;

    // put this edge into a stack.
    vector<int> e_stack;
    e_stack.push_back(eid);

    bool is_non_manifold = false;

    while (!e_stack.empty()) {
      int e = e_stack.back();
      e_stack.pop_back();

      // check if this edge is manifold.
      if (mesh.edge_list[e].num_inc_face != 2) {
        // mark the edge and the incident
        // vertices non-manifold.

        is_non_manifold = true;
        break;
      }

      // if the edge is manifold then rotate around
      // the vertex to collect another edge.

      // take the facet incident on the edge which
      // is not visited.
      int f[2] = {-1, -1};
      mesh.edge_list[e].get_inc_face(f[0], f[1]);

      assert(f[0] != -1 && f[1] != -1);

      for (int j = 0; j < 2; j++) {
        bool found = false;
        for (int k = 0; k < (int)vis_f_vector.size(); k++) {
          if (vis_f_vector[k] == f[j]) {
            found = true;
            break;
          }
        }
        if (found)
          continue;

        // else mark this face visited.
        vis_f_vector.push_back(f[j]);

        // if not found then this is a new facet
        // push the other edge incident to this vertex
        // on this facet in the stack.
        int inc_v = mesh.face_list[f[j]].get_corner(0) +
                    mesh.face_list[f[j]].get_corner(1) +
                    mesh.face_list[f[j]].get_corner(2) -
                    mesh.edge_list[e].get_endpoint(0) -
                    mesh.edge_list[e].get_endpoint(1);
        // now the new edge is between i and inc_v.
        int new_eid = -1;
        assert(mv.get_eid(inc_v, new_eid));
        assert(new_eid != -1);

        // find if new_eid is already visited.
        // if not push it into the stack.

        found = false;
        for (int k = 0; k < (int)vis_e_vector.size(); k++) {
          if (vis_e_vector[k] == new_eid) {
            found = true;
            break;
          }
        }

        if (found)
          continue;
        else {
          e_stack.push_back(new_eid);
          vis_e_vector.push_back(new_eid);
        }
      }
    }

    if (is_non_manifold) {
      // mark the vertex.
      mesh.vert_list[i].set_e_nm_flag(true);
      is_there_any_edge_non_manifold = true;
    } else {
      if ((int)vis_e_vector.size() < mv.num_inc_vert) {
        // mark the vertex again because of pinching.
        mesh.vert_list[i].set_v_nm_flag(true);
        is_there_any_vertex_non_manifold = true;
      }
    }
  }

  if (is_there_any_edge_non_manifold)
    cerr << "edge non manifold" << " ";
  if (is_there_any_vertex_non_manifold)
    cerr << "vertex non manifold" << " ";
}

/*

// ------------------------
// mark_sharp_edges
// ------------------------
void
mark_sharp_edges(Mesh &mesh)
{

    for(int i = 0; i < mesh.get_ne(); i ++)
    {
            MEdge me = mesh.edge_list[i];
            if(me.non_manifold_edge()) continue;
            if(me.num_inc_face != 2) continue;

            // find the two incident faces and check the angle between them
            int f1 = -1, f2 = -1;
            me.get_inc_face(f1, f2);
            MFace mf1 = mesh.face_list[f1];
            MFace mf2 = mesh.face_list[f2];

            int e0 = me.get_endpoint(0);
            int e1 = me.get_endpoint(1);

            int u0 = mf1.get_corner(0) + mf1.get_corner(1) + mf1.get_corner(2)
- e0 - e1; int v0 = mf2.get_corner(0) + mf2.get_corner(1) + mf2.get_corner(2)
- e0 - e1;

            VECTOR3 n1 = cross((mesh.vert_list[e1].point() -
                                             mesh.vert_list[e0].point()),
                                            (mesh.vert_list[u0].point() -
                                             mesh.vert_list[e1].point()) );
            VECTOR3 n2 = cross((mesh.vert_list[e0].point() -
                                             mesh.vert_list[e1].point()),
                                            (mesh.vert_list[v0].point() -
                                             mesh.vert_list[e0].point()) );
            if(cosine(n1,n2) < -0.8)
                    mesh.edge_list[i].set_sharp_edge(true);
    }
}

*/

// ------------------------
// align_mesh_triangles
// ------------------------
void align_mesh_triangles(Mesh &mesh) {
  int nf = mesh.get_nf();

  for (int i = 0; i < nf; i++) {
    MFace mf = mesh.face_list[i];
    if (mf.visited)
      continue;

    // if the face has any nonmanifold vertex continue.
    if (mesh.vert_list[mf.get_corner(0)].e_nm() ||
        mesh.vert_list[mf.get_corner(1)].e_nm() ||
        mesh.vert_list[mf.get_corner(2)].e_nm())
      continue;
    if (mesh.vert_list[mf.get_corner(0)].v_nm() ||
        mesh.vert_list[mf.get_corner(1)].v_nm() ||
        mesh.vert_list[mf.get_corner(2)].v_nm())
      continue;

    // keep a stack of already oriented faces in the boundary.
    vector<int> walk_stack;

    walk_stack.push_back(i);
    mesh.face_list[i].visited = true;

    while (!walk_stack.empty()) {
      int fid = walk_stack.back();
      MFace f = mesh.face_list[fid];
      walk_stack.pop_back();

      assert(f.visited);

      int v0 = f.get_corner(0), v1 = f.get_corner(1), v2 = f.get_corner(2);

      // get the three faces incident to it.
      for (int j = 0; j < 3; j++) {
        // get the neighbor.
        int eid = f.get_edge(j);

        // if the edge is non-manifold continue.
        if (mesh.edge_list[eid].num_inc_face != 2)
          continue;
        // if the edge is sharp continue.
        if (mesh.edge_list[eid].sharp_edge())
          continue;

        int f1 = -1, f2 = -1;
        mesh.edge_list[eid].get_inc_face(f1, f2);
        int new_f = (f1 == fid) ? f2 : f1;
        if (new_f == -1)
          continue;
        if (mesh.face_list[new_f].visited)
          continue;

        // align this face and put it into the stack.
        int e0 = mesh.edge_list[eid].get_endpoint(0),
            e1 = mesh.edge_list[eid].get_endpoint(1);

        int w0 = mesh.face_list[new_f].get_corner(0),
            w1 = mesh.face_list[new_f].get_corner(1),
            w2 = mesh.face_list[new_f].get_corner(2);

        if ((e0 == v0 && e1 == v1) || (e0 == v1 && e1 == v2) ||
            (e0 == v2 && e1 == v0)) {
          mesh.face_list[new_f].set_corner(0, w0 + w1 + w2 - e0 - e1);
          mesh.face_list[new_f].set_corner(1, e1);
          mesh.face_list[new_f].set_corner(2, e0);
        } else if ((e0 == v1 && e1 == v0) || (e0 == v2 && e1 == v1) ||
                   (e0 == v0 && e1 == v2)) {
          mesh.face_list[new_f].set_corner(0, w0 + w1 + w2 - e0 - e1);
          mesh.face_list[new_f].set_corner(1, e0);
          mesh.face_list[new_f].set_corner(2, e1);
        } else
          assert(false);

        // before oushing the new face in the stack
        // see if this has any nonmanifold vertex.
        // if the face has any nonmanifold vertex continue.
        MFace mnew_f = mesh.face_list[new_f];
        if (mesh.vert_list[mnew_f.get_corner(0)].e_nm() ||
            mesh.vert_list[mnew_f.get_corner(1)].e_nm() ||
            mesh.vert_list[mnew_f.get_corner(2)].e_nm())
          continue;
        if (mesh.vert_list[mnew_f.get_corner(0)].v_nm() ||
            mesh.vert_list[mnew_f.get_corner(1)].v_nm() ||
            mesh.vert_list[mnew_f.get_corner(2)].v_nm())
          continue;

        walk_stack.push_back(new_f);
        // mark the face visited.
        mesh.face_list[new_f].visited = true;
      }
    }
  }
}

void am(Mesh &mesh) {
  // read the mesh and align it.
  cerr << ".";
  find_nonmanifold_region(mesh);
  cerr << ".";
  // mark_sharp_edges(mesh);
  // cerr << ".";
  align_mesh_triangles(mesh);
  cerr << ".";

  return;
}

void compute_mesh_vertex_normal(Mesh &mesh, const bool &flip) {
  // for every vertex, compute the average normal of incident triangles.
  // store that in the vertex.
  for (int i = 0; i < mesh.get_nv(); i++) {
    MVertex mv = mesh.vert_list[i];
    if (mv.e_nm() || mv.v_nm())
      continue;
    Vector normal = CGAL::NULL_VECTOR;
    for (int j = 0; j < mv.num_inc_edge; j++) {
      MEdge me = mesh.edge_list[mv.inc_edge(j)];
      // find the two incident facets.
      int fid[2] = {-1, -1};
      me.get_inc_face(fid[0], fid[1]);

      MFace mf[2] = {mesh.face_list[fid[0]], mesh.face_list[fid[1]]};

      for (int k = 0; k < 2; k++) {
        Point v0 = mesh.vert_list[mf[k].get_corner(0)].point();
        Point v1 = mesh.vert_list[mf[k].get_corner(1)].point();
        Point v2 = mesh.vert_list[mf[k].get_corner(2)].point();
        normal = normal + CGAL::cross_product(v1 - v0, v2 - v0);
      }
    }
    normalize(normal);
    if (flip)
      normal = -1.0 * normal;
    mesh.vert_list[i].set_mesh_normal(normal);
  }
}

} // namespace Curation
