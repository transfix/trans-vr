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

#ifndef MDS_H
#define MDS_H

#include <Curation/datastruct.h>

// arand: fixed, 4-25-2011
// #warning REMOVE USING NAMESPACE STD
// using namespace std;

namespace Curation {

// -----------------------------------------------------------------------
// Mesh Datastructure
// -----------------------------------------------------------------------

// stores the Euclidean coordinate of a vertex
// and the other attributes.

class MVertex {
public:
  MVertex() { init(); }

  MVertex(const Point &p) {
    coord = p;
    init();
  }

  void set_point(const Point &p) { coord = p; }
  Point point() const { return coord; }

  void set_mesh_normal(const Vector &v) { mv_normal = v; }
  Vector mesh_normal() const { return mv_normal; }

  void set_iso(const bool flag) { mv_iso = flag; }
  bool iso() const { return mv_iso; }

  void set_e_nm_flag(const bool flag) { mv_e_nm = flag; }
  bool e_nm() const { return mv_e_nm; }

  void set_v_nm_flag(const bool flag) { mv_v_nm = flag; }
  bool v_nm() const { return mv_v_nm; }

  void add_inc_vert(const int i) {
    inc_vert_list.push_back(i);
    num_inc_vert++;
  }
  int inc_vert(int i) const { return inc_vert_list[i]; }
  bool is_inc_vert(const int v) {
    for (int i = 0; i < num_inc_vert; i++)
      if (inc_vert_list[i] == v)
        return true;
    return false;
  }

  void add_inc_edge(const int i) {
    inc_edge_list.push_back(i);
    num_inc_edge++;
  }
  int inc_edge(int i) const { return inc_edge_list[i]; }
  bool get_eid(const int v, int &eid) {
    eid = -1;
    assert(num_inc_vert == num_inc_edge);
    for (int i = 0; i < num_inc_vert; i++)
      if (inc_vert_list[i] == v) {
        eid = inc_edge_list[i];
        return true;
      }
    return false;
  }

  void set_sp(const double &d) { mv_sp = d; }
  double sp() const { return mv_sp; }

  void set_sp_parent(const int &pid) { mv_sp_parent = pid; }
  int sp_parent() const { return mv_sp_parent; }

  int id;
  bool visited;

  int num_inc_edge;
  int num_inc_vert;

  inline void init() {
    mv_iso = false;
    mv_e_nm = false;
    mv_v_nm = false;
    id = -1;
    num_inc_vert = 0;
    num_inc_edge = 0;
    mv_sp = DBL_MAX;
    mv_sp_parent = -1;
    inc_vert_list.clear();
    inc_edge_list.clear();
  }

private:
  Point coord;
  Vector mv_normal;
  bool mv_iso;
  bool mv_e_nm;
  bool mv_v_nm;
  vector<int> inc_vert_list;
  vector<int> inc_edge_list;
  double mv_sp;
  int mv_sp_parent;
};

// stores the endpoints of an edge and other attributes.

class MEdge {
public:
  MEdge() { init(); }

  MEdge(const int v1, const int v2) {
    init();
    endpoint[0] = v1;
    endpoint[1] = v2;
  }

  void set_endpoint(const int i, const int val) { endpoint[i] = val; }
  int get_endpoint(int i) const { return endpoint[i]; }

  void add_inc_face(const int fid) {
    inc_face[num_inc_face] = fid;
    num_inc_face++;
  }
  void get_inc_face(int &f1, int &f2) {
    f1 = inc_face[0];
    f2 = inc_face[1];
  }

  void set_non_manifold_edge(const bool flag) {
    me_non_manifold_edge_flag = flag;
  }
  bool non_manifold_edge() const { return me_non_manifold_edge_flag; }

  void set_sharp_edge(const bool flag) { me_sharp_edge_flag = flag; }
  bool sharp_edge() const { return me_sharp_edge_flag; }

  int num_inc_face;

  inline void init() {
    endpoint[0] = endpoint[1] = -1;
    inc_face[0] = inc_face[1] = -1;
    num_inc_face = 0;
    me_non_manifold_edge_flag = false;
    me_sharp_edge_flag = false;
  }

private:
  int endpoint[2];
  int inc_face[2];

  bool me_non_manifold_edge_flag;
  bool me_sharp_edge_flag;
};

// stores the ids of three vertices of the face
// and any other attributes.

class MFace {
public:
  MFace() { init(); }
  MFace(int v1, int v2, int v3) {
    init();
    corner[0] = v1;
    corner[1] = v2;
    corner[2] = v3;
  }

  void set_corner(const int i, const int val) { corner[i] = val; }
  int get_corner(int i) const { return corner[i]; }

  void set_edge(const int i, const int val) { edge_array[i] = val; }
  int get_edge(int i) const { return edge_array[i]; }

  void set_color(const int i, const double c) { rgba[i] = c; }
  double get_color(int i) const { return rgba[i]; }

  bool visited;
  int id;

  inline void init() {
    corner[0] = corner[1] = corner[2] = -1;
    edge_array[0] = edge_array[1] = edge_array[2] = -1;
    rgba[0] = rgba[1] = rgba[2] = rgba[3] = 1;

    visited = false;
    id = -1;
  }

private:
  int corner[3];
  int edge_array[3];
  double rgba[4];
};

class Mesh {
public:
  Mesh() { init(); }
  Mesh(int v, int f) {
    init();
    // initialize the number of vertices and faces.
    nv = v;
    nf = f;
  }

  void set_nv(const int n) { nv = n; }
  int get_nv() const { return nv; }

  void set_nf(const int n) { nf = n; }
  int get_nf() const { return nf; }

  void set_ne(const int n) { ne = n; }
  int get_ne() const { return ne; }

  void add_vertex(MVertex v) { vert_list.push_back(v); }
  MVertex vertex(int i) const { return vert_list[i]; }

  void add_face(const MFace f) { face_list.push_back(f); }
  MFace face(int i) const { return face_list[i]; }

  vector<MVertex> vert_list;
  vector<MEdge> edge_list;
  vector<MFace> face_list;

  inline void init() {
    nv = 0;
    nf = 0;
    ne = 0;
    vert_list.clear();
    face_list.clear();
  }

private:
  // number of vertices.
  int nv;
  // number of faces.
  int nf;
  // number of edges.
  int ne;
};

} // namespace Curation

#endif // MDS_H
