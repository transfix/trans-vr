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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef __SKEL_H__
#define __SKEL_H__

#include <Histogram/histogram_data.h>
#include <SecondaryStructures/datastruct_ss.h>
#include <SecondaryStructures/helix.h>
#include <SecondaryStructures/hfn_util.h>
#include <SecondaryStructures/init.h>
#include <SecondaryStructures/intersect.h>
#include <SecondaryStructures/medax.h>
#include <SecondaryStructures/op.h>
#include <SecondaryStructures/rcocone.h>
#include <SecondaryStructures/robust_cc.h>
#include <SecondaryStructures/tcocone.h>
#include <SecondaryStructures/u1.h>
#include <SecondaryStructures/u2.h>
#include <SecondaryStructures/util.h>
#include <cvcraw_geometry/cvcgeom.h>

class MyPolyline {
public:
  MyPolyline() {}
  MyPolyline(const int &_nv, const int &_ne) {
    nv = _nv;
    ne = _ne;
  }
  ~MyPolyline(){};
  int nv;
  int ne;
  vector<SecondaryStructures::Point> vlist;
  vector<pair<int, int>> elist;
};

class SVertex {
public:
  SVertex() { init(); }
  SVertex(const SecondaryStructures::Point &p) {
    coord = p;
    init();
  }
  ~SVertex() {}

  void set_point(const SecondaryStructures::Point &p) { coord = p; }
  SecondaryStructures::Point point() const { return coord; }

  void set_iso(const bool flag) { sv_iso = flag; }
  bool iso() const { return sv_iso; }

  void set_e_nm_flag(const bool flag) { sv_e_nm = flag; }
  bool e_nm() const { return sv_e_nm; }

  void set_v_nm_flag(const bool flag) { sv_v_nm = flag; }
  bool v_nm() const { return sv_v_nm; }

  void add_inc_vert(const int i) {
    inc_vert_list.push_back(i);
    num_inc_vert++;
  }
  int inc_vert(int i) const { return inc_vert_list[i]; }
  bool is_inc_vert(const int v) {
    for (int i = 0; i < num_inc_vert; i++)
      if (inc_vert_list[i] == v) {
        return true;
      }
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

  void add_inc_face(const int i) {
    inc_face_list.push_back(i);
    num_inc_face++;
  }
  int inc_face(int i) const { return inc_face_list[i]; }

  int id;
  bool visited;
  bool on_u1, on_u2;

  int num_inc_face;
  int num_inc_edge;
  int num_inc_vert;

  inline void init() {
    sv_iso = true;
    sv_e_nm = false;
    sv_v_nm = false;
    id = -1;
    on_u1 = on_u2 = false;
    num_inc_vert = 0;
    num_inc_edge = 0;
    num_inc_face = 0;
    inc_vert_list.clear();
    inc_edge_list.clear();
  }

private:
  SecondaryStructures::Point coord;
  bool sv_iso;
  bool sv_e_nm;
  bool sv_v_nm;
  vector<int> inc_vert_list;
  vector<int> inc_edge_list;
  vector<int> inc_face_list;
};

class SEdge {
public:
  SEdge() { init(); }
  SEdge(const int v1, const int v2) {
    init();
    endpoint[0] = v1;
    endpoint[1] = v2;
  }
  void set_endpoint(const int i, const int val) { endpoint[i] = val; }
  int get_endpoint(int i) const { return endpoint[i]; }
  void add_inc_face(const int fid) {
    inc_face_list.push_back(fid);
    num_inc_face++;
  }
  int get_n_inc_faces() const { return inc_face_list.size(); }
  int inc_face(int i) const { return inc_face_list[i]; }

  void set_enm(const bool flag) { se_enm = flag; }
  bool enm() const { return se_enm; }
  int num_inc_face;
  bool on_u1, on_u2;
  double width; // circumradius of dual Delaunay triangle.
  inline void init() {
    endpoint[0] = endpoint[1] = -1;
    inc_face_list.clear();
    num_inc_face = 0;
    se_enm = false;
    on_u1 = on_u2 = false;
    width = 0;
  }

private:
  int endpoint[2];
  vector<int> inc_face_list;
  bool se_enm;
};

class SFace {
public:
  SFace() { init(); }
  SFace(const vector<int> &_vlist) {
    init();
    for (int i = 0; i < (int)_vlist.size(); i++) {
      vlist.push_back(_vlist[i]);
    }
    v_cnt = (int)vlist.size();
  }

  void add_edge(const int eid) {
    elist.push_back(eid);
    edge_cnt++;
  }
  int edge(int i) const { return elist[i]; }

  int get_vertex(int i) const { return vlist[i]; }

  bool visited;
  int id;
  int comp_id;

  double width; // lenght of dual Delaunay edge.
  bool beta;

  int edge_cnt;
  int v_cnt;

  inline void init() {
    vlist.clear();
    elist.clear();
    visited = false;
    id = -1;
    comp_id = -1;
    width = 0;
    beta = false;
    edge_cnt = 0;
    v_cnt = 0;
  }

private:
  vector<int> vlist;
  vector<int> elist;
};

class Skel {

public:
  Skel();

  void set_nv(const int n) { nv = n; }
  int get_nv() const { return nv; }

  void set_nf(const int n) { nf = n; }
  int get_nf() const { return nf; }

  void set_ne(const int n) { ne = n; }
  int get_ne() const { return ne; }

  void add_vertex(SVertex v) { vert_list.push_back(v); }
  SVertex vertex(int i) const { return vert_list[i]; }

  void add_edge(SEdge e) { edge_list.push_back(e); }
  SEdge edge(int i) const { return edge_list[i]; }

  void add_face(const SFace f) { face_list.push_back(f); }
  SFace face(int i) const { return face_list[i]; }

  /* inital computations */
  int compute_secondary_structures(cvcraw_geometry::cvcgeom_t *inputGeom);
  /* additional "interactive computations" */
  HistogramData get_alpha_histogram_data();
  HistogramData get_beta_histogram_data();
  //	void update_display(Geometry* helixGeom, Geometry* sheetGeom,
  //Geometry* curveGeom,
  void buildAllGeometry(int alphaCount, int betaCount, float alphaMinWidth,
                        float alphaMaxWidth, float betaMinWidth,
                        float betaMaxWidth, bool alphaHistogramChanged,
                        bool betaHistogramChanged);

  int getAlphaCount() { return helix_cnt; }
  int getBetaCount() { return beta_cnt; }
  int getDefaultAlphaCount() { return DEFAULT_ALPHA_COUNT; }
  int getDefaultBetaCount() { return DEFAULT_BETA_COUNT; }
  cvcraw_geometry::cvcgeom_t *helixGeom;
  cvcraw_geometry::cvcgeom_t *sheetGeom;
  cvcraw_geometry::cvcgeom_t *skelGeom;
  cvcraw_geometry::cvcgeom_t *curveGeom;

private:
  // interative options:
  // For selection of beta sheets and alpha helices.
  int helix_cnt;
  int beta_cnt;
  int _max_sheets;
  double bw;
  double b_tol;

  /* skel state */
  vector<SVertex> vert_list;
  vector<SEdge> edge_list;
  vector<SFace> face_list;

  // colors for beta sheets
  struct Color {
    Color(float r_ = 0, float g_ = 0, float b_ = 0, float a_ = 1.0f) {
      r = r_;
      g = g_;
      b = b_;
      a = a_;
    }
    float r, g, b, a;
  };

  // helices
  vector<vector<SecondaryStructures::Point>> helices;

  // beta sheets
  vector<int> comp_id; // components sorted by area
  int comp_cnt;
  vector<vector<int>> comps;
  std::vector<Color> comp_colors;
  vector<double> A;
  vector<SecondaryStructures::Point> C;
  vector<bool> comp_pl;
  vector<vector<int>> star;

  MyPolyline L;

  // number of vertices.
  int nv;
  // number of faces.
  int nf;
  // number of edges.
  int ne;

  // -- flatness marking ---
  static const double DEFAULT_ANGLE;

  // -- robust cocone ---
  static const double DEFAULT_BIGBALL_RATIO;
  static const double DEFAULT_THETA_IF_d;
  static const double DEFAULT_THETA_FF_d;

  // -- medial axis ---
  // static const double DEFAULT_MED_THETA = M_PI*22.5/180.0; // original:
  // M_PI*22.5/180.0;
  static const double DEFAULT_MED_RATIO;

  // -- bounding box ---
  vector<double> bounding_box;
  static const double BB_SCALE;
  double bbox_diagonal;

  static const int REFINE_FACTOR = 10;

  // -- default constraint options
  static const int DEFAULT_ALPHA_COUNT = 20;
  static const int DEFAULT_BETA_COUNT = 2;

  // -- thresholds
  // static const int SHEET_AREA_THRESHOLD = 20;
  static const int SHEET_AREA_THRESHOLD = 10; // GUESS

  /* methods that operate on the skel */
  void add_u1_to_skel(SecondaryStructures::Triangulation &triang);
  void
  add_u2_to_skel(const pair<vector<vector<SecondaryStructures::Cell_handle>>,
                            vector<SecondaryStructures::Facet>> &u2);
  bool in_same_comp(const int &v1, const int &v2) const;
  void do_star();
  void filter(const int &pc);
  void refine_skel(const double &eps);

  /* op methods that operate on the skel */
  void write_u1_skel(const char *file_prefix) const;
  void write_u2_skel(const char *file_prefix) const;
  void write_L_skel(const char *file_prefix) const;
  void write_skel(const char *file_prefix) const;
  void write_beta(const char *file_prefix) const;
  void write_axis(const SecondaryStructures::Triangulation &triang,
                  const int &biggest_medax_comp_id, const char *file_prefix);

  /* helix methods that operate on the skel */
  float _alphaMinWidth, _alphaMaxWidth;
  float _betaMinWidth, _betaMaxWidth;
  vector<SEdge> get_valid_edge_list();
  void compute_helices(const char *file_prefix);

  /* saved internal state for building points */
  vector<vector<int>> _curve_comps;
  Curve _curve;

  /* build geometry */
  cvcraw_geometry::cvcgeom_t *buildSheetGeometry();
  cvcraw_geometry::cvcgeom_t *buildHelixGeometry();
  cvcraw_geometry::cvcgeom_t *buildSkeletonGeometry();
  cvcraw_geometry::cvcgeom_t *
  buildCurveGeometry(const std::vector<std::vector<int>> &curve_comps,
                     const Curve &curve);
};

#endif
