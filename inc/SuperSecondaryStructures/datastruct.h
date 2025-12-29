#ifndef DATASTRUCT_H
#define DATASTRUCT_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <vector>

// for compiler discrepancies
#include <CGAL/Cartesian.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h> // implements Traits.
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Point_3.h>
#include <CGAL/Quotient.h>
#include <CGAL/Timer.h>
#include <CGAL/Triangulation_cell_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h> // check
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/basic.h>
#include <CGAL/number_utils.h>
#include <CGAL/utility.h>

using namespace std;

namespace SuperSecondaryStructures {

// #define __OUTSIDE__
#define __INSIDE__

// -----------------------------------------------------------------------
// Define a new Vertex_base with additional attributes.
// -----------------------------------------------------------------------

template <class Gt, class Vb = CGAL::Triangulation_vertex_base_3<Gt>>
class Seg_vertex : public Vb {
  typedef Vb Base;

public:
  typedef typename Vb::Point Point;
  typedef typename Point::R Rep;
  typedef typename Vb::Vertex_handle Vertex_handle;
  typedef typename Vb::Cell_handle Cell_handle;
  typedef CGAL::Vector_3<Rep> Vector;

  template <typename TDS2> struct Rebind_TDS {
    typedef typename Vb::template Rebind_TDS<TDS2>::Other Vb2;
    typedef Seg_vertex<Gt, Vb2> Other;
  };

  Seg_vertex() { init(); }
  Seg_vertex(const Point &p) : Base(p) { init(); }
  Seg_vertex(const Point &p, void *cell) : Base(p, cell) { init(); }
  Seg_vertex(void *cell) : Base(cell) { init(); }

  void set_normal(const Vector &v) { v_normal = v; }
  Vector normal() const { return v_normal; }

  void set_pole(const Point &p) { v_pole = p; }
  Point pole() const { return v_pole; }

  void set_width(const double &w) { v_width = w; }
  double width() const { return v_width; }

  void set_diameter(const double &d) { v_diameter = d; }
  double diameter() const { return v_diameter; }

  void set_flat(const bool b) { v_flat = b; }
  bool is_flat() const { return v_flat; }

  void set_convex_hull(const bool b) { v_chull = b; }
  bool on_convex_hull() const { return v_chull; }

  void set_isolated(const bool b) { v_isolated = b; }
  bool is_isolated() const { return v_isolated; }

  void set_umbrella(const bool b) { v_umbrella = b; }
  bool has_umbrella() const { return v_umbrella; }

  // --------- For Robust Cocone ----------

  void set_on_smooth_surface(const bool b) { v_on_smooth_surface = b; }
  bool on_smooth_surface() const { return v_on_smooth_surface; }

  // --------------------------------------

  void set_outs2_bdy(const bool &b) { v_outs2_bdy = b; }
  bool outs2_bdy() const { return v_outs2_bdy; }

  void set_in_s1(const bool &b) { v_in_s1 = b; }
  bool in_s1() const { return v_in_s1; }

  int id;
  bool tag; // Some tag needed during computations.
  bool visited;
  bool bad;
  bool bad_neighbor;
  vector<int> bad_v;

  // --------- for robust cocone -----
  vector<double> nnd_vector;
  // --------------

  // For Medial Axis
  vector<Vector> normal_stack; // Used to store the normals of the
                               // incident disk

private:
  inline void init() {
    v_normal = CGAL::NULL_VECTOR;
    v_pole = this->point();
    v_width = 0.0;
    v_diameter = 0.0;
    v_flat = true;
    v_chull = false;
    v_umbrella = false;
    v_isolated = true;

    // -------- for robust cocone ----
    v_on_smooth_surface = false;
    nnd_vector.clear();
    nnd_vector.resize(3, HUGE);

    // -------------------------------

    v_outs2_bdy = false;
    v_in_s1 = false;
  }

  Vector v_normal;   // Estimated normal at the vertex.
  Point v_pole;      // The negative pole.
  double v_width;    // Radius of the cocone.
  double v_diameter; // Diamter of the the incident umbrella.
  bool v_flat;       // Indicates whether the vertex is flat.
  bool v_chull;      // Indicates wheter the vertex lies on the boundary
                     // of the convex hull.
  bool v_umbrella;   // Indicates if v has a (non sharp) umbrella.
  bool v_isolated;   // Indicates whether the vertex is isolated.

  // ------ robust cocone ----
  bool v_on_smooth_surface;
  // -------------------------

  bool v_outs2_bdy;
  bool v_in_s1;
};

// -----------------------------------------------------------------------
// Define a new Cell_base storing associated Voronoi vertices
// and cocone flags.
// -----------------------------------------------------------------------

template <class Gt, class Cb = CGAL::Triangulation_cell_base_3<Gt>>
class Seg_cell : public Cb {
  typedef Cb Base;

public:
  typedef typename Cb::Point Point;
  typedef typename Cb::Vertex_handle Vertex_handle;
  typedef typename Cb::Cell_handle Cell_handle;

  Seg_cell() { init(); }
  Seg_cell(Vertex_handle v0, Vertex_handle v1, Vertex_handle v2,
           Vertex_handle v3)
      : Base(v0, v1, v2, v3) {
    init();
  }
  Seg_cell(Vertex_handle v0, Vertex_handle v1, Vertex_handle v2,
           Vertex_handle v3, Cell_handle c0, Cell_handle c1, Cell_handle c2,
           Cell_handle c3)
      : Base(v0, v1, v2, v3, c0, c1, c2, c3) {
    init();
  }

  template <typename TDS2> struct Rebind_TDS {
    typedef typename Cb::template Rebind_TDS<TDS2>::Other Cb2;
    typedef Seg_cell<Gt, Cb2> Other;
  };

  void set_voronoi(const Point &p) { c_voronoi = p; }
  Point voronoi() const { return c_voronoi; }

  void set_sink(const bool &b) { c_sink = b; }
  bool sink() const { return c_sink; }

  void set_dirty(const bool &b) { c_dirty = b; }
  bool dirty() const { return c_dirty; }

  void set_cosph_pair(const int &i, const bool &b) { c_cosph_pair[i] = b; }
  bool cosph_pair(int i) const { return c_cosph_pair[i]; }

  void set_flip(const int &i, const bool &b) { c_flip[i] = b; }
  bool flip(int i) const { return c_flip[i]; }

  void set_saddle(const int &i, const bool &b) { c_saddle[i] = b; }
  bool saddle(int i) const { return c_saddle[i]; }

  void set_source(const int &i, const bool &b) { c_source[i] = b; }
  bool source(int i) const { return c_source[i]; }

  void set_terminus(const int &i, const bool &b) { c_terminus[i] = b; }
  bool terminus(int i) const { return c_terminus[i]; }

  void set_cell_radius(const double r) { c_cell_radius = r; }
  double cell_radius() const { return c_cell_radius; }

  void set_con_cl_id(const int &id) { c_con_cl_id = id; }
  int con_cl_id() const { return c_con_cl_id; }

  void set_cluster_id(const int &i) { c_cluster_id = i; }
  int cluster_id() const { return c_cluster_id; }

  void set_cosph_leader(const bool &b) { c_cosph_leader = b; }
  bool cosph_leader() const { return c_cosph_leader; }

  void set_cosph_leader_id(const int &id) { c_cosph_leader_id = id; }
  int cosph_leader_id() const { return c_cosph_leader_id; }

  void set_cocone_flag(int i, bool new_value) {
    CGAL_precondition(0 <= i && i < 4);
    c_cocone_flag[i] = new_value;
  }

  bool cocone_flag(int i) const {
    CGAL_precondition(0 <= i && i < 4);
    return c_cocone_flag[i];
  }

  void set_removable(int i, bool new_value) {
    CGAL_precondition(0 <= i && i < 4);
    c_removable[i] = new_value;
  }

  bool removable(int i) const {
    CGAL_precondition(0 <= i && i < 4);
    return c_removable[i];
  }

  void set_mouth(const int &id, const bool &b) { c_mouth[id] = b; }
  bool mouth(int id) const { return c_mouth[id]; }

  // Functions for Medial Axis
  void set_VF_on_medax(const int i, const int j, const bool val) {
    CGAL_precondition(0 <= i && i <= 3);
    CGAL_precondition(0 <= j && j <= 3);
    CGAL_precondition(i != j);
    c_VF_on_medax[i][j] = val;
    c_VF_on_medax[j][i] = val;
  }

  bool VF_on_medax(int i, int j) const {
    CGAL_precondition(0 <= i && i <= 3);
    CGAL_precondition(0 <= j && j <= 3);
    CGAL_precondition(i != j);
    return c_VF_on_medax[i][j];
  }

  void set_VV_on_medax(const bool &b) { c_VV_on_medax = b; }
  bool VV_on_medax() const { return c_VV_on_medax; }

  void set_bb(const bool flag) { c_bb = flag; }
  bool bb() const { return c_bb; }

  void set_deep_int(const int i, const bool flag) { c_deep_int[i] = flag; }
  bool deep_int(int i) const { return c_deep_int[i]; }

  bool tag[4]; // Some tag needed in computations.
  bool visited;
  bool f_visited[4];
  int id;
  bool flag;

  bool transp;
  bool outside;

  bool bdy[4];
  bool opaque[4];
  int umbrella_member[4][4];

  inline void init() {
    c_voronoi = CGAL::ORIGIN;
    c_sink = false;
    c_dirty = false;
    c_cell_radius = 0.0;
    c_cluster_id = -1;
    c_cosph_leader = false;
    c_cosph_leader_id = -1;
    c_con_cl_id = -1;
    c_bb = false;
    c_VV_on_medax = false;
    for (int i = 0; i < 4; i++) {
      c_cosph_pair[i] = false;
      c_flip[i] = false;
      c_saddle[i] = false;
      c_source[i] = false;
      c_terminus[i] = false;
      c_cocone_flag[i] = false;
      c_removable[i] = true;
      c_deep_int[i] = false;
      f_visited[i] = false;

      c_mouth[i] = false;

      for (int j = 0; j < 4; j++)
        c_VF_on_medax[i][j] = false;
    }
  }

private:
  Point c_voronoi;
  bool c_dirty;
  bool c_sink;
  bool c_cosph_pair[4];
  bool c_flip[4];
  bool c_saddle[4];
  bool c_source[4];
  bool c_terminus[4];
  double c_cell_radius;
  int c_cluster_id;
  bool c_cosph_leader;
  int c_cosph_leader_id;
  bool c_VV_on_medax;
  bool c_VF_on_medax[4][4];

  bool c_cocone_flag[4];
  bool c_removable[4];

  bool c_deep_int[4];
  bool c_bb;

  int c_con_cl_id;

  bool c_mouth[4];
};

struct K : CGAL::Exact_predicates_inexact_constructions_kernel {};

typedef Seg_vertex<K> Vertex;
typedef Seg_cell<K> Cell;
typedef CGAL::Triangulation_data_structure_3<Vertex, Cell> TDS;
typedef CGAL::Delaunay_triangulation_3<K, TDS> Triangulation;

typedef Triangulation::Point Point;
typedef Point::R Rep;
typedef CGAL::Tetrahedron_3<Rep> Tetrahedron;
typedef CGAL::Triangle_3<Rep> Triangle_3;
typedef CGAL::Vector_3<Rep> Vector;
typedef CGAL::Ray_3<Rep> Ray;
typedef CGAL::Segment_3<Rep> Segment;

typedef Triangulation::Vertex_handle Vertex_handle;
typedef Triangulation::Edge Edge;
typedef Triangulation::Facet Facet;
typedef Triangulation::Cell_handle Cell_handle;

typedef Triangulation::Finite_vertices_iterator FVI;
typedef Triangulation::Finite_edges_iterator FEI;
typedef Triangulation::Finite_facets_iterator FFI;
typedef Triangulation::Finite_cells_iterator FCI;
typedef Triangulation::All_vertices_iterator AVI;
typedef Triangulation::All_edges_iterator AEI;
typedef Triangulation::All_facets_iterator AFI;
typedef Triangulation::All_cells_iterator ACI;

typedef Triangulation::Facet_circulator Facet_circulator;
typedef Triangulation::Cell_circulator Cell_circulator;

// -----------------------------------------------------------------------
// SEGMENT :
// ---------
// Clusters are to be maintained in a UNION data structure.
// -----------------------------------------------------------------------
struct cell_cluster {

  // Constructor
  cell_cluster()
      : rep(0), nxt(0), tail(0), in_cluster(false), outside(true), sq_r(0) {}
  cell_cluster(int i)
      : rep(i), nxt(0), tail(i), in_cluster(false), outside(true), sq_r(0) {}
  cell_cluster(int i, double r)
      : rep(i), nxt(0), tail(i), in_cluster(false), outside(true), sq_r(r) {}

  // Member Functions
  int find() const { return rep; }
  bool is_in_cluster() const { return in_cluster; }

  // Fields for traversal and ownership.
  int rep;
  cell_cluster *nxt;
  int tail;

  // Fields for attributes.
  bool in_cluster;
  bool outside;
  double sq_r;
};

} // namespace SuperSecondaryStructures
#endif // DATASTRUCT_H
