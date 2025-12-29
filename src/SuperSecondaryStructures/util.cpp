
#include <SuperSecondaryStructures/util.h>

namespace SuperSecondaryStructures {

double my_drand() { return ((double)(rand() % 10000)) / 10000.0; }

void normalize(Vector &v) {
  v = (1. / (sqrt(CGAL::to_double(v * v)))) * v;
  return;
}

double length_of_seg(const Segment &s) {
  return sqrt(
      CGAL::to_double((s.point(0) - s.point(1)) * (s.point(0) - s.point(1))));
}

// check if the angle <p0,p2,p1 > 90 degree
bool is_obtuse(const Point &p0, const Point &p1, const Point &p2) {
  Vector v0 = p0 - p2;
  Vector v1 = p1 - p2;
  return (CGAL::to_double(v0 * v1) < 0);
}

double cell_volume(const Cell_handle &c) {
  Tetrahedron t = Tetrahedron(c->vertex(0)->point(), c->vertex(1)->point(),
                              c->vertex(2)->point(), c->vertex(3)->point());
  return (CGAL::to_double(CGAL::abs(t.volume())));
}

bool is_inf_VF(const Triangulation &triang, const Cell_handle &c,
               const int uid, const int vid) {
  Facet_circulator fcirc = triang.incident_facets(Edge(c, uid, vid));
  Facet_circulator begin = fcirc;
  do {
    Cell_handle cur_c = (*fcirc).first;
    if (triang.is_infinite(cur_c))
      return true;
    fcirc++;
  } while (fcirc != begin);
  return false;
}

// -----------------------------
// find_0_volume_tetrahedra
// -----------------------------
// Identify the tetrahedra which are
// coplanar and mark them dirty
// -----------------------------
bool find_0_volume_tetrahedron(Triangulation &triang) {

  bool is_there_atleast_one_0_volume_tetrahedron = false;

  for (FCI cit = triang.finite_cells_begin();
       cit != triang.finite_cells_end(); cit++) {
    Tetrahedron t =
        Tetrahedron(cit->vertex(0)->point(), cit->vertex(1)->point(),
                    cit->vertex(2)->point(), cit->vertex(3)->point());
    // if(t.volume() <= NT(0))
    if (!CGAL::is_positive(CGAL::abs(t.volume()))) {
      cit->set_dirty(true);
      is_there_atleast_one_0_volume_tetrahedron = true;
    }
  }

  return is_there_atleast_one_0_volume_tetrahedron;
}

// -----------------------------
// check_del_vor_property
// ----------------------
// Test the correctness of the
// Delaunay and voronoi properties in a certain
// Triangulation.
// -----------------------------
bool check_del_vor_property(Triangulation &triang) {

  bool is_there_atleast_one_violation_of_del_vor_property = false;

  // Check : If the circumcenters of a pair fo tetrahedron satisfy
  //         the properties of delaunay and voronoi.

  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); fit++) {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;

    if (triang.is_infinite(ch) || triang.is_infinite(ch->neighbor(id)))
      continue;
    Cell_handle c[2];
    c[0] = ch;
    c[1] = ch->neighbor(id);

    Point p[5];
    p[0] = c[0]->vertex(id)->point();
    p[1] = c[0]->vertex((id + 1) % 4)->point();
    p[2] = c[0]->vertex((id + 2) % 4)->point();
    p[3] = c[0]->vertex((id + 3) % 4)->point();
    p[4] = c[1]->vertex(c[1]->index(c[0]))->point();

    Point vp[2];
    vp[0] = ch->voronoi();
    vp[1] = ch->neighbor(id)->voronoi();

    Tetrahedron t[2];
    t[0] = Tetrahedron(p[1], p[2], p[3], p[0]);
    t[1] = Tetrahedron(p[1], p[2], p[3], p[4]);

    // test - 1
    // The circumcenters shouldn't flip.

    // if one of the tetrahedron is already dirty
    // this computation is not very stable.
    if (c[0]->dirty() || c[1]->dirty())
      continue;

    Tetrahedron tvp[2];
    tvp[0] = Tetrahedron(p[1], p[2], p[3], vp[0]);
    tvp[1] = Tetrahedron(p[1], p[2], p[3], vp[1]);

    bool flip_flag = false;

    /*
     * if(t[0].volume()*tvp[0].volume() < NT(0) &&
     *    t[1].volume()*tvp[1].volume() < NT(0) )
     */

    if (CGAL::is_negative(t[0].volume() * tvp[0].volume()) &&
        CGAL::is_negative(t[1].volume() * tvp[1].volume())) {
      c[0]->set_dirty(true);
      c[1]->set_dirty(true);
      flip_flag = true;
    }

    /*
     * if(t[0].volume()*tvp[0].volume() == NT(0) &&
     *    t[1].volume()*tvp[1].volume() < NT(0) )
     */

    if (CGAL::is_zero(t[0].volume() * tvp[0].volume()) &&
        CGAL::is_negative(t[1].volume() * tvp[1].volume())) {
      c[0]->set_dirty(true);
      c[1]->set_dirty(true);
      flip_flag = true;
    }

    /*
     * if(t[0].volume()*tvp[0].volume() < NT(0) &&
     *    t[1].volume()*tvp[1].volume() == NT(0) )
     */
    if (CGAL::is_negative(t[0].volume() * tvp[0].volume()) &&
        CGAL::is_zero(t[1].volume() * tvp[1].volume())) {
      c[0]->set_dirty(true);
      c[1]->set_dirty(true);
      flip_flag = true;
    }

    c[0]->set_flip(id, flip_flag);
    c[1]->set_flip(c[1]->index(c[0]), flip_flag);

    // assume if flip then cospherical.
    if (flip_flag) {
      c[0]->set_cosph_pair(id, true);
      c[1]->set_cosph_pair(c[1]->index(c[0]), true);

      is_there_atleast_one_violation_of_del_vor_property = true;
    }
  }

  return is_there_atleast_one_violation_of_del_vor_property;
}

// ------------------------------------------------------
// identify_cospherical_neighbor
// -----------------------------
// Identify the pair of tetrahedra which are cospherical
// ------------------------------------------------------
bool identify_cospherical_neighbor(Triangulation &triang) {

  bool is_there_atleast_one_pair_of_cospherical_tetrahedra = false;

  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); fit++) {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;

    // if one of the cells is infinite the two cells
    // cannot be co-spherical.
    if (triang.is_infinite(ch) || triang.is_infinite(ch->neighbor(id)))
      continue;

    if (triang.side_of_sphere(
            ch,
            ch->neighbor(id)->vertex(ch->neighbor(id)->index(ch))->point()) ==
        CGAL::ON_BOUNDARY) {
      CGAL_assertion(
          triang.side_of_sphere(ch->neighbor(id), ch->vertex(id)->point()) ==
          CGAL::ON_BOUNDARY);
      ch->set_cosph_pair(id, true);
      ch->neighbor(id)->set_cosph_pair(ch->neighbor(id)->index(ch), true);

      is_there_atleast_one_pair_of_cospherical_tetrahedra = true;

      continue;
    }
    if (triang.side_of_sphere(
            ch,
            ch->neighbor(id)->vertex(ch->neighbor(id)->index(ch))->point()) ==
        CGAL::ON_UNBOUNDED_SIDE) {
      CGAL_assertion(
          triang.side_of_sphere(ch->neighbor(id), ch->vertex(id)->point()) ==
          CGAL::ON_UNBOUNDED_SIDE);
    } else
      cerr << "Empty Sphere property is violated ";
  }

  return is_there_atleast_one_pair_of_cospherical_tetrahedra;
}

// ------------------------------------------------------
// cluster_cospherical_tetrahedra
// ------------------------------
// Cluster the tetrahedra which are cospherical.
// ------------------------------------------------------
void cluster_cospherical_tetrahedra(Triangulation &triang) {
  // after identifying pairwise the cospherical tetrahedra
  // we need to identify the clusters of cospherical tetrahedra.
  // for each cluster there will be one tetrahedron which is
  // the leader of the cluster.
  for (FCI cit = triang.finite_cells_begin();
       cit != triang.finite_cells_end(); cit++)
    cit->visited = false;
  for (FCI cit = triang.finite_cells_begin();
       cit != triang.finite_cells_end(); cit++) {
    if (cit->visited)
      continue;
    if (cit->dirty())
      continue;

    vector<Cell_handle> cosph_stack;

    if (cit->cosph_pair(0) || cit->cosph_pair(1) || cit->cosph_pair(2) ||
        cit->cosph_pair(3)) {
      cosph_stack.push_back(cit);
      cit->set_cosph_leader(true);
      cit->set_cosph_leader_id(cit->id);
      cit->visited = true;
    }

    while (!cosph_stack.empty()) {
      Cell_handle ch = cosph_stack.back();
      cosph_stack.pop_back();

      // find out if any neighbor is cospherical
      // with this one.
      for (int i = 0; i < 4; i++) {
        if (ch->neighbor(i)->visited)
          continue;
        if (ch->cosph_pair(i)) {
          ch->neighbor(i)->set_cosph_leader_id(cit->id);
          ch->neighbor(i)->visited = true;
          cosph_stack.push_back(ch->neighbor(i));
        }
      }
    }
  }
  for (FCI cit = triang.finite_cells_begin();
       cit != triang.finite_cells_end(); cit++) {
    if (!cit->cosph_pair(0) && !cit->cosph_pair(1) && !cit->cosph_pair(2) &&
        !cit->cosph_pair(3))
      continue;
    for (int i = 0; i < 4; i++) {
      if (cit->cosph_pair(i)) {
        CGAL_assertion(cit->cosph_leader_id() ==
                       cit->neighbor(i)->cosph_leader_id());
      }
    }
  }
}

// ----------------------------------------------------------
// end of the functions to circumvent the problems with CGAL
// ----------------------------------------------------------

/*

// ------------------------------------------------------
// find_flow_direction
// ------------------------
// Given a tetrahedralization find out the direction of flow
// on each voronoi edge
// ------------------------------------------------------
void
find_flow_direction(Triangulation &triang )

*/

// ------------------------------------------------------
// identify_sink_and_saddle
// ------------------------
// Given a tetrahedralization finds out which one is sink
// and which is not
// ------------------------------------------------------
void identify_sink_and_saddle(Triangulation &triang) {

  // We need the visited field. So reset it.
  for (FCI cit = triang.finite_cells_begin();
       cit != triang.finite_cells_end(); cit++) {
    cit->visited = false;
    CGAL_assertion(!cit->sink());
  }

  // Iterate over all the cells and flag the sinks.
  for (FCI cit = triang.finite_cells_begin();
       cit != triang.finite_cells_end(); cit++) {
    if (cit->dirty())
      continue;
    CGAL_assertion(!cit->sink());

    if (cit->cosph_pair(0) || cit->cosph_pair(1) || cit->cosph_pair(2) ||
        cit->cosph_pair(3)) {
      if (!cit->cosph_leader())
        continue;

      // this tetrahedron is a leader in a cluster
      // of cospherical tetrahedron.

      // to see if this cluster is a sink we have to walk
      // locally to check the flow on all the voronoi edges
      // in the cluster.
      vector<Cell_handle> cosph_stack;
      cosph_stack.push_back(cit);
      cit->visited = true;

      bool is_sink = true;

      while (!cosph_stack.empty()) {
        Cell_handle ch = cosph_stack.back();
        cosph_stack.pop_back();

        for (int i = 0; i < 4; i++) {
          if (ch->cosph_pair(i)) {
            CGAL_assertion(ch->cosph_leader_id() ==
                           ch->neighbor(i)->cosph_leader_id());
            CGAL_assertion(ch->cosph_leader_id() == cit->id);
            if (!ch->neighbor(i)->visited) {
              cosph_stack.push_back(ch->neighbor(i));
              ch->neighbor(i)->visited = true;
            }
          } else {
            if (ch->source(i)) {
              CGAL_assertion(
                  ch->neighbor(i)->terminus(ch->neighbor(i)->index(ch)));
              is_sink = false;
            } else {
              CGAL_assertion(ch->terminus(i));
            }
          }
        }
      }

      cit->set_sink(is_sink);

      continue;
    }

    cit->set_sink(true);

    for (int i = 0; i < 4; i++) {
      if (cit->source(i))
        cit->set_sink(false);
      if (cit->terminus(i) &&
          cit->neighbor(i)->terminus(cit->neighbor(i)->index(cit))) {
        cit->set_saddle(i, true);
        cit->neighbor(i)->set_saddle(cit->neighbor(i)->index(cit), true);
      }
    }
    if (cit->source(0) + cit->source(1) + cit->source(2) + cit->source(3) >
        2) {
      cerr << "error in flow direction :: source for more than 2 " << endl;
    }
    if (cit->sink()) {
      CGAL_assertion(!cit->dirty());
      CGAL_assertion(cit->cell_radius() > 0);
      CGAL_assertion(!cit->source(0) && !cit->source(1) && !cit->source(2) &&
                     !cit->source(3));
      for (int i = 0; i < 4; i++) {
        if (!cit->terminus(i)) {
          CGAL_assertion(cit->cosph_pair(i));
          CGAL_assertion(cit->cosph_leader());
        }
      }
    }
  }
}

// -----------------------------------------------------------------------
// Some small functions
// -----------------------------------------------------------------------

// Compute the cosine of the smaller of the two angles
// made by the vectors v and w
double cosine(const Vector &v, const Vector &w) {
  return CGAL::to_double(v * w) /
         sqrt(CGAL::to_double(v * v) * CGAL::to_double(w * w));
}

// Find the index of the third vertex for a facet where this vertex
// is neither *v nor *w.
int find_third_vertex_index(const Facet &f, Vertex_handle v,
                            Vertex_handle w) {
  int id = f.second;
  for (int i = 1; i <= 3; ++i) {
    if (f.first->vertex((id + i) % 4) != v &&
        f.first->vertex((id + i) % 4) != w)
      return (id + i) % 4;
  }
  return -1;
}

// Compute the index of an edge. The index of an edge in a facet
// is defined as the index of a facet in cell, i.e. it is the index
// of the opposite vertex. The arguments are the facet index of the
// facet witth respect to the cell and the indices of the vertices
// incident to the edge also with respect to the cell.
int edge_index(const int facet_index, const int first_vertex_index,
               const int second_vertex_index) {
  return 6 - facet_index - first_vertex_index - second_vertex_index;
}

// Compute the indices of the vertices incident to an edge.
void vertex_indices(const int facet_index, const int edge_index,
                    int &first_vertex, int &second_vertex) {
  if ((facet_index == 0 && edge_index == 1) ||
      (facet_index == 1 && edge_index == 0)) {
    first_vertex = 2;
    second_vertex = 3;
  } else if ((facet_index == 0 && edge_index == 2) ||
             (facet_index == 2 && edge_index == 0)) {
    first_vertex = 1;
    second_vertex = 3;
  } else if ((facet_index == 0 && edge_index == 3) ||
             (facet_index == 3 && edge_index == 0)) {
    first_vertex = 1;
    second_vertex = 2;
  } else if ((facet_index == 1 && edge_index == 2) ||
             (facet_index == 2 && edge_index == 1)) {
    first_vertex = 0;
    second_vertex = 3;
  } else if ((facet_index == 1 && edge_index == 3) ||
             (facet_index == 3 && edge_index == 1)) {
    first_vertex = 0;
    second_vertex = 2;
  } else if ((facet_index == 2 && edge_index == 3) ||
             (facet_index == 3 && edge_index == 2)) {
    first_vertex = 0;
    second_vertex = 1;
  }
}

// ------------------------
// end of small routines
// ------------------------

}; // namespace SuperSecondaryStructures
