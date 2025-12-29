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

#include <SecondaryStructures/util.h>

using namespace SecondaryStructures;

double my_drand() { return ((double)(rand() % 10000)) / 10000.0; }

// cosine
double cosine(const Vector &v, const Vector &w) {
  return CGAL::to_double(v * w) /
         sqrt(CGAL::to_double(v * v) * CGAL::to_double(w * w));
}

// arand - eliminated some redundant functions...

// normalize
void normalize(Vector &v) {
  v = (1. / (sqrt(CGAL::to_double(v * v)))) * v;
  return;
}

// check if the angle <p0,p2,p1 > 90 degree
bool is_obtuse(const Point &p0, const Point &p1, const Point &p2) {
  Vector v0 = p0 - p2;
  Vector v1 = p1 - p2;
  return (CGAL::to_double(v0 * v1) < 0);
}

// length_of_seg
double length_of_seg(const Segment &s) {
  return sqrt(
      CGAL::to_double((s.point(0) - s.point(1)) * (s.point(0) - s.point(1))));
}

bool are_neighbors(const Cell_handle c0, const Cell_handle c1) {
  int uid[4], vid[4];
  for (int i = 0; i < 4; i++) {
    uid[i] = c0->vertex(i)->id;
    vid[i] = c1->vertex(i)->id;
  }
  int cnt = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (uid[i] == vid[j]) {
        cnt++;
      }
  if (cnt == 3) {
    c0->index(c1);
    return true;
  }
  return false;
}

// Find the index of the third vertex for a facet where this vertex is neither
// *v nor *w.
int find_third_vertex_index(const Facet &f, Vertex_handle v,
                            Vertex_handle w) {
  int id = f.second;
  for (int i = 1; i <= 3; ++i) {
    if (f.first->vertex((id + i) % 4) != v &&
        f.first->vertex((id + i) % 4) != w) {
      return (id + i) % 4;
    }
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

// ---------------------------------------
// vertex_indices
// --------------
// Compute the indices of the vertices
// incident to an edge. Given, facet_id
// and edge_id.
// ---------------------------------------
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

void transform(vector<Point> &target, const vector<Vector> &source,
               const Vector &t, const Point &o, const Vector &a,
               const double &c, const double &s) {
  // clear the target.
  target.clear();
  // set the trandformation matrix.
  double ax = CGAL::to_double(a.x()), ay = CGAL::to_double(a.y()),
         az = CGAL::to_double(a.z());
  double tx = CGAL::to_double(t.x()), ty = CGAL::to_double(t.y()),
         tz = CGAL::to_double(t.z());
  double trnsf_mat_1[4][4] = {
      {ax * ax * (1 - c) + c, ax * ay * (1 - c) - az * s,
       ax * az * (1 - c) + ay * s, tx},
      {ay * ax * (1 - c) + az * s, ay * ay * (1 - c) + c,
       ay * az * (1 - c) - ax * s, ty},
      {ax * az * (1 - c) - ay * s, ay * az * (1 - c) + ax * s,
       az * az * (1 - c) + c, tz},
      {0, 0, 0, 1}};
  for (int j = 0; j < (int)source.size(); j++) {
    double px = CGAL::to_double(source[j].x());
    double py = CGAL::to_double(source[j].y());
    double pz = CGAL::to_double(source[j].z());
    double _px = px * trnsf_mat_1[0][0] + py * trnsf_mat_1[0][1] +
                 pz * trnsf_mat_1[0][2] + trnsf_mat_1[0][3];
    double _py = px * trnsf_mat_1[1][0] + py * trnsf_mat_1[1][1] +
                 pz * trnsf_mat_1[1][2] + trnsf_mat_1[1][3];
    double _pz = px * trnsf_mat_1[2][0] + py * trnsf_mat_1[2][1] +
                 pz * trnsf_mat_1[2][2] + trnsf_mat_1[2][3];
    target.push_back(Point(_px, _py, _pz));
  }
  return;
}

// Given a ray and two points a and b find out if a and
// b is in the same side of the ray.
// Assumption: The ray, a and b are in the same plane.
// [*] Instead of the Ray we will work on a pair of points
// (p0, p1) where p1 is dummy and can be shifted. [*]
bool is_same_side_of_ray(const Point &p0, const Point &p1, const Point &a,
                         const Point &b) {
  return (CGAL::to_double(CGAL::cross_product(p1 - p0, a - p1) *
                          CGAL::cross_product(p1 - p0, b - p1)) >= 0);
}

// Given two rays and a segment find out if the segment
// is fully/partially contained in the infinite triangle.
// Parameters: ray 1 = (p0, p1), ray 2 = (p0, p2).
// Both p1 and p2 are just two points on the rays.
//             segment = (a, b)
// [*] Note: a or b can coincide with p1 or p2.
// coincidence_vector records that information.
// coincidence_vector[0] = {0,1,2} - matches with none, p1, p2.
// coincidence_vector[1] = {0,1,2} - matches with none, p1, p2.
//             two booleans co0 and co1 indicate that.
// Result will be stored in contained[].
// contained[0] says if a is within the fan.
// contained[1] says if b is within the fan.
// If both entries true the segment is fully contained.
// If both entries false the segment is fully outside.
// Otherwise it's partially contained.
void is_contained_in_inf_tr(const Point &p0, const Point &p1, const Point &p2,
                            const Point &a, const Point &b,
                            const vector<int> &coincidence_vector,
                            bool *contained) {
  if (coincidence_vector[0] == 0 && coincidence_vector[1] == 0) {
    // point a is within the opening (or visible by p0) if
    // it is in the same side of ray1 (p0, p1) as p2 is and
    // it is in the same side of ray2 (p0, p2) as p1 is.
    contained[0] = is_same_side_of_ray(p0, p1, p2, a) &&
                   is_same_side_of_ray(p0, p2, p1, a);
    // same for b.
    contained[1] = is_same_side_of_ray(p0, p1, p2, b) &&
                   is_same_side_of_ray(p0, p2, p1, b);
    return;
  } else {
    // point a = p1 and b doesn't match.
    if (coincidence_vector[0] == 1 && coincidence_vector[1] == 0) {
      if (!is_same_side_of_ray(p0, p1, p2, b)) {
        contained[0] = contained[1] = false;
      } else {
        contained[0] = true;
        if (is_same_side_of_ray(p0, p2, p1, b)) {
          contained[1] = true;
        } else {
          contained[1] = false;
        }
      }
      return;
    }
    // point a = p2 and b doesn't match.
    if (coincidence_vector[0] == 2 && coincidence_vector[1] == 0) {
      if (!is_same_side_of_ray(p0, p2, p1, b)) {
        contained[0] = contained[1] = false;
      } else {
        contained[0] = true;
        if (is_same_side_of_ray(p0, p1, p2, b)) {
          contained[1] = true;
        } else {
          contained[1] = false;
        }
      }
      return;
    }
    // point b = p1 and a doesn't match.
    if (coincidence_vector[0] == 0 && coincidence_vector[1] == 1) {
      if (!is_same_side_of_ray(p0, p1, p2, a)) {
        contained[0] = contained[1] = false;
      } else {
        contained[1] = true;
        if (is_same_side_of_ray(p0, p2, p1, a)) {
          contained[0] = true;
        } else {
          contained[0] = false;
        }
      }
      return;
    }
    // point b = p2 and a doesn't match.
    if (coincidence_vector[0] == 0 && coincidence_vector[1] == 2) {
      if (!is_same_side_of_ray(p0, p2, p1, a)) {
        contained[0] = contained[1] = false;
      } else {
        contained[1] = true;
        if (is_same_side_of_ray(p0, p1, p2, a)) {
          contained[0] = true;
        } else {
          contained[0] = false;
        }
      }
      return;
    }
    // otherwise the segment(a,b) coincides with the segment(p1,p2).
    // the segment must be contained in the inf_tr (p0, p1, p2).
    contained[0] = true;
    contained[1] = true;
    return;
  }
  return;
}

// is_outside_bounding_box
bool is_outside_bounding_box(const Point &p,
                             const vector<double> &bounding_box) {
  if (CGAL::to_double(p.x()) < bounding_box[0] ||
      CGAL::to_double(p.x()) > bounding_box[1] ||
      CGAL::to_double(p.y()) < bounding_box[2] ||
      CGAL::to_double(p.y()) > bounding_box[3] ||
      CGAL::to_double(p.z()) < bounding_box[4] ||
      CGAL::to_double(p.z()) > bounding_box[5]) {
    return true;
  }
  return false;
}

// is_outside_bounding_box
bool is_outside_bounding_box(const vector<Point> &points,
                             const vector<double> &bounding_box) {
  for (int i = 0; i < (int)points.size(); i++) {
    Point p = points[i];
    if (is_outside_bounding_box(p, bounding_box)) {
      return true;
    }
  }
  return false;
}

// is_VF_outside_bounding_box
bool is_VF_outside_bounding_box(const Triangulation &triang, const Edge &e,
                                const vector<double> &bounding_box) {
  Facet_circulator fcirc = triang.incident_facets(e);
  Facet_circulator begin = fcirc;
  do {
    if (is_outside_bounding_box((*fcirc).first->voronoi(), bounding_box)) {
      return true;
    }
    fcirc++;
  } while (fcirc != begin);
  return false;
}

bool is_outside_VF(const Triangulation &triang, const Edge &e) {
  Facet_circulator fcirc = triang.incident_facets(e);
  Facet_circulator begin = fcirc;
  do {
    if (!(*fcirc).first->outside) {
      return false;
    }
    fcirc++;
  } while (fcirc != begin);
  return true;
}

bool is_inside_VF(const Triangulation &triang, const Edge &e) {
  Facet_circulator fcirc = triang.incident_facets(e);
  Facet_circulator begin = fcirc;
  do {
    if ((*fcirc).first->outside) {
      return false;
    }
    fcirc++;
  } while (fcirc != begin);
  return true;
}

bool is_surf_VF(const Triangulation &triang, const Edge &e) {
  Facet_circulator fcirc = triang.incident_facets(e);
  Facet_circulator begin = fcirc;
  bool in = false, out = false;
  do {
    Cell_handle cur_c = (*fcirc).first;
    if (cur_c->outside) {
      out = true;
    } else {
      in = true;
    }
    fcirc++;
  } while (fcirc != begin);
  return (in && out);
}

bool is_surf_VF(const Triangulation &triang, const Cell_handle &c,
                const int uid, const int vid) {
  Facet_circulator fcirc = triang.incident_facets(Edge(c, uid, vid));
  Facet_circulator begin = fcirc;
  do {
    Cell_handle cur_c = (*fcirc).first;
    int cur_fid = (*fcirc).second;
    if (cur_c->cocone_flag(cur_fid)) {
      return true;
    }
    fcirc++;
  } while (fcirc != begin);
  return false;
}

bool is_inf_VF(const Triangulation &triang, const Cell_handle &c,
               const int uid, const int vid) {
  Facet_circulator fcirc = triang.incident_facets(Edge(c, uid, vid));
  Facet_circulator begin = fcirc;
  do {
    Cell_handle cur_c = (*fcirc).first;
    if (triang.is_infinite(cur_c)) {
      return true;
    }
    fcirc++;
  } while (fcirc != begin);
  return false;
}

/*
void mark_VF_on_u1(Triangulation& triang, Cell_handle& c, int uid, int vid)
{

   Facet_circulator fcirc = triang.incident_facets(Edge(c,uid,vid));
   Facet_circulator begin = fcirc;
   do{
      Cell_handle cur_c = (*fcirc).first;
      int cur_fid = (*fcirc).second;
      // marking: VF is on um(i1).
      int cur_uid = -1, cur_vid = -1;
      for(int k = 1; k < 4; k ++)
      {
         if(cur_c->vertex((cur_fid+k)%4)->id != c->vertex(uid)->id &&
            cur_c->vertex((cur_fid+k)%4)->id != c->vertex(vid)->id )
         {
            vertex_indices(cur_fid, (cur_fid+k)%4, cur_uid, cur_vid);
            break;
         }
      }
      CGAL_assertion(cur_uid != -1 && cur_vid != -1 && cur_uid != cur_vid);
      CGAL_assertion((c->vertex(uid)->id == cur_c->vertex(cur_uid)->id &&
                      c->vertex(vid)->id == cur_c->vertex(cur_vid)->id) ||
                     (c->vertex(uid)->id == cur_c->vertex(cur_vid)->id &&
                      c->vertex(vid)->id == cur_c->vertex(cur_uid)->id) );
      cur_c->set_VF_on_um_i1(cur_uid, cur_vid, true);
      CGAL_assertion(cur_c->VV_on_um_i1());
      fcirc ++;
   } while(fcirc != begin);
}

void
mark_VF_visited(Triangulation& triang,
               Cell_handle& c, int uid, int vid)
{
   Facet_circulator fcirc = triang.incident_facets(Edge(c, uid, vid));
   Facet_circulator begin = fcirc;

   do{
      Cell_handle cur_c = (*fcirc).first;
      int cur_fid = (*fcirc).second;
      // mark it visited.
      int cur_uid = -1, cur_vid = -1;
      for(int k = 1; k < 4; k ++)
      {
         if(cur_c->vertex((cur_fid+k)%4)->id != c->vertex(uid)->id &&
            cur_c->vertex((cur_fid+k)%4)->id != c->vertex(vid)->id )
         {
            vertex_indices(cur_fid, (cur_fid+k)%4, cur_uid, cur_vid);
            break;
         }
      }
      CGAL_assertion(cur_uid != -1 && cur_vid != -1 && cur_uid != cur_vid);
      CGAL_assertion((c->vertex(uid)->id == cur_c->vertex(cur_uid)->id &&
                      c->vertex(vid)->id == cur_c->vertex(cur_vid)->id) ||
                     (c->vertex(uid)->id == cur_c->vertex(cur_vid)->id &&
                      c->vertex(vid)->id == cur_c->vertex(cur_uid)->id) );
      cur_c->e_visited[cur_uid][cur_vid] = true;
      cur_c->e_visited[cur_vid][cur_uid] = true;
      fcirc ++;
   } while(fcirc != begin);
}

// Set the patch id to VF dual to
// Edge(c,uid,vid).
void set_patch_id(Triangulation& triang, Cell_handle& c, int uid, int vid,
const int& id)
{
   Facet_circulator fcirc = triang.incident_facets(Edge(c, uid, vid));
   Facet_circulator begin = fcirc;

   do{
      Cell_handle cur_c = (*fcirc).first;
      int cur_fid = (*fcirc).second;
      // set id.
      int cur_uid = -1, cur_vid = -1;
      for(int k = 1; k < 4; k ++)
      {
         if(cur_c->vertex((cur_fid+k)%4)->id != c->vertex(uid)->id &&
            cur_c->vertex((cur_fid+k)%4)->id != c->vertex(vid)->id )
         {
            vertex_indices(cur_fid, (cur_fid+k)%4, cur_uid, cur_vid);
            break;
         }
      }
      CGAL_assertion(cur_uid != -1 && cur_vid != -1 && cur_uid != cur_vid);
      CGAL_assertion((c->vertex(uid)->id == cur_c->vertex(cur_uid)->id &&
                      c->vertex(vid)->id == cur_c->vertex(cur_vid)->id) ||
                     (c->vertex(uid)->id == cur_c->vertex(cur_vid)->id &&
                      c->vertex(vid)->id == cur_c->vertex(cur_uid)->id) );
      cur_c->patch_id[cur_uid][cur_vid] = id;
      cur_c->patch_id[cur_vid][cur_uid] = id;
      fcirc ++;
   } while(fcirc != begin);
}*/

bool is_cospherical_pair(const Triangulation &triang, const Facet &f) {
  Cell_handle c[2];
  int id[2];
  c[0] = f.first;
  id[0] = f.second;
  c[1] = c[0]->neighbor(id[0]);
  id[1] = c[1]->index(c[0]);
  if (triang.side_of_sphere(c[0], c[1]->vertex(id[1])->point()) ==
      CGAL::ON_BOUNDARY) {
    CGAL_assertion(
        triang.side_of_sphere(c[1], c[0]->vertex(id[0])->point()) ==
        CGAL::ON_BOUNDARY);
    return true;
  }
  return false;
}

// Identify the pair of tetrahedra which are cospherical
bool identify_cospherical_neighbor(Triangulation &triang) {
  bool is_there_atleast_one_pair_of_cospherical_tetrahedra = false;
  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); fit++) {
    Cell_handle c[2];
    int id[2];
    c[0] = fit->first;
    id[0] = fit->second;
    c[1] = c[0]->neighbor(id[0]);
    id[1] = c[1]->index(c[0]);
    // if one of the cells is infinite the two cells
    // cannot be co-spherical.
    if (triang.is_infinite(c[0]) || triang.is_infinite(c[1])) {
      continue;
    }
    if (is_cospherical_pair(triang, (*fit))) {
      // record it in the cells.
      c[0]->set_cosph_pair(id[0], true);
      c[1]->set_cosph_pair(id[1], true);
      c[0]->c_cosph = true;
      c[1]->c_cosph = true;
      is_there_atleast_one_pair_of_cospherical_tetrahedra = true;
    } else if (triang.side_of_sphere(c[0], c[1]->vertex(id[1])->point()) ==
               CGAL::ON_UNBOUNDED_SIDE)
      CGAL_assertion(
          triang.side_of_sphere(c[1], c[0]->vertex(id[0])->point()) ==
          CGAL::ON_UNBOUNDED_SIDE);
    else {
      cerr << "Empty Sphere property is violated ";
    }
  }
  return is_there_atleast_one_pair_of_cospherical_tetrahedra;
}

bool is_there_any_common_element(const vector<int> &vec1,
                                 const vector<int> &vec2) {
  for (int i = 0; i < (int)vec1.size(); i++)
    for (int j = 0; j < (int)vec2.size(); j++)
      if (vec1[i] == vec2[j]) {
        return true;
      }
  return false;
}
