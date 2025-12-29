/*
  Copyright 2008-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <cvcraw_geometry/cvcraw_geometry.h>
#include <list>
#include <map>
// #include <algorithm>
// #include <stdint.h>

namespace cvcraw_geometry {
geometry_t::geometry_t() {
  for (int i = 0; i < 3; i++) {
    min_ext[i] = 0.0;
    max_ext[i] = 0.0;
  }
}

geometry_t::geometry_t(const geometry_t &copy)
    : points(copy.points), boundary(copy.boundary), normals(copy.normals),
      colors(copy.colors), lines(copy.lines), tris(copy.tris),
      quads(copy.quads), min_ext(copy.min_ext), max_ext(copy.max_ext) {}

geometry_t::~geometry_t() {}

geometry_t &geometry_t::operator=(const geometry_t &copy) {
  points = copy.points;
  boundary = copy.boundary;
  normals = copy.normals;
  colors = copy.colors;
  lines = copy.lines;
  tris = copy.tris;
  quads = copy.quads;
  min_ext = copy.min_ext;
  max_ext = copy.max_ext;
  return *this;
}

bool geometry_t::empty() const {
  return points.empty() && boundary.empty() && normals.empty() &&
         colors.empty() && lines.empty() && tris.empty() && quads.empty();
}

geometry_t &geometry_t::calculate_extents() {
  using namespace std;

  if (points.empty()) {
    fill(min_ext.begin(), min_ext.end(), 0.0);
    fill(max_ext.begin(), max_ext.end(), 0.0);
    return *this;
  }

  min_ext = max_ext = points[0];
  for (points_t::const_iterator i = points.begin(); i != points.end(); i++) {
    if (min_ext[0] > (*i)[0])
      min_ext[0] = (*i)[0];
    if (min_ext[1] > (*i)[1])
      min_ext[1] = (*i)[1];
    if (min_ext[2] > (*i)[2])
      min_ext[2] = (*i)[2];
    if (max_ext[0] < (*i)[0])
      max_ext[0] = (*i)[0];
    if (max_ext[1] < (*i)[1])
      max_ext[1] = (*i)[1];
    if (max_ext[2] < (*i)[2])
      max_ext[2] = (*i)[2];
  }

  return *this;
}

geometry_t &geometry_t::merge(const geometry_t &geom) {
  using namespace std;

  // append vertex info
  points.insert(points.end(), geom.points.begin(), geom.points.end());
  normals.insert(normals.end(), geom.normals.begin(), geom.normals.end());
  colors.insert(colors.end(), geom.colors.begin(), geom.colors.end());

  // we apparently don't have normal iterators for boundary_t :/
  unsigned int old_size = boundary.size();
  boundary.resize(boundary.size() + geom.boundary.size());
  for (unsigned int i = 0; i < geom.boundary.size(); i++)
    boundary[old_size + i] = geom.boundary[i];

  // append and modify index info
  lines.insert(lines.end(), geom.lines.begin(), geom.lines.end());
  {
    lines_t::iterator i = lines.begin();
    advance(i, lines.size() - geom.lines.size());
    while (i != lines.end()) {
      for (line_t::iterator j = i->begin(); j != i->end(); j++)
        *j += points.size() - geom.points.size();
      i++;
    }
  }

  tris.insert(tris.end(), geom.tris.begin(), geom.tris.end());
  {
    triangles_t::iterator i = tris.begin();
    advance(i, tris.size() - geom.tris.size());
    while (i != tris.end()) {
      for (triangle_t::iterator j = i->begin(); j != i->end(); j++)
        *j += points.size() - geom.points.size();
      i++;
    }
  }

  quads.insert(quads.end(), geom.quads.begin(), geom.quads.end());
  {
    quads_t::iterator i = quads.begin();
    advance(i, quads.size() - geom.quads.size());
    while (i != quads.end()) {
      for (quad_t::iterator j = i->begin(); j != i->end(); j++)
        *j += points.size() - geom.points.size();
      i++;
    }
  }

  calculate_extents();
  return *this;
}

geometry_t geometry_t::tri_surface() const {
  geometry_t new_geom;

  new_geom.points = points;
  new_geom.colors = colors;

  // if we don't have boundary info, just insert the tris directly
  if (boundary.size() != points.size()) {
    for (triangles_t::const_iterator i = tris.begin(); i != tris.end(); i++)
      new_geom.tris.push_back(*i);

    for (quads_t::const_iterator i = quads.begin(); i != quads.end(); i++) {
      triangle_t quad_tris[2];
      quad_tris[0][0] = (*i)[0];
      quad_tris[0][1] = (*i)[1];
      quad_tris[0][2] = (*i)[3];

      quad_tris[1][0] = (*i)[1];
      quad_tris[1][1] = (*i)[2];
      quad_tris[1][2] = (*i)[3];

      new_geom.tris.push_back(quad_tris[0]);
      new_geom.tris.push_back(quad_tris[1]);
    }
  } else {
    for (triangles_t::const_iterator i = tris.begin(); i != tris.end(); i++)
      if (boundary[(*i)[0]] && boundary[(*i)[1]] && boundary[(*i)[2]])
        new_geom.tris.push_back(*i);

    for (quads_t::const_iterator i = quads.begin(); i != quads.end(); i++)
      if (boundary[(*i)[0]] && boundary[(*i)[1]] && boundary[(*i)[2]] &&
          boundary[(*i)[3]]) {
        triangle_t quad_tris[2];
        quad_tris[0][0] = (*i)[0];
        quad_tris[0][1] = (*i)[1];
        quad_tris[0][2] = (*i)[3];

        quad_tris[1][0] = (*i)[1];
        quad_tris[1][1] = (*i)[2];
        quad_tris[1][2] = (*i)[3];

        new_geom.tris.push_back(quad_tris[0]);
        new_geom.tris.push_back(quad_tris[1]);
      }
  }

  return new_geom;
}

geometry_t &geometry_t::calculate_surf_normals() {
  using namespace std;
  geometry_t tri_geom = this->tri_surface();

  // find all the triangles that each point is a part of
  map<point_t, list<unsigned int>> neighbor_tris;
  for (triangles_t::iterator i = tri_geom.tris.begin();
       i != tri_geom.tris.end(); i++)
    for (triangle_t::iterator j = i->begin(); j != i->end(); j++)
      neighbor_tris[tri_geom.points[*j]].push_back(
          distance(tri_geom.tris.begin(), i));

  // now iterate over our points, and for each point push back a new normal
  // vector
  normals.clear();
  for (points_t::iterator i = points.begin(); i != points.end(); i++) {
    normal_t norm = {{0.0, 0.0, 0.0}};
    list<unsigned int> &neighbors = neighbor_tris[*i];
    for (list<unsigned int>::iterator j = neighbors.begin();
         j != neighbors.end(); j++) {
      normal_t cur_tri_norm;
      triangle_t &tri = tri_geom.tris[*j];
      point_t p[3] = {tri_geom.points[tri[0]], tri_geom.points[tri[1]],
                      tri_geom.points[tri[2]]};
      vector_t v1 = {
          {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]}};
      vector_t v2 = {
          {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]}};
      utility::cross(cur_tri_norm, v1, v2);
      utility::normalize(cur_tri_norm);
      for (int k = 0; k < 3; k++)
        norm[k] += cur_tri_norm[k];
    }
    if (!neighbors.empty())
      for (int j = 0; j < 3; j++)
        norm[j] /= neighbors.size();
    utility::normalize(norm);
    normals.push_back(norm);
  }

  return *this;
}

geometry_t &geometry_t::calculate_surf_normals_angle_weight() {
  using namespace std;
  geometry_t tri_geom = this->tri_surface();

  // find all the triangles that each point is a part of
  map<point_t, list<unsigned int>> neighbor_tris;
  for (triangles_t::iterator i = tri_geom.tris.begin();
       i != tri_geom.tris.end(); i++)
    for (triangle_t::iterator j = i->begin(); j != i->end(); j++)
      neighbor_tris[tri_geom.points[*j]].push_back(
          distance(tri_geom.tris.begin(), i));

  // now iterate over our points, and for each point push back a new normal
  // vector
  normals.clear();
  for (points_t::iterator i = points.begin(); i != points.end(); i++) {
    normal_t norm = {{0.0, 0.0, 0.0}};
    list<unsigned int> &neighbors = neighbor_tris[*i];
    for (list<unsigned int>::iterator j = neighbors.begin();
         j != neighbors.end(); j++) {
      normal_t cur_tri_norm;
      triangle_t &tri = tri_geom.tris[*j];
      point_t p[3] = {tri_geom.points[tri[0]], tri_geom.points[tri[1]],
                      tri_geom.points[tri[2]]};
      vector_t v1 = {
          {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]}};
      vector_t v2 = {
          {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]}};
      vector_t v3 = {
          {p[2][0] - p[1][0], p[2][1] - p[1][1], p[2][2] - p[1][2]}};
      double a, b, c, alpha;
      a = utility::dot(v1, v1);
      b = utility::dot(v2, v2);
      c = utility::dot(v3, v3);
      alpha = acos((a + b - c) / (2 * sqrt(a * b)));

      utility::cross(cur_tri_norm, v1, v2);
      utility::normalize(cur_tri_norm);
      for (int k = 0; k < 3; k++)
        norm[k] += alpha * cur_tri_norm[k];
    }
    if (!neighbors.empty())
      //  for(int j = 0; j < 3; j++)
      //   norm[j] /= neighbors.size();
      utility::normalize(norm);
    normals.push_back(norm);
  }

  return *this;
}

geometry_t geometry_t::generate_wire_interior() const {
  geometry_t new_geom(*this);

  if (!boundary.empty() && boundary.size() == points.size()) {
    // tet mesh
    if (!tris.empty()) {
      new_geom.tris.clear();
      for (triangles_t::const_iterator j = tris.begin(); j != tris.end();
           j++) {
        // add the tri face boundary lines if both verts
        // of the potential line to be added do not lie
        // on the boundary
        if (!boundary[(*j)[0]] || !boundary[(*j)[1]]) {
          line_t line = {{(*j)[0], (*j)[1]}};
          new_geom.lines.push_back(line);
        }
        if (!boundary[(*j)[1]] || !boundary[(*j)[2]]) {
          line_t line = {{(*j)[1], (*j)[2]}};
          new_geom.lines.push_back(line);
        }
        if (!boundary[(*j)[2]] || !boundary[(*j)[0]]) {
          line_t line = {{(*j)[2], (*j)[0]}};
          new_geom.lines.push_back(line);
        }

        // add the triangle to new_geom if all verts are on the boundary
        if (boundary[(*j)[0]] && boundary[(*j)[1]] && boundary[(*j)[2]])
          new_geom.tris.push_back(*j);
      }
    }
    // hex mesh
    if (!quads.empty()) {
      new_geom.quads.clear();
      for (quads_t::const_iterator j = quads.begin(); j != quads.end(); j++) {
        // same as comment above except for quads...
        if (!boundary[(*j)[0]] || !boundary[(*j)[1]]) {
          line_t line = {{(*j)[0], (*j)[1]}};
          new_geom.lines.push_back(line);
        }
        if (!boundary[(*j)[1]] || !boundary[(*j)[2]]) {
          line_t line = {{(*j)[1], (*j)[2]}};
          new_geom.lines.push_back(line);
        }
        if (!boundary[(*j)[2]] || !boundary[(*j)[3]]) {
          line_t line = {{(*j)[2], (*j)[3]}};
          new_geom.lines.push_back(line);
        }
        if (!boundary[(*j)[3]] || !boundary[(*j)[0]]) {
          line_t line = {{(*j)[3], (*j)[0]}};
          new_geom.lines.push_back(line);
        }

        if (boundary[(*j)[0]] && boundary[(*j)[1]] && boundary[(*j)[2]] &&
            boundary[(*j)[3]])
          new_geom.quads.push_back(*j);
      }
    }
  }

  return new_geom;
}

geometry_t &geometry_t::invert_normals() {
  for (normals_t::iterator i = normals.begin(); i != normals.end(); i++)
    for (int j = 0; j < 3; j++)
      (*i)[j] *= -1.0;
  return *this;
}

geometry_t &geometry_t::reorient() {
  using namespace std;

  // TODO: make this work for quads

  if (!tris.empty()) {
    // find all the triangles that each point is a part of
    map<point_t, list<unsigned int>> neighbor_tris;
    for (triangles_t::iterator i = tris.begin(); i != tris.end(); i++)
      for (triangle_t::iterator j = i->begin(); j != i->end(); j++)
        neighbor_tris[points[*j]].push_back(distance(tris.begin(), i));

    /*
      IGNORE THIS

      for each triangle (i),
      for each point (j) in that triangle (i)
      for each triangle (k) that point (j) is a part of
      if the angle between the normal of k and i is obtuse,
      reverse the vertex order of k
    */

    // if we don't have normals, lets calculate them
    if (points.size() != normals.size())
      calculate_surf_normals();

    for (points_t::iterator i = points.begin(); i != points.end(); i++) {
      // skip non boundary verts because their normals are null
      if (!boundary[distance(points.begin(), i)])
        continue;

      list<unsigned int> &neighbors = neighbor_tris[*i];
      for (list<unsigned int>::iterator j = neighbors.begin();
           j != neighbors.end(); j++) {
        triangle_t &tri = tris[*j];
        for (int k = 0; k < 3; k++)
          if (utility::dot(normals[tri[k]],
                           normals[distance(points.begin(), i)]) < 0)
            for (int l = 0; l < 3; l++)
              normals[tri[k]][l] *= -1.0;
      }
    }
  }

  return *this;
}

geometry_t &geometry_t::clear() { return (*this = geometry_t()); }
} // namespace cvcraw_geometry
