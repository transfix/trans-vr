/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Skeletonization.

  Skeletonization is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Skeletonization is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/*
 * poly_tri.c
 *
 * Program to take a polygon definition and convert it into triangles
 * that may then be rendered by the standard triangle rendering
 * algorithms.  This assumes all transformations have been performed
 * already and cuts them up into optimal triangles based on their
 * screen-space representation.
 *
 *      Copyright (c) 1988, Evans & Sutherland Computer Corporation
 *
 * Permission to use all or part of this program without fee is
 * granted provided that it is not used or distributed for direct
 * commercial gain, the above copyright notice appears, and
 * notice is given that use is by permission of Evans & Sutherland
 * Computer Corporation.
 *
 *      Written by Reid Judd and Scott R. Nelson at
 *      Evans & Sutherland Computer Corporation (January, 1988)
 *
 * To use this program, either write your own "draw_triangle" routine
 * that can draw triangles from the definitions below, or modify the
 * code to call your own triangle or polygon rendering code.  Call
 * "draw_poly" from your main program.
 */

/* $Id: PolyTess.cpp 4741 2011-10-21 21:22:06Z transfix $ */

#include <Skeletonization/PolyTess.h>
#include <vector>

// #define V_MAX 100       /* Maximum number of vertices allowed (arbitrary)
// */

#define BIG 1.0e30 /* A number bigger than we expect to find here */

#define COUNTER_CLOCKWISE 0
#define CLOCKWISE 1

namespace PolyTess {
/* A single vertex */
typedef struct {
  // int color;              /* RGB */
  float r, g, b, a;
  float x;
  float y;
  float z;
} vertex;

/* A triangle made up of three vertices */
typedef vertex triangle[3];

/*
 * orientation
 *
 * Return either clockwise or counter_clockwise for the orientation
 * of the polygon.
 */

int orientation(int n,      /* Number of vertices */
                vertex v[]) /* The vertex list */
{
  float area;
  int i;

  /* Do the wrap-around first */
  area = v[n - 1].x * v[0].y - v[0].x * v[n - 1].y;

  /* Compute the area (times 2) of the polygon */
  for (i = 0; i < n - 1; i++)
    area += v[i].x * v[i + 1].y - v[i + 1].x * v[i].y;

  if (area >= 0.0)
    return COUNTER_CLOCKWISE;
  else
    return CLOCKWISE;
}

/*
 * determinant
 *
 * Computes the determinant of the three points.
 * Returns whether the triangle is clockwise or counter-clockwise.
 */
int determinant(int p1, int p2, int p3, /* The vertices to consider */
                vertex v[])             /* The vertex list */
{
  float x1, x2, x3, y1, y2, y3;
  float determ;

  x1 = v[p1].x;
  y1 = v[p1].y;
  x2 = v[p2].x;
  y2 = v[p2].y;
  x3 = v[p3].x;
  y3 = v[p3].y;

  determ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
  if (determ >= 0.0)
    return COUNTER_CLOCKWISE;
  else
    return CLOCKWISE;
} /* End of determinant */

/*
 * distance2
 *
 * Returns the square of the distance between the two points
 */
float distance2(float x1, float y1, float x2, float y2) {
  float xd, yd; /* The distances in X and Y */
  float dist2;  /* The square of the actual distance */

  xd = x1 - x2;
  yd = y1 - y2;
  dist2 = xd * xd + yd * yd;

  return dist2;
} /* End of distance2 */

/*
 * no_interior
 *
 * Returns 1 if no other point in the vertex list is inside
 * the triangle specified by the three points.  Returns
 * 0 otherwise.
 */
int no_interior(int p1, int p2, int p3, /* The vertices to consider */
                vertex v[],             /* The vertex list */
                int vp[],    /* The vertex pointers (which are left) */
                int n,       /* Number of vertices */
                int poly_or) /* Polygon orientation */
{
  int i; /* Iterative counter */
  int p; /* The test point */

  for (i = 0; i < n; i++) {
    p = vp[i]; /* The point to test */
    if ((p == p1) || (p == p2) || (p == p3))
      continue; /* Don't bother checking against yourself */
    if ((determinant(p2, p1, p, v) == poly_or) ||
        (determinant(p1, p3, p, v) == poly_or) ||
        (determinant(p3, p2, p, v) == poly_or)) {
      continue; /* This point is outside */
    } else {
      return 0; /* The point is inside */
    }
  }
  return 1; /* No points inside this triangle */
} /* End of no_interior */

/*
 * draw_poly
 *
 * Call this procedure with a polygon, this divides it into triangles
 * and calls the triangle routine once for each triangle.
 *
 * Note that this does not work for polygons with holes or self
 * penetrations.
 */
Skeletonization::Polygon_vec
draw_poly(int n,      /* Number of vertices in triangle */
          vertex v[]) /* The vertex list (implicit closure) */
{
  int prev, cur, next; /* Three points currently being considered */
  // int vp[V_MAX];              /* Pointers to vertices still left */
  std::vector<int> vp(n);
  int count;            /* How many vertices left */
  int min_vert;         /* Vertex with minimum distance */
  int i;                /* Iterative counter */
  float dist;           /* Distance across this one */
  float min_dist;       /* Minimum distance so far */
  int poly_orientation; /* Polygon orientation */
  triangle t;           /* Triangle structure */

  Skeletonization::Polygon_vec result;

  /*
  if (n > V_MAX) {
    fprintf( stderr, "Error, more than %d vertices.\n", V_MAX);
    return;
  }
  */

  poly_orientation = orientation(n, v);

  for (i = 0; i < n; i++)
    vp[i] = i; /* Put vertices in order to begin */

  /* Slice off clean triangles until nothing remains */

  count = n;
  while (count > 3) {
    min_dist = BIG; /* A real big number */
    min_vert = 0;   /* Just in case we don't find one... */
    for (cur = 0; cur < count; cur++) {
      prev = cur - 1;
      next = cur + 1;
      if (cur == 0) /* Wrap around on the ends */
        prev = count - 1;
      else if (cur == count - 1)
        next = 0;
      /* Pick out shortest distance that forms a good triangle */
      if ((determinant(vp[prev], vp[cur], vp[next], v) == poly_orientation)
          /* Same orientation as polygon */
          && no_interior(vp[prev], vp[cur], vp[next], v, &(vp[0]), count,
                         poly_orientation)
          /* No points inside */
          && ((dist = distance2(v[vp[prev]].x, v[vp[prev]].y, v[vp[next]].x,
                                v[vp[next]].y)) < min_dist))
      /* Better than any so far */
      {
        min_dist = dist;
        min_vert = cur;
      }
    } /* End of for each vertex (cur) */

    /* The following error should "never happen". */
    if (min_dist == BIG)
      fprintf(stderr, "Error: Didn't find a triangle.\n");

    prev = min_vert - 1;
    next = min_vert + 1;
    if (min_vert == 0) /* Wrap around on the ends */
      prev = count - 1;
    else if (min_vert == count - 1)
      next = 0;

    /* Output this triangle */

    t[0].x = v[vp[prev]].x;
    t[0].y = v[vp[prev]].y;
    t[0].z = v[vp[prev]].z;
    // t[0].color = v[vp[prev]].color;
    t[0].r = v[vp[prev]].r;
    t[0].g = v[vp[prev]].g;
    t[0].b = v[vp[prev]].b;
    t[0].a = v[vp[prev]].a;
    t[1].x = v[vp[min_vert]].x;
    t[1].y = v[vp[min_vert]].y;
    t[1].z = v[vp[min_vert]].z;
    // t[1].color = v[vp[min_vert]].color;
    t[1].r = v[vp[min_vert]].r;
    t[1].g = v[vp[min_vert]].g;
    t[1].b = v[vp[min_vert]].b;
    t[1].a = v[vp[min_vert]].a;
    t[2].x = v[vp[next]].x;
    t[2].y = v[vp[next]].y;
    t[2].z = v[vp[next]].z;
    // t[2].color = v[vp[next]].color;
    t[2].r = v[vp[next]].r;
    t[2].g = v[vp[next]].g;
    t[2].b = v[vp[next]].b;
    t[2].a = v[vp[next]].a;

    // draw_triangle( t );
    Skeletonization::Simple_polygon poly;
    for (int idx = 0; idx < 3; idx++)
      poly.push_back(Skeletonization::Simple_vertex(
          Skeletonization::Point(t[idx].x, t[idx].y, t[idx].z),
          Skeletonization::Simple_color(t[idx].r, t[idx].g, t[idx].b,
                                        t[idx].a)));
    result.push_back(poly);

    /* Remove the triangle from the polygon */

    count -= 1;
    for (i = min_vert; i < count; i++)
      vp[i] = vp[i + 1];
  }

  /* Output the final triangle */

  t[0].x = v[vp[0]].x;
  t[0].y = v[vp[0]].y;
  t[0].z = v[vp[0]].z;
  // t[0].color = v[vp[0]].color;
  t[0].r = v[vp[0]].r;
  t[0].g = v[vp[0]].g;
  t[0].b = v[vp[0]].b;
  t[0].a = v[vp[0]].a;
  t[1].x = v[vp[1]].x;
  t[1].y = v[vp[1]].y;
  t[1].z = v[vp[1]].z;
  // t[1].color = v[vp[1]].color;
  t[1].r = v[vp[1]].r;
  t[1].g = v[vp[1]].g;
  t[1].b = v[vp[1]].b;
  t[1].a = v[vp[1]].a;
  t[2].x = v[vp[2]].x;
  t[2].y = v[vp[2]].y;
  t[2].z = v[vp[2]].z;
  // t[2].color = v[vp[2]].color;
  t[2].r = v[vp[2]].r;
  t[2].g = v[vp[2]].g;
  t[2].b = v[vp[2]].b;
  t[2].a = v[vp[2]].a;

  // draw_triangle( t );
  Skeletonization::Simple_polygon poly;
  for (int idx = 0; idx < 3; idx++)
    poly.push_back(Skeletonization::Simple_vertex(
        Skeletonization::Point(t[idx].x, t[idx].y, t[idx].z),
        Skeletonization::Simple_color(t[idx].r, t[idx].g, t[idx].b,
                                      t[idx].a)));
  result.push_back(poly);

  return result;
} /* End of draw_poly */

boost::tuple<std::vector<Skeletonization::Simple_vertex> /* verts */,
             std::vector<unsigned int> /* tris */>
getTris(const Skeletonization::Polygon_set &polygons) {
  Skeletonization::Polygon_vec triangles;

  for (Skeletonization::Polygon_set::const_iterator i = polygons.begin();
       i != polygons.end(); i++) {
    if (i->size() < 3)
      continue;
    std::vector<vertex> verts;
    for (Skeletonization::Simple_polygon::const_iterator j = i->begin();
         j != i->end(); j++) {
      vertex v;
      v.x = j->get<0>().x();
      v.y = j->get<0>().y();
      v.z = j->get<0>().z();
      v.r = j->get<1>().get<0>();
      v.g = j->get<1>().get<1>();
      v.b = j->get<1>().get<2>();
      v.a = j->get<1>().get<3>();
      verts.push_back(v);
    }

    Skeletonization::Polygon_vec newtris =
        draw_poly(verts.size(), &(verts[0]));
    triangles.insert(triangles.end(), newtris.begin(), newtris.end());
  }

  // insert vertices into the set to remove duplicates and to get vertex
  // handles (iterators into the set)
  typedef std::set<Skeletonization::Simple_vertex> Vertices_set;
  typedef std::vector<Skeletonization::Simple_vertex> Vertices_vec;
  typedef std::vector<Vertices_set::iterator> Triangle;
  typedef std::vector<unsigned int> Index_vec;
  Vertices_set vertices_set;
  std::vector<Triangle> handle_triangles;
  for (Skeletonization::Polygon_vec::const_iterator i = triangles.begin();
       i != triangles.end(); i++) {
    Triangle tri;
    for (Skeletonization::Simple_polygon::const_iterator j = i->begin();
         j != i->end(); j++) {
      std::pair<Vertices_set::iterator, bool> result =
          vertices_set.insert(*j);
      tri.push_back(result.first);
    }
    handle_triangles.push_back(tri);
  }

  // we need a random access iterator for the index calculation below
  Vertices_vec vertices_vec(vertices_set.begin(), vertices_set.end());
  Index_vec indices_vec;
  for (std::vector<Triangle>::const_iterator i = handle_triangles.begin();
       i != handle_triangles.end(); i++)
    for (Triangle::const_iterator j = i->begin(); j != i->end(); j++)
      indices_vec.push_back(
          std::lower_bound(vertices_vec.begin(), vertices_vec.end(), **j) -
          vertices_vec.begin());

  return boost::make_tuple(vertices_vec, indices_vec);
}
} // namespace PolyTess
/* End of poly_tri.cpp */
