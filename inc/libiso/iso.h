/*
  Copyright 2000-2002 The University of Texas at Austin

        Authors: Sanghun Park <hun@ices.utexas.edu>
                 Xiaoyu Zhang <xiaoyu@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of volren.

  volren is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  volren is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/******
       iso.h : This library computes the intersection between a ray and
       isosurface.
******/

#ifndef CCV_ISO_ISO_H
#define CCV_ISO_ISO_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CCV_ISO_GEOMETRY
#define CCV_ISO_GEOMETRY
typedef float Point[3];
typedef float Normal[3];
typedef float Vector[3];

typedef struct ray {
  Point orig; /* starting point of ray */
  Vector dir; /* ray direction */
} iRay;

typedef struct triangle {
  Point vert[3];
} Triangle;

typedef struct cell {
  int id[3];
  float orig[3];
  float span[3];
  float func[8];
  /*Normal *norm;*/
  Normal norm[8];
} Cell;
#endif
/**
   iso_intersect : This fuction computes the intersection of ray and
   isosurface in a cell. Input is the ray, functions defined at the eight
   vertices of the cell and normals at these vertices. This function return 0
   if the ray doesn't intersect with the isosurface or the cell doesn't have
   isosurface of given value. It returns 1 and modifies the passed out
   intersection point and normal at that point using function parameters.
*/
int iso_intersect(iRay ray, float val, Cell cell, Point pnt, Normal vec);

/**
 * This function computes the local offset of the intersectio point.
 * @return 0 if no intersection is available
 */
int iso_intersectW(iRay ray, float val, Cell *cell, float w[3]);

#ifdef __cplusplus
}
#endif

#endif
