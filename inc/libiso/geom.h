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

#ifndef CCV_ISO_GEOMETRY
#define CCV_ISO_GEOMETRY

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif
