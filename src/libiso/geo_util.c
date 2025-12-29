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

#include <libiso/geo_util.h>
#include <math.h>
#include <stdio.h>

float distance(float v1[3], float v2[3]) {
  float v[3];
  int i;
  for (i = 0; i < 3; i++)
    v[i] = v2[i] - v1[i];
  return length(v);
}

float length(float v[3]) { return ((float)sqrt(dot_product(v, v))); }

float dot_product(float v1[3], float v2[3]) {
  return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
}

void normalize(float v[3]) {
  float len;
  int i;
  len = length(v);
  if (len == 0.0) {
    // fprintf(stderr, "the length of the vector is 0\n");
    return;
  }
  for (i = 0; i < 3; i++)
    v[i] = v[i] / len;
}

void cross_product(float v1[3], float v2[3], float normal[3]) {
  normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
  normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
  normal[2] = v1[0] * v2[1] - v1[1] * v2[0];
}
