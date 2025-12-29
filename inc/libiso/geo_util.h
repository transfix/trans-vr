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
 *	geo_util.h: collects often used geometrical subroutines
 ******/

#ifndef GEO_UTIL_H
#define GEO_UTIL_h

#ifdef __cplusplus
extern "C" {
#endif

void cross_product(float v1[3], float v2[3], float normal[3]);

float dot_product(float v1[3], float v2[3]);

void normalize(float v[3]);

float length(float v[3]);

float distance(float v1[3], float v2[3]);

#ifdef __cplusplus
}
#endif

#endif
