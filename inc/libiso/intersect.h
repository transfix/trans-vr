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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef CCV_ISO_INTERSECT
#define CCV_ISO_INTERSECT

#include <libiso/geom.h>

#ifdef __cplusplus
extern "C" {
#endif
	/**
	*	intersect_triangle: Computes the intersection of a ray with a given triangle.
	*						It returns the parameter at the intersection that should 
	*						be positive. Negative return value means failure of 
	*						intersection.
	*/
	float intersect_triangle(iRay ray, Triangle tri, Point point);

	void triangle_normal(Triangle tri, Normal *p_normal);

	/**
	*	in_triangle: return 1 if point is inside triangle
	*/
	int in_triangle(Point point, Triangle tri);

#ifdef __cplusplus
}
#endif

#endif
