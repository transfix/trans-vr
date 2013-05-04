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

#include <stdio.h>
#include <math.h>
#include <libiso/intersect.h>
#include <libiso/geo_util.h>

void triangle_normal(Triangle tri, Normal* p_normal)
{
	float v1[3], v2[3];
	int i;
	for(i = 0; i < 3; i++) {
		v1[i] = tri.vert[1][i] - tri.vert[0][i];
		v2[i] = tri.vert[2][i] - tri.vert[0][i];
	}
	cross_product(v1, v2, *p_normal);
	normalize(*p_normal);
}

float intersect_triangle(iRay ray, Triangle tri, Point point)
{
	Normal norm;			/* triangle normal */
	float t, fz, fm;
	int i;
	triangle_normal(tri, &norm);
	fz = 0; fm = 0;
	for (i = 0; i < 3; i++) {
		fz += norm[i] * (tri.vert[0][i]-ray.orig[i]);
		fm += norm[i] * ray.dir[i];
	}
	if(fm == 0.0) return -1;
	t = fz / fm;
	/* don't consider negative parameter any more */
	if(t < 0.0) return t;
	for(i = 0; i < 3; i++) point[i] = ray.orig[i] + ray.dir[i] * t;
#ifdef _DEBUG
	printf("intersection point: (%f, %f, %f)\n", 
			point[0], point[1], point[2]);
#endif
	if(in_triangle(point, tri)) return t;
	return -1;
}

int in_triangle(Point point, Triangle tri)
{
	/* This implementation is based on "Graphics Gems" p390 */
	int i1, i2;
	Normal norm;
	float u0, v0, u1, v1, u2, v2;
	float alpha, beta;
	int inter = 0;

	triangle_normal(tri, &norm);
	if(fabs(norm[0]) >= fabs(norm[1]) && fabs(norm[0]) >= fabs(norm[2])) {
		i1 = 1; i2 = 2;
	}else if(fabs(norm[1]) >= fabs(norm[0]) && fabs(norm[1]) >= fabs(norm[2])) {
		i1 = 0; i2 = 2;
	}else {
		i1 = 0; i2 = 1;
	}
	u0 = point[i1] - tri.vert[0][i1];
	v0 = point[i2] - tri.vert[0][i2];
	u1 = tri.vert[1][i1] - tri.vert[0][i1];
	v1 = tri.vert[1][i2] - tri.vert[0][i2];
	u2 = tri.vert[2][i1] - tri.vert[0][i1];
	v2 = tri.vert[2][i2] - tri.vert[0][i2];

    if(u1 == 0) {
        beta = u0/u2;
        if((beta >= 0.) && (beta <= 1.)) {
            alpha = (v0 - beta*v2)/v1;
            inter = ((alpha >= 0.) && ((alpha+beta) <= 1.))? 1:0;
        }
    } else {
        beta = (v0*u1 - u0*v1)/(v2*u1 - u2*v1);
        if ((beta >= 0.)&&(beta <= 1.)) {
            alpha = (u0 - beta*u2)/u1;
            inter = ((alpha >= 0) && ((alpha+beta) <= 1.))? 1:0;
        }
    }

	return inter;
}
