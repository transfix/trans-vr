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
#include <assert.h>
#include <libiso/intersect.h>
#include <libiso/iso_util.h>
#include <libiso/iso.h>
#include <libiso/geo_util.h>

int iso_intersect(iRay ray, float val, Cell cell, Point point, Normal vec)
{
	int nt, i, j;
	Triangle tris[5];                 /* at most 5 triangles in a cell */
	float t, t0;
	Point pnt, pnt0;
	float w[3];

	nt = extract_contour(val, cell, tris);
	if(nt == 0)	return 0;

	t0 = -1;
	for(i = 0; i < nt; i++) {
		t = intersect_triangle(ray, tris[i], pnt);
		if(t < 0) continue;
		if(t0 < 0 || t < t0) {
			t0 = t;
			for(j = 0; j < 3; j++) pnt0[j] = pnt[j];
		}
	}
	if(t0 < 0) return 0;

	/* find coordinates of the intersection in the cell 
	   and get its normal by trilinear interpolation
	*/
	for(i = 0; i < 3; i++) point[i] = pnt0[i];
	for(i = 0; i < 3; i++) {
		w[i] = (point[i] - (cell.orig[i] + cell.id[i]*cell.span[i]))/cell.span[i];
		if(!(w[i] >= 0 && w[i] <= 1)) {
			/*printf("w: %f %f %f\n", w[0], w[1], w[2]);
			printf("point: %f %f %f\n", point[0], point[1], point[2]);
			printf("orig: %f %f %f\n", cell.orig[0], cell.orig[1], cell.orig[2]);
			printf("id: %d %d %d\n",cell.id[0], cell.id[1], cell.id[2]);
			printf("span: %f %f %f\n", cell.span[0], cell.span[1], cell.span[2]);*/
		}
		/*assert(w[i] >= 0 && w[i] <= 1);*/
	}
	/* get normal at the intersection by trilinear interpolation */

	vec[0] = -(1-w[1])*(1-w[2])*cell.func[0] + (1-w[1])*(1-w[2])*cell.func[1] 
			 +(1-w[1])*w[2]*cell.func[2] - (1-w[1])*w[2]*cell.func[3]
			 -w[1]*(1-w[2])*cell.func[4] + w[1]*(1-w[2])*cell.func[5]
			 +w[1]*w[2]*cell.func[6] - w[1]*w[2]*cell.func[7];

	vec[1] = -(1-w[0])*(1-w[2])*cell.func[0] - w[0]*(1-w[2])*cell.func[1] 
			 -w[0]*w[2]*cell.func[2] - (1-w[0])*w[2]*cell.func[3]
			 +(1-w[0])*(1-w[2])*cell.func[4] + w[0]*(1-w[2])*cell.func[5]
			 +w[0]*w[2]*cell.func[6] + (1-w[0])*w[2]*cell.func[7];

	vec[2] = -(1-w[0])*(1-w[1])*cell.func[0] - w[0]*(1-w[1])*cell.func[1] 
			 +w[0]*(1-w[1])*cell.func[2] + (1-w[0])*(1-w[1])*cell.func[3]
			 -(1-w[0])*w[1]*cell.func[4] - w[0]*w[1]*cell.func[5]
			 +w[0]*w[1]*cell.func[6] + (1-w[0])*w[1]*cell.func[7];

	for(i = 0; i < 3; i++) {
		vec[i] = vec[i] / cell.span[i];
	}
	normalize(vec);
	return 1;           /* intersection exists */
}

int iso_intersectW(iRay ray, float val, Cell* cell, float w[3])
{
	int nt, i, j;
	Triangle tris[5];                 /* at most 5 triangles in a cell */
	float t, t0;
	Point pnt, pnt0;

	nt = extract_contour(val, *cell, tris);
	if(nt == 0)	return 0;

	t0 = -1;
	for(i = 0; i < nt; i++) {
		t = intersect_triangle(ray, tris[i], pnt);
		if(t < 0) continue;
		if(t0 < 0 || t < t0) {
			t0 = t;
			for(j = 0; j < 3; j++) pnt0[j] = pnt[j];
		}
	}
	if(t0 < 0) return 0;

	/* find coordinates of the intersection in the cell 
	   and get its normal by trilinear interpolation
	*/
	for(i = 0; i < 3; i++) {
		w[i] = (pnt0[i] - (cell->orig[i] + cell->id[i]*cell->span[i]))/cell->span[i];
	}
	return 1;
}
