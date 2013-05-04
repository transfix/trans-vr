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

#ifndef CCV_ISO_UTIL_H
#define CCV_ISO_UTIL_H

#include <libiso/geom.h>

#ifdef __cplusplus
extern "C" {
#endif
	/**
	 *	extract_triangles: extract isocontour triangles in a cell
	 *	                   It returns the number of triangles extracted
	 **/
	int extract_contour(float val, Cell cell, Triangle tris[5]);

	void interp_edge(float val, Cell cell, int edge, Point point);

	void interp_x(int i, int j, int k, int v1, int v2, float val,
				  Cell cell, Point point);

	void interp_y(int i, int j, int k, int v1, int v2, float val,
				  Cell cell, Point point);

	void interp_z(int i, int j, int k, int v1, int v2, float val,
				  Cell cell, Point point);
	
#ifdef __cplusplus
}
#endif

#endif
