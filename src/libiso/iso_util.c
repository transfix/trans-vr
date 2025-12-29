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

#include <libiso/cubes.h>
#include <libiso/iso_util.h>
#include <libiso/vtkMarchingCubesCases.h>

int extract_contour(float val, Cell cell, Triangle tris[5]) {
  int code;
  int e, edge, t, i, j, k;
  Point edge_v[12];

  code = 0;
  if (cell.func[0] < val)
    code |= 0x01;
  if (cell.func[1] < val)
    code |= 0x02;
  if (cell.func[2] < val)
    code |= 0x04;
  if (cell.func[3] < val)
    code |= 0x08;
  if (cell.func[4] < val)
    code |= 0x10;
  if (cell.func[5] < val)
    code |= 0x20;
  if (cell.func[6] < val)
    code |= 0x40;
  if (cell.func[7] < val)
    code |= 0x80;

  /* common case where no isosurface exists */
  if (cubeedges[code][0] == 0)
    return 0;
  for (e = 0; e < cubeedges[code][0]; e++) {
    edge = cubeedges[code][1 + e];
    interp_edge(val, cell, edge, edge_v[edge]);
  }

  i = 0;
  for (t = 0; triCases[code].edges[t] != -1;) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++)
        tris[i].vert[j][k] = edge_v[triCases[code].edges[t]][k];
      t++;
    }
    i++;
  }
  return i;
}

void interp_edge(float val, Cell cell, int edge, Point point) {
  EdgeInfo *ei = &edgeinfo[edge];

  switch (ei->dir) {
  case 0:
    interp_x(cell.id[0] + ei->di, cell.id[1] + ei->dj, cell.id[2] + ei->dk,
             ei->d1, ei->d2, val, cell, point);
    break;
  case 1:
    interp_y(cell.id[0] + ei->di, cell.id[1] + ei->dj, cell.id[2] + ei->dk,
             ei->d1, ei->d2, val, cell, point);
    break;
  case 2:
    interp_z(cell.id[0] + ei->di, cell.id[1] + ei->dj, cell.id[2] + ei->dk,
             ei->d1, ei->d2, val, cell, point);
    break;
  }
}

void interp_x(int i, int j, int k, int v1, int v2, float val, Cell cell,
              Point point) {
  double x;

  x = (val - cell.func[v1]) / (cell.func[v2] - cell.func[v1]);
  point[0] = (float)(cell.orig[0] + cell.span[0] * (i + x));
  point[1] = cell.orig[1] + cell.span[1] * j;
  point[2] = cell.orig[2] + cell.span[2] * k;
}

void interp_y(int i, int j, int k, int v1, int v2, float val, Cell cell,
              Point point) {
  double x;

  x = (val - cell.func[v1]) / (cell.func[v2] - cell.func[v1]);
  point[0] = cell.orig[0] + cell.span[0] * i;
  point[1] = (float)(cell.orig[1] + cell.span[1] * (j + x));
  point[2] = cell.orig[2] + cell.span[2] * k;
}

void interp_z(int i, int j, int k, int v1, int v2, float val, Cell cell,
              Point point) {
  double x;

  x = (val - cell.func[v1]) / (cell.func[v2] - cell.func[v1]);
  point[0] = cell.orig[0] + cell.span[0] * i;
  point[1] = cell.orig[1] + cell.span[1] * j;
  point[2] = (float)(cell.orig[2] + cell.span[2] * (k + x));
}
