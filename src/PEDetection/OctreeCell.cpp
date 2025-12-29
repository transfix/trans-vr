/*
  Copyright 2006 The University of Texas at Austin

        Authors: Sangmin Park <smpark@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PEDetection.

  PEDetection is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PEDetection is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <PEDetection/CompileOptions.h>
#include <PEDetection/Octree.h>
#include <stdio.h>

cOctreeCell::cOctreeCell() {
  int i;
  Parent_m = NULL;
  for (i = 0; i < 8; i++)
    Children_m[i] = NULL;
  for (i = 0; i < 3; i++) {
    StartCoord_mi[i] = -1;
    EndCoord_mi[i] = -1;
  }
  for (i = 0; i < NUM_CLASSES; i++)
    Class_mi[i] = -1;
  CellID_mi = -1;
}

// destructor
cOctreeCell::~cOctreeCell() {
  int i;
  for (i = 0; i < 8; i++) {
    delete Children_m[i];
    Children_m[i] = NULL;
  }
}

void cOctreeCell::setStartCoord(int X, int Y, int Z) {
  StartCoord_mi[0] = X;
  StartCoord_mi[1] = Y;
  StartCoord_mi[2] = Z;
}

void cOctreeCell::setEndCoord(int X, int Y, int Z) {
  EndCoord_mi[0] = X;
  EndCoord_mi[1] = Y;
  EndCoord_mi[2] = Z;
}
