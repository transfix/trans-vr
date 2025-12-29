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

#ifndef FILE_OCTREE_H
#define FILE_OCTREE_H

#include <stdio.h>
#include <stdlib.h>

#define DEBUG_OCTREE

#define NUM_CLASSES 6

#define OTHERS 0
#define PRESYNAPTIC 1
#define PRESYNAPTIC_NERVE 2
#define SYNAPTIC_CLEFT 3
#define POSTSYNAPTIC_MUSCLE 4
#define INTRA_CELLULAR 5

// pre-synaptic membrane
// pre-synaptic nerve cell membrane
// between pre- and post-synaptic membranes,post-synaptic membrane
// the post-synaptic muscle membrane
// below post-synaptic muscle membrane

class cOctreeCell {
public:
  cOctreeCell *Parent_m;
  cOctreeCell *Children_m[8];
  int StartCoord_mi[3], EndCoord_mi[3];
  int Class_mi[NUM_CLASSES];
  int CellID_mi;

public:
  cOctreeCell();
  ~cOctreeCell();

  void setStartCoord(int X, int Y, int Z);
  void setEndCoord(int X, int Y, int Z);
};

template <class _DataType> class cOctree {

protected:
  _DataType *Data_mT;
  float MinData_mf, MaxData_mf;
  int Width_mi, Height_mi, Depth_mi;
  int WtimesH_mi, WHD_mi;
  char *OutputFileName_mi;
  int CellID_mi;

  cOctreeCell *Root_m;

public:
  cOctree();
  ~cOctree();
  void setData(_DataType *Data);
  void setWHD(int W, int H, int D);
  void setOutputFileName(char *FileName);
  void ComputeOctree();
  void PrintOctree();

private:
  void CountClasses(int *Classes, int *Start3, int *End3);
  void GenerateOctree(cOctreeCell *CurrCell, int Level);
  void TextOutput(FILE *fp, cOctreeCell *Cell);
  int Index(int X, int Y, int Z);
};

#endif
