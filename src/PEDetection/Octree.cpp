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

template <class _DataType> cOctree<_DataType>::cOctree() {
  Width_mi = Height_mi = Depth_mi = -1;
  Data_mT = NULL;
  Root_m = NULL;
}

// destructor
template <class _DataType> cOctree<_DataType>::~cOctree() { delete Root_m; }

template <class _DataType> void cOctree<_DataType>::setData(_DataType *Data) {
  Data_mT = Data;
  //	MinData_mf=Min,
  //	MaxData_mf=Max;
}

template <class _DataType>
void cOctree<_DataType>::setWHD(int W, int H, int D) {
  Width_mi = W;
  Height_mi = H;
  Depth_mi = D;

  WtimesH_mi = W * H;
  WHD_mi = W * H * D;
}

template <class _DataType>
void cOctree<_DataType>::setOutputFileName(char *FileName) {
  OutputFileName_mi = FileName;
}

template <class _DataType>
void cOctree<_DataType>::CountClasses(int *Classes, int *Start3, int *End3) {
  int i, j, k, loc[3];

  for (i = 0; i < NUM_CLASSES; i++)
    Classes[i] = 0;
  for (k = Start3[2]; k < End3[2]; k++) {
    for (j = Start3[1]; j < End3[1]; j++) {
      for (i = Start3[0]; i < End3[0]; i++) {
        loc[0] = Index(i, j, k);
        if (Data_mT[loc[0]] > 0)
          Classes[(int)Data_mT[loc[0]]]++;
      }
    }
  }
}

template <class _DataType> void cOctree<_DataType>::ComputeOctree() {
  if (Width_mi < 0 || Height_mi < 0 || Depth_mi < 0)
    printf("ComputeOctree: Error!\n");

#ifdef DEBUG_OCTREE
  printf("Computing Octree ... ");
  printf("\n");
  fflush(stdout);
#endif

  CellID_mi = 0;

  Root_m = new cOctreeCell;
  Root_m->setStartCoord(0, 0, 0);
  Root_m->setEndCoord(Width_mi, Height_mi, Depth_mi);
  Root_m->CellID_mi = CellID_mi;
  CountClasses(&(Root_m->Class_mi[0]), &(Root_m->StartCoord_mi[0]),
               &(Root_m->EndCoord_mi[0]));
  CellID_mi++;

  int MaxLevel = 10;
  GenerateOctree(Root_m, MaxLevel);
}

template <class _DataType>
void cOctree<_DataType>::GenerateOctree(cOctreeCell *CurrCell,
                                        int CurrLevel) {
  if (CurrLevel < 0)
    return;
  if (CurrCell == NULL)
    return;

  int i, j, CurrStart[3], CurrEnd[3], CurrMid[3];
  cOctreeCell *Children[8];

  for (i = 0; i < 3; i++) {
    CurrStart[i] = CurrCell->StartCoord_mi[i];
    CurrEnd[i] = CurrCell->EndCoord_mi[i];
    CurrMid[i] = (CurrStart[i] + CurrEnd[i]) / 2;
    if (CurrStart[i] == CurrEnd[i])
      return;
  }

#ifdef DEBUG_OCTREE
  printf("Id = %3d, ", CurrCell->CellID_mi);
  printf("L = %3d, ", CurrLevel);
  printf("Start = %3d %3d %3d, ", CurrStart[0], CurrStart[1], CurrStart[2]);
  printf("End = %3d %3d %3d ", CurrEnd[0], CurrEnd[1], CurrEnd[2]);
//		printf ("Mid = %3d %3d %3d ", CurrMid[0], CurrMid[1],
//CurrMid[2]);
#endif

  for (i = 0; i < 8; i++) {
    Children[i] = new cOctreeCell;
    CurrCell->Children_m[i] = Children[i];
    Children[i]->Parent_m = CurrCell;
  }

  Children[0]->setStartCoord(CurrStart[0], CurrStart[1], CurrStart[2]);
  Children[0]->setEndCoord(CurrMid[0], CurrMid[1], CurrMid[2]);
  Children[1]->setStartCoord(CurrMid[0], CurrStart[1], CurrStart[2]);
  Children[1]->setEndCoord(CurrEnd[0], CurrMid[1], CurrMid[2]);
  Children[2]->setStartCoord(CurrMid[0], CurrMid[1], CurrStart[2]);
  Children[2]->setEndCoord(CurrEnd[0], CurrEnd[1], CurrMid[2]);
  Children[3]->setStartCoord(CurrStart[0], CurrMid[1], CurrStart[2]);
  Children[3]->setEndCoord(CurrMid[0], CurrEnd[1], CurrMid[2]);

  Children[4]->setStartCoord(CurrStart[0], CurrStart[1], CurrMid[2]);
  Children[4]->setEndCoord(CurrMid[0], CurrMid[1], CurrEnd[2]);
  Children[5]->setStartCoord(CurrMid[0], CurrStart[1], CurrMid[2]);
  Children[5]->setEndCoord(CurrEnd[0], CurrMid[1], CurrEnd[2]);
  Children[6]->setStartCoord(CurrMid[0], CurrMid[1], CurrMid[2]);
  Children[6]->setEndCoord(CurrEnd[0], CurrEnd[1], CurrEnd[2]);
  Children[7]->setStartCoord(CurrStart[0], CurrMid[1], CurrMid[2]);
  Children[7]->setEndCoord(CurrMid[0], CurrEnd[1], CurrEnd[2]);
  for (i = 0; i < 8; i++) {
    Children[i]->CellID_mi = CellID_mi++;
    CountClasses(&(Children[i]->Class_mi[0]),
                 &(Children[i]->StartCoord_mi[0]),
                 &(Children[i]->EndCoord_mi[0]));
  }

#ifdef DEBUG_OCTREE
  printf("Classes = ");
  for (i = 1; i < NUM_CLASSES; i++) {
    printf("%6d ", CurrCell->Class_mi[i]);
  }
  printf("\n");
  fflush(stdout);
#endif

  int NumClasses;
  for (i = 0; i < 8; i++) {
    NumClasses = 0;
    for (j = 1; j < NUM_CLASSES; j++) {
      if (Children[i]->Class_mi[j] > 0)
        NumClasses++;
    }
    if (NumClasses >= 2)
      GenerateOctree(CurrCell->Children_m[i], CurrLevel - 1);
  }
}

template <class _DataType> void cOctree<_DataType>::PrintOctree() {
  char OutFileName_c[512];
  FILE *fp;

  sprintf(OutFileName_c, "%s_octree.txt", OutputFileName_mi);
  fp = fopen(OutFileName_c, "w");

  TextOutput(fp, Root_m);
}

// Format of Output
// (Cell ID) (Parent ID) (Child IDs ...)
// (Start XYZ) (End XYZ)
// (# Each Class)
//
template <class _DataType>
void cOctree<_DataType>::TextOutput(FILE *fp, cOctreeCell *Cell) {
  int i;

  if (Cell == NULL)
    return;

  fprintf(fp, "%5d ", Cell->CellID_mi);
  if (Cell->Parent_m != NULL)
    fprintf(fp, "%5d ", Cell->Parent_m->CellID_mi);
  else
    fprintf(fp, "   -1 ");
  for (i = 0; i < 8; i++) {
    if (Cell->Children_m[i] != NULL)
      fprintf(fp, "%5d ", Cell->Children_m[i]->CellID_mi);
    else
      fprintf(fp, "   -1 ");
  }
  fprintf(fp, "%3d %3d %3d ", Cell->StartCoord_mi[0], Cell->StartCoord_mi[1],
          Cell->StartCoord_mi[2]);
  fprintf(fp, "%3d %3d %3d ", Cell->EndCoord_mi[0], Cell->EndCoord_mi[1],
          Cell->EndCoord_mi[2]);
  for (i = 0; i < NUM_CLASSES; i++)
    fprintf(fp, "%8d ", Cell->Class_mi[i]);
  fprintf(fp, "\n");
  fflush(fp);

  for (i = 0; i < 8; i++)
    TextOutput(fp, Cell->Children_m[i]);
}

template <class _DataType>
int cOctree<_DataType>::Index(int X, int Y, int Z) {
  if (X < 0 || Y < 0 || Z < 0 || X >= Width_mi || Y >= Height_mi ||
      Z >= Depth_mi)
    return 0;
  else
    return (Z * WtimesH_mi + Y * Width_mi + X);
}

cOctree<unsigned char> __Octree0;
// cOctree<unsigned short>		__Octree1;
// cOctree<int>				__Octree2;
// cOctree<float>				__Octree3;
