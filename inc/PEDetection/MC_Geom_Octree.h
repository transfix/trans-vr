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

#ifndef FILE_MC_GEON_OCTREE_H
#define FILE_MC_GEON_OCTREE_H

#include <PEDetection/MarchingCubes.h>
#include <PEDetection/Stack.h>

#define MAT_NUM_DOES_NOT_EXIST -99999
#define MAX_X_RESOLUTION 1024
#define MAX_Y_RESOLUTION 1024

template <class _ValueType> struct sOctree<_ValueType> {
  _ValueType Vertices_t[8];
  int ConfigurationNum_i;
  int EdgeToVertex_i[12];
  double Xd, Yd, Zd;

  int IthChild;
  struct sOctree<_ValueType> *Parent_p;
  struct sOctree<_ValueType> *Children_p[8];
};

// Connected Component searching class
// Inherited from the marching cube class

template <class _DataType, class _ExtraDataType>
class cMC_Geom : public cMarchingCubes<_DataType> {

protected:
  // Vertex To Triangle link Index
  int *VertexToTriangleIndex_mi;
  int MaxNumVToTLinks_mi; // Max # of vertex to triangle links

  // Triangle Classification
  int *MatNumOfTriangle_mi;

  // Classification Related Variables
  int NumMaterials_mi;
  int *Ranges_mi;

  int MaxNumMaterials;
  map<int, unsigned char> BoundaryLocs_map[30];

  cPEDetection<_ExtraDataType> PED_m;

  float *ZeroCrossingVolume_mf;

  struct sOctree<_DataType> *OctreeRoot_ms, OctreeRootPrev_ms;
  struct sOctree<_DataType> *OctreeChildCurrBuffer_ms[MAX_X_RESOLUTION];
  struct sOctree<_DataType>
      *OctreeChildPrevBuffer_ms[MAX_X_RESOLUTION * MAX_Y_RESOLUTION];

public:
  cMC_Geom();
  ~cMC_Geom();
  void setNumMaterials(int NumMat);
  void setAMaterialRange(int MatNum, int Intensity1, int Intensity2);
  void setPEDObject(cPEDetection<_ExtraDataType> &PED) { PED_m = PED; };

public:
  void IsosurfaceGenerationFromSecondD();
  void ExtractingIsosurfacesFromSecondD(float IsoValue, float RangeBegin,
                                        float RangeEnd);
  void TriangleClassification();
  void BuildVertexToTriangleLink();
  int DecidingMaterialNum(map<int, unsigned char> &TriangleIndex_map);
  void BoundaryVoxelExtraction(int MatNum, _DataType MatMin, _DataType MatMax,
                               map<int, unsigned char> &BoundaryLocs_map);
  void AdjustingBoundary(map<int, unsigned char> &BoundaryLocs_map);
  void SearchingBoundary(cStack<int> &StackInitBLocs);
  void SearchingBoundary2(map<int, unsigned char> &BoundaryLocs_map);
  int IsMaterialBoundaryUsingMinMax(int DataLoc, _DataType MatMin,
                                    _DataType MatMax);
  int MarkingVoxelEdges(int *DataCoor3, double *GradVec3,
                        double *ZeroCrossingPt3);

private:
  void OctreeSubdivision(int ConfigIndex, int Xi, int Yi, int Zi,
                         _DataType *DataCube8);
  void ComputeAndAddTriangles_Octree(int ConfigIndex, int Xi, int Yi, int Zi,
                                     _DataType *DataCube8);

  // Octree Related Functions
protected:
  void ClearOctree(struct sOctree<_DataType> *Node);

public:
  void SaveMatGeometry_RAW(char *filename, int MatNum);
  void Destroy();
};

// extern int		TempMatNum;

// Control.cpp
template <class T>
void SaveVolume(T *data, float Minf, float Maxf, char *Name);

/*
        The Structures of the vertex and triangle indexes

VertexIndex			VertexToTriangleIndex_mi
        V1	X1 Y1 Z1	I1 I2 I3 I4 -1 -1 -1 -1
        V2	X2 Y2 Z2	I1 I2 I3 I4 I5 I6 -1 -1
        V3	X3 Y3 Z3	I1 -1 -1 -1 -1 -1 -1 -1
        .
        .
        .


TriangleIndex				MatNumOfTriangle_mi
                (VertexIndex_mi)	(MatNumOfTriangle_mi)
        1	V1 V2 V3				M1
        2	V1 V2 V4				M1
        3	V1 V2 V5				M1
        .
        .
        .

*/

#endif
