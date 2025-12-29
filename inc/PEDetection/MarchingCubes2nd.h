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

#ifndef FILE_MARCHINGCUBES2ND_H
#define FILE_MARCHINGCUBES2ND_H

#include <map.h>

#define VRML_LINE 1000
#define VRML_FACE 2000

class Vector3f;

template <class _DataType> class cMarchingCubes2nd {

protected:
  int Width_mi, Height_mi, Depth_mi;
  int WtimesH_mi, WHD_mi;

  float *GradientVec_mf;

  int NumVertices_mi, MaxNumVertices_mi;
  int NumTriangles_mi, MaxNumTriangles_mi;
  Vector3f *Vertices_mv3, *Normals_mv3;
  int *VertexIndex_mi;

  int *VertexBuffPrev_mi[12];
  int *VertexBuffCurr_mi[12];

  map<int, int *> VertexIndexBuffer;

public:
  cMarchingCubes();
  ~cMarchingCubes();

  void setData(_DataType *Data, float Minf, float Maxf);
  void setGradientVectors(float *GradVec);
  void setWHD(int W, int H, int D);
  void InitializeBuffer();

public:
  void ExtractingIsosurfaces(float IsoValue);

private:
  void ComputeAndAddTriangles(int ConfigIndex, int Xi, int Yi, int Zi,
                              _DataType *DataCube8);
  int IsInBuffer(int Xi, int Yi, int Zi, int EdgeIndex);
  int AddATriangle(int *TriIdx);
  int AddAVertex(Vector3f &Vertex);
  void CopyVertexBuffer();

  int GradVecInterpolation(double *LocXYZ, double *GradVec_Ret);
  int GradVecInterpolation(float LocX, float LocY, float LocZ,
                           float *GradVec_Ret);
  int GradVecInterpolation(double LocX, double LocY, double LocZ,
                           double *GradVec_Ret);

  void ComputeVertexNormals();

public:
  void SaveGeometry_RAW(char *filename);
  void SaveGeometry_VRML(char *filename, int DrawingOption);
  void Destroy();
};

#endif
