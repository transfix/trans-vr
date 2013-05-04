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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef FILE_MARCHINGCUBES_H
#define FILE_MARCHINGCUBES_H

#include <map.h>
#include <PEDetection/PEDetection.h>


#define		VRML_LINE		1000
#define		VRML_FACE		2000

//#define		DEBUG_MC


//#define		ISO_NMJ


class Vector3f;


template <class _DataType> 
class cMarchingCubes {

	protected:
		int			Width_mi, Height_mi, Depth_mi;
		int			WtimesH_mi, WHD_mi;
		
		_DataType	*Data_mT;
		float		MinData_mf, MaxData_mf;
		float		IsoValue_mf;

		float		*GradientVec_mf;
		float		*GradientMag_mf;
		float		MinGradMag_mf, MaxGradMag_mf;
		
		float		*SecondDerivative_mf;
		float		MinSecond_mf, MaxSecond_mf;
		
		float		SpanX_mf, SpanY_mf, SpanZ_mf;

		
		long int	NumVertices_mi, MaxNumVertices_mi;
		long int	NumTriangles_mi, MaxNumTriangles_mi;
		Vector3f	*Vertices_mv3, *Normals_mv3, *Colors_mv3;
		int			*VertexIndex_mi;

		int			*VertexBuffPrev_mi[12];
		int			*VertexBuffCurr_mi[12];
		int			*VertexBuffInit_mi;
		
		int			RecomputeMinMaxData_mi;
		
		unsigned char	*ColorIDVolume_muc;
		int			ColorTable_mi[256][3];


	public:
		cMarchingCubes();
		~cMarchingCubes();		

		void setData(_DataType *Data);
		void setData(_DataType *Data, float Minf, float Maxf);
		void setGradientVectors(float *GradVec);
		void setWHD(int W, int H, int D);
		void setSpanXYZ(float SpanX, float SpanY, float SpanZ);
		void setGradient(float *Grad, float Min, float Max);
		void setSecondDerivative(float *SecondD, float Min, float Max);
		void setIDVolume(unsigned char *BVVolume_uc);
		void setIDToColorTable(int *ColorTable256);
		void ClearGeometry();
		

	public:
		int	getNumVertices() { return NumVertices_mi; };
		int getNumTriangles() { return NumTriangles_mi; };
		float* getVerticesPtsf() { return (float *)Vertices_mv3; };
		int* getVertexIndex() { return (int *)VertexIndex_mi; };


	public:
		void ExtractingIsosurfaces(float IsoValue);
		void ExtractingIsosurfacesIDColor(float IsoValue, unsigned char ClassNum);

	protected:
		void ComputeAndAddTriangles(int ConfigIndex, int Xi, int Yi, int Zi, _DataType *DataCube8);
		void ComputeAndAddTriangles(int ConfigIndex, int Xi, int Yi, int Zi, _DataType *DataCube8,
									float R, float G, float B);
		int IsInBuffer(int Xi, int Yi, int Zi, int EdgeIndex);
		int AddATriangle(int *TriIdx);
		int AddAVertex(Vector3f& Vertex);
		int AddAVertex(Vector3f& Vertex, float R, float G, float B);
		void CopyVertexBuffer();
		void MultSpanXYZToVertices();
		
		int GradVecInterpolation(double* LocXYZ, double* GradVec_Ret);
		int GradVecInterpolation(float LocX, float LocY, float LocZ, float* GradVec_Ret);
		int GradVecInterpolation(double LocX, double LocY, double LocZ, double* GradVec_Ret);
		
		void ComputeVertexNormals();
		float ranf(float Minf, float Maxf);
	
	public:
		void ComputeSurfaceArea();
		double TriangleArea(float *Pt1, float *Pt2, float *Pt3);
		void SaveGeometry_RAW(char *filename);
		void SaveGeometry_RAWC(char *filename, char *Postfix, int FlipX);
		void SaveGeometry_RAWNC(char *filename, char *Postfix);
		void SaveGeometry_RAW(char *filename, char *tail);
		void SaveGeometry_VRML(char *filename, int DrawingOption);
		void SaveGeometry_OBJ(char *filename);

		void SaveGeometry_Samrat(char *filename, char *Postfix);

		void Destroy();

};


// "MC_Configuration.h"
extern int	ConfigurationTable[256][16][2];
extern int	RelativeLocByIndex[8][3];
extern int	CubeEdgeDir[8][8][3];
extern int	LevelOneUnConfiguration_gi[256];


extern float	SpanX_gf, SpanY_gf, SpanZ_gf;

#endif

