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

#ifndef FILE_CC_GEOM_H
#define FILE_CC_GEOM_H


#include <PEDetection/MarchingCubes.h>

#define		MAT_NUM_DOES_NOT_EXIST		-99999


// Connected Component searching class
// Inherited from the marching cube class


template <class _DataType> 
class cMC_Geom : public cMarchingCubes<_DataType> {

	protected:

		// Vertex To Triangle link Index
		int			*VertexToTriangleIndex_mi;
		int			MaxNumVToTLinks_mi;			// Max # of vertex to triangle links

		// Triangle Classification
		int			*MatNumOfTriangle_mi;

		// Classification Related Variables
		int			NumMaterials_mi;
		int			*Ranges_mi;


		int			MaxNumMaterials;
		map<int, unsigned char>		BoundaryLocs_map[30];
		

	public:
		cMC_Geom();
		~cMC_Geom();		
		void setNumMaterials(int   NumMat);
		void setAMaterialRange(int MatNum, int Intensity1, int Intensity2);

	public:
		void TriangleClassification();
		void BuildVertexToTriangleLink();
		int DecidingMaterialNum(map<int, unsigned char>& TriangleIndex_map);
		void BoundaryVoxelExtraction(int MatNum, _DataType MatMin, _DataType MatMax, 
										map<int, unsigned char>& BoundaryLocs_map);
		void AdjustingBoundary(map<int, unsigned char>& BoundaryLocs_map);
		int IsMaterialBoundaryUsingMinMax(int DataLoc, _DataType MatMin, _DataType MatMax);

	public:
		void SaveMatGeometry_RAW(char *filename, int MatNum);
		void Destroy();

};



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

