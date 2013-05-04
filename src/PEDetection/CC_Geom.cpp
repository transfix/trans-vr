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

#include <PEDetection/CC_Geom.h>
#include <PEDetection/CompileOptions.h>


template <class _DataType>
cMC_Geom<_DataType>::cMC_Geom()
{
	VertexToTriangleIndex_mi = NULL;
	MaxNumVToTLinks_mi = 20;
	
	// Classification Related Variables
	NumMaterials_mi = 0;
	Ranges_mi = NULL;
	
	MaxNumMaterials = 30;
	for (int i=0; i<MaxNumMaterials; i++) {
		BoundaryLocs_map[i].clear();
	}
}


// destructor
template <class _DataType>
cMC_Geom<_DataType>::~cMC_Geom()
{
	delete [] VertexToTriangleIndex_mi;
	delete [] Ranges_mi;

}

template <class _DataType>
void cMC_Geom<_DataType>::TriangleClassification()
{
	int		i, j, VertIdx[4], TriIndex[2], MatNum;
	

	for (i=0; i<NumMaterials_mi; i++) {
		BoundaryVoxelExtraction(i, Ranges_mi[i*2], Ranges_mi[i*2 + 1], BoundaryLocs_map[i]);
	}
	
	BuildVertexToTriangleLink();

	MatNumOfTriangle_mi = new int [NumTriangles_mi];
	for (i=0; i<NumTriangles_mi; i++) MatNumOfTriangle_mi[i] = -1;


	map<int, unsigned char>			ConnectedTriangleIndex_map;
	map<int, unsigned char>			VertexIndex_map;
	map<int, unsigned char>::iterator	VertexIndex_it;
	map<int, unsigned char>::iterator	TriangleIndex_it;
	ConnectedTriangleIndex_map.clear();
	VertexIndex_map.clear();

	int		NumDisconnectedObjects;

	for (i=0; i<NumTriangles_mi; i++) {
		
		if (MatNumOfTriangle_mi[i]==-1) { // -1 means that it is not touched
			
			NumDisconnectedObjects++;
			
			VertexIndex_map[VertexIndex_mi[i*3 + 0]] = (unsigned char)0;
			VertexIndex_map[VertexIndex_mi[i*3 + 1]] = (unsigned char)0;
			VertexIndex_map[VertexIndex_mi[i*3 + 2]] = (unsigned char)0;
			
			ConnectedTriangleIndex_map.clear();
			ConnectedTriangleIndex_map[i] = (unsigned char)0;
			
			do {
				
				VertexIndex_it = VertexIndex_map.begin();
				VertIdx[0] = (*VertexIndex_it).first;
				VertexIndex_map.erase(VertIdx[0]);

				for (j=0; j<MaxNumVToTLinks_mi; j++) {
					if (VertexToTriangleIndex_mi[VertIdx[0]*MaxNumVToTLinks_mi + j]>0) {
						TriIndex[0]= VertexToTriangleIndex_mi[VertIdx[0]*MaxNumVToTLinks_mi + j];
						TriangleIndex_it = ConnectedTriangleIndex_map.find(TriIndex[0]);
						
						if (TriangleIndex_it==ConnectedTriangleIndex_map.end()) {
							ConnectedTriangleIndex_map[TriIndex[0]] = (unsigned char)0;
						
							VertexIndex_map[VertexIndex_mi[TriIndex[0]*3 + 0]] = (unsigned char)0;
							VertexIndex_map[VertexIndex_mi[TriIndex[0]*3 + 1]] = (unsigned char)0;
							VertexIndex_map[VertexIndex_mi[TriIndex[0]*3 + 2]] = (unsigned char)0;

						}
					}
				}
				
			
			} while(VertexIndex_map.size()>0);
			
			MatNum = DecidingMaterialNum(ConnectedTriangleIndex_map);
			
			TriangleIndex_it =ConnectedTriangleIndex_map.begin();
			for (j=0; j<ConnectedTriangleIndex_map.size(); j++, TriangleIndex_it++) {
				TriIndex[0] = (*TriangleIndex_it).first;
				MatNumOfTriangle_mi[TriIndex[0]] = MatNum;
			}
			
#ifdef		DEBUG_MC_GEOM
			printf ("The # of disconnedted objects = %d, ", NumDisconnectedObjects);
			printf ("Mat # = %3d, ", MatNum);
			printf ("# Connected Triangles = %d\n", (int)ConnectedTriangleIndex_map.size());
//			if (NumDisconnectedObjects==5) break;
#endif

		}

#ifdef		DEBUG_MC_GEOM
		printf ("Progressed Triangles = %d / %d\n", i, NumTriangles_mi);
#endif		
		
	}

	printf ("The Total # of disconnedted objects = %d\n", NumDisconnectedObjects);

}


template <class _DataType>
int cMC_Geom<_DataType>::DecidingMaterialNum(map<int, unsigned char>& TriangleIndex_map)
{
	int		i, j, k, VertexIndex, TriangleIndex, loc[3];
	map<int, unsigned char>::iterator	TriangleIndex_it;
	map<int, unsigned char>::iterator	BoundaryLocs_it;
	int		*MatHittingRecords = new int [NumMaterials_mi];
	float	*Vertices = getVerticesPtsf();

	for (i=0; i<NumMaterials_mi; i++) MatHittingRecords[i] = 0;

	TriangleIndex_it = TriangleIndex_map.begin();
	for (i=0; i<TriangleIndex_map.size(); i++, TriangleIndex_it++) {
	
		TriangleIndex = (*TriangleIndex_it).first;
		for (j=0; j<3; j++) {
			VertexIndex = VertexIndex_mi[TriangleIndex*3 + j];
			
			loc[0] = ((int)(Vertices[VertexIndex*3 + 2]))*WtimesH_mi +
					 ((int)(Vertices[VertexIndex*3 + 1]))*Width_mi +
					 ((int)(Vertices[VertexIndex*3 + 0]));
		}
		
		for (k=0; k<NumMaterials_mi; k++) {
			BoundaryLocs_it = BoundaryLocs_map[k].find(loc[0]);
			if (BoundaryLocs_it!=BoundaryLocs_map[k].end()) MatHittingRecords[k]++;
		}
	}
	int		MaxHitting=-1, MaxHittingNum=-1;
	for (i=0; i<NumMaterials_mi; i++) {
		if (MatHittingRecords[i]>MaxHitting) {
			MaxHittingNum = i;
			MaxHitting = MatHittingRecords[i];
		}
	}
	
	delete [] MatHittingRecords;
	if (MaxHittingNum>0) return MaxHittingNum;
	else return MAT_NUM_DOES_NOT_EXIST;
}


template <class _DataType>
void cMC_Geom<_DataType>::BuildVertexToTriangleLink()
{
	int			i, j, k, VertexIdx;


	VertexToTriangleIndex_mi = new int [NumVertices_mi*MaxNumVToTLinks_mi];

	for (i=0; i<NumVertices_mi*MaxNumVToTLinks_mi; i++) {
		VertexToTriangleIndex_mi[i] = -1;
	}
	
	// i = triangle index
	for (i=0; i<NumTriangles_mi; i++) {
		for (j=0; j<3; j++) {
			VertexIdx = VertexIndex_mi[i*3 + j];
			
#ifdef		DEBUG_MC_GEOM
			if (VertexToTriangleIndex_mi[VertexIdx*MaxNumVToTLinks_mi + MaxNumVToTLinks_mi-1]>0) {
				printf ("MaxNumVToTLinks_mi should be bigger than %d at vertex # %d of\n", 
							MaxNumVToTLinks_mi, VertexIdx);
			}
#endif
			for (k=0; k<MaxNumVToTLinks_mi; k++) {


#ifdef		DEBUG_MC_GEOM
				if (VertexToTriangleIndex_mi[VertexIdx*MaxNumVToTLinks_mi + k]==i) {
					printf ("The triangle index(%d) is overlapped at verex %d\n", i, VertexIdx);
					continue;
				}
#endif

				if (VertexToTriangleIndex_mi[VertexIdx*MaxNumVToTLinks_mi + k]<0) {
					VertexToTriangleIndex_mi[VertexIdx*MaxNumVToTLinks_mi + k] = i;
					break;
				}
			}
		}
	}
}


template<class _DataType>
void cMC_Geom<_DataType>::setNumMaterials(int	NumMat)
{
	NumMaterials_mi = NumMat;
	Ranges_mi = new int [NumMaterials_mi*2];
}

template<class _DataType>
void cMC_Geom<_DataType>::setAMaterialRange(int MatNum, int Intensity1, int Intensity2)
{
	Ranges_mi[MatNum*2 + 0] = Intensity1;
	Ranges_mi[MatNum*2 + 1] = Intensity2;
}

// Boundary Extraction in 3D space of data value, 1st and 2nd derivative
template <class _DataType>
void cMC_Geom<_DataType>::BoundaryVoxelExtraction(int MatNum, _DataType MatMin, _DataType MatMax, 
								map<int, unsigned char>& BoundaryLocs_map)
{
	map<int, unsigned char>::iterator	Boundary_it;
	BoundaryLocs_map.clear();


	for (int i=0; i<WHD_mi; i++) {
		if (IsMaterialBoundaryUsingMinMax(i, MatMin, MatMax)) {
			BoundaryLocs_map[i] = (unsigned char)0;
		}
	}

#ifdef	DEBUG_PED_BEXT
	printf ("MC_Geom: Ajusting Boundary....\n");
	printf ("MC_Geom: Size of the boundary map = %d \n", (int)BoundaryLocs_map.size());
	fflush (stdout);
#endif

	// Finding the maximum gradient magnitudes and removing unnecessarily classified voxels
	AdjustingBoundary(BoundaryLocs_map);

#ifdef	DEBUG_PED_BEXT
	printf ("MC_Geom: The End of the Ajusting \n");
	printf ("MC_Geom: Size of the initial Boundary map = %d\n", (int)BoundaryLocs_map.size());
	fflush (stdout);
#endif

/*
	int		l, m, n;
	int			i, loc[3], DataCoor_i[3];
	
	Boundary_it = BoundaryLocs_map.begin();
	for (i=0; i<BoundaryLocs_map.size(); i++, Boundary_it++) {
		loc[0] = (*Boundary_it).first;
		DataCoor_i[2] = loc[0]/WtimesH_mi;
		DataCoor_i[1] = (loc[0] - DataCoor_i[2]*WtimesH_mi)/Width_mi;
		DataCoor_i[0] = loc[0] % Width_mi;
	
		for (n=-1; n<=1; n++) {
			for (m=-1; m<=1; m++) {
				for (l=-1; l<=1; l++) {
			
					if ((DataCoor_i[0]+l)<0 || (DataCoor_i[0]+l)>=Width_mi) continue;
					if ((DataCoor_i[1]+m)<0 || (DataCoor_i[1]+m)>=Height_mi) continue;
					if ((DataCoor_i[2]+n)<0 || (DataCoor_i[2]+n)>=Depth_mi) continue;
					loc[1] = (DataCoor_i[2]+n)*WtimesH_mi + (DataCoor_i[1]+m)*Width_mi + (DataCoor_i[0]+l);
					ZeroCrossingVolume_f[loc[1]] = SecondDerivative_mf[loc[1]];
					BoundaryLocs_map.begin()
				
				}
			}
		}
	}
*/

}


// Adjusting the boundary acoording to gradient magnitudes
// Removing the false classified voxels
template <class _DataType>
void cMC_Geom<_DataType>::AdjustingBoundary(map<int, unsigned char>& BoundaryLocs_map)
{
	int			i, j, loc[7], DataCoor_i[3];
	map<int, unsigned char>				BoundaryUntouched_map;
	map<int, unsigned char>::iterator	Boundary_it;
	int			*UntouchedCoor_i, NumUntouchedVoxels;


	UntouchedCoor_i = new int [BoundaryLocs_map.size()*3];

	// Copy the boundary map
	NumUntouchedVoxels = BoundaryLocs_map.size();
	Boundary_it = BoundaryLocs_map.begin();
	for (i=0; i<BoundaryLocs_map.size(); i++, Boundary_it++) {
		loc[0] = (*Boundary_it).first;

		DataCoor_i[2] = loc[0]/WtimesH_mi;
		DataCoor_i[1] = (loc[0] - DataCoor_i[2]*WtimesH_mi)/Width_mi;
		DataCoor_i[0] = loc[0] % Width_mi;

		UntouchedCoor_i[i*3 + 2] = DataCoor_i[2];
		UntouchedCoor_i[i*3 + 1] = DataCoor_i[1];
		UntouchedCoor_i[i*3 + 0] = DataCoor_i[0];
	}
	
	printf ("Start to find the maximum gradient magnitudes ... \n");
	printf ("# Untouched Voxels = %d\n", NumUntouchedVoxels);
	
	int		Iteration=0;

	do {

		BoundaryUntouched_map.clear();

		for (i=0; i<NumUntouchedVoxels; i++) {

			DataCoor_i[2] = UntouchedCoor_i[i*3 + 2];
			DataCoor_i[1] = UntouchedCoor_i[i*3 + 1];
			DataCoor_i[0] = UntouchedCoor_i[i*3 + 0];

			loc[1] = DataCoor_i[2]*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0]+1;
			loc[2] = DataCoor_i[2]*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0]-1;
			loc[3] = DataCoor_i[2]*WtimesH_mi + (DataCoor_i[1]+1)*Width_mi + DataCoor_i[0];
			loc[4] = DataCoor_i[2]*WtimesH_mi + (DataCoor_i[1]-1)*Width_mi + DataCoor_i[0];
			loc[5] = (DataCoor_i[2]+1)*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0];
			loc[6] = (DataCoor_i[2]-1)*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0];

			for (j=1; j<=6; j++) {
				if (GradientMag_mf[loc[j]] >  GradientMag_mf[loc[0]]) {
					BoundaryUntouched_map[loc[j]] = (unsigned char)0;
					BoundaryLocs_map[loc[j]] = (unsigned char)0;
				}
			}
		}

		delete [] UntouchedCoor_i;
		NumUntouchedVoxels = BoundaryUntouched_map.size();
		if (NumUntouchedVoxels<=0) break;
		UntouchedCoor_i = new int[NumUntouchedVoxels*3];
		
		Boundary_it = BoundaryUntouched_map.begin();
		for (i=0; i<BoundaryUntouched_map.size(); i++, Boundary_it++) {
			loc[0] = (*Boundary_it).first;

			DataCoor_i[2] = loc[0]/WtimesH_mi;
			DataCoor_i[1] = (loc[0] - DataCoor_i[2]*WtimesH_mi)/Width_mi;
			DataCoor_i[0] = loc[0] % Width_mi;

			UntouchedCoor_i[i*3 + 2] = DataCoor_i[2];
			UntouchedCoor_i[i*3 + 1] = DataCoor_i[1];
			UntouchedCoor_i[i*3 + 0] = DataCoor_i[0];
		}
	
		printf ("The # Iterations of the boundary adjusting = %d\n", Iteration++);

	} while (1);

	// Copy the boundary map
	BoundaryUntouched_map.clear();
	Boundary_it = BoundaryLocs_map.begin();
	for (i=0; i<BoundaryLocs_map.size(); i++, Boundary_it++) {
		BoundaryUntouched_map[(*Boundary_it).first] = (unsigned char)0;
	}
	BoundaryLocs_map.clear();
	
	Boundary_it = BoundaryUntouched_map.begin();
	do {
	
		loc[0] = (*Boundary_it).first;
		DataCoor_i[2] = loc[0]/WtimesH_mi;
		DataCoor_i[1] = (loc[0] - DataCoor_i[2]*WtimesH_mi)/Width_mi;
		DataCoor_i[0] = loc[0] % Width_mi;
		BoundaryUntouched_map.erase(loc[0]);
		
		loc[1] = DataCoor_i[2]*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0]+1;
		loc[2] = DataCoor_i[2]*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0]-1;
		loc[3] = DataCoor_i[2]*WtimesH_mi + (DataCoor_i[1]+1)*Width_mi + DataCoor_i[0];
		loc[4] = DataCoor_i[2]*WtimesH_mi + (DataCoor_i[1]-1)*Width_mi + DataCoor_i[0];
		loc[5] = (DataCoor_i[2]+1)*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0];
		loc[6] = (DataCoor_i[2]-1)*WtimesH_mi + DataCoor_i[1]*Width_mi + DataCoor_i[0];

		if (SecondDerivative_mf[loc[1]]*SecondDerivative_mf[loc[2]]<0 ||
			SecondDerivative_mf[loc[3]]*SecondDerivative_mf[loc[4]]<0 ||
			SecondDerivative_mf[loc[5]]*SecondDerivative_mf[loc[6]]<0) {
			BoundaryLocs_map[loc[0]] = (unsigned char)0;
		}
		
		Boundary_it = BoundaryUntouched_map.begin();
		
	} while (BoundaryUntouched_map.size()>0);
		

	printf ("The end of removing the maximum gradient magnitudes ... \n");
	
}


template <class _DataType>
int cMC_Geom<_DataType>::IsMaterialBoundaryUsingMinMax(int DataLoc, _DataType MatMin, _DataType MatMax)
{
	int			i, j, k, loc[3];
	int			XCoor, YCoor, ZCoor;
	
	
	// The given location should be between the min and max values
	if (Data_mT[DataLoc] < MatMin || MatMax < Data_mT[DataLoc]) return false;
	
	ZCoor = DataLoc / WtimesH_mi;
	YCoor = (DataLoc - ZCoor*WtimesH_mi) / Height_mi;
	XCoor = DataLoc % Width_mi;

	// Checking all 26 neighbors, whether at least one of them is a different material
	for (k=ZCoor-1; k<=ZCoor+1; k++) {
		if (k<0 || k>=Depth_mi) continue;
		for (j=YCoor-1; j<=YCoor+1; j++) {
			if (j<0 || j>=Height_mi) continue;
			for (i=XCoor-1; i<=XCoor+1; i++) {
				if (i<0 || i>=Width_mi) continue;

				loc[0] = k*WtimesH_mi + j*Width_mi + i;
				if (Data_mT[loc[0]] < MatMin || MatMax < Data_mT[loc[0]]) return true;

			}
		}
	}
	
	return false;
}



template <class _DataType>
void cMC_Geom<_DataType>::SaveMatGeometry_RAW(char *filename, int MatNum)
{
	FILE 		*fp_out;
	int			i;
	char		OutFileName[500];
	float		*Vertices = getVerticesPtsf();
	

	printf ("Saving the geometry using the raw format...\n");

	sprintf (OutFileName, "%s_Mat%02d_Geom.raw", filename, MatNum);
	fp_out = fopen(OutFileName,"w");
	if (fp_out==NULL) {
		fprintf(stderr, "Cannot open the file: %s\n", OutFileName);
		exit(1);
	}

	int	NumTriangleOfTheMaterial = 0;
	for (i=0; i<NumTriangles_mi; i++) {
		if (MatNumOfTriangle_mi[i]==MatNum) NumTriangleOfTheMaterial++;
	}

	if (NumTriangleOfTheMaterial==0) return;

//	fprintf (fp_out, "%d %d\n", NumVertices_mi, NumTriangles_mi);
	fprintf (fp_out, "%d %d\n", NumVertices_mi, NumTriangleOfTheMaterial);

	for (i=0; i<NumVertices_mi; i++) {
		fprintf (fp_out, "%f %f %f\n", Vertices[i*3 + 0], Vertices[i*3 + 1], Vertices[i*3 + 2]);
	}
	for (i=0; i<NumTriangles_mi; i++) {
		if (MatNumOfTriangle_mi[i]==MatNum) {
			fprintf (fp_out, "%d %d %d\n", VertexIndex_mi[i*3+0], 
						VertexIndex_mi[i*3+1], VertexIndex_mi[i*3+2]);
		}
	}
	fprintf (fp_out, "\n");
	fclose (fp_out);
}

	

template <class _DataType>
void cMC_Geom<_DataType>::Destroy()
{
	delete [] VertexToTriangleIndex_mi;
	VertexToTriangleIndex_mi = NULL;
	
	delete [] MatNumOfTriangle_mi;
	MatNumOfTriangle_mi = NULL;
	
	delete [] Ranges_mi;
	Ranges_mi = NULL;
	
	for (int i=0; i<MaxNumMaterials; i++) {
		BoundaryLocs_map[i].clear();
	}
}

cMC_Geom<unsigned char>		__MC_GeomValue0;
//cMC_Geom<unsigned short>	__MC_GeomValue1;
//cMC_Geom<int>				__MC_GeomValue2;
//cMC_Geom<float>				__MC_GeomValue3;
