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
#include <PEDetection/MC_Geom.h>
#include <math.h>

template <class _DataType, class _ExtraDataType>
cMC_Geom<_DataType, _ExtraDataType>::cMC_Geom() {
  VertexToTriangleIndex_mi = NULL;
  MaxNumVToTLinks_mi = 15;

  // Classification Related Variables
  NumMaterials_mi = 0;
  Ranges_mi = NULL;

  MaxNumMaterials = 30;
  for (int i = 0; i < MaxNumMaterials; i++) {
    BoundaryLocs_map[i].clear();
  }
  ZeroCrossingVolume_mf = NULL;
}

// destructor
template <class _DataType, class _ExtraDataType>
cMC_Geom<_DataType, _ExtraDataType>::~cMC_Geom() {
  delete[] VertexToTriangleIndex_mi;
  delete[] Ranges_mi;
  for (int i = 0; i < MaxNumMaterials; i++) {
    BoundaryLocs_map[i].clear();
  }
  delete[] ZeroCrossingVolume_mf;
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::IsosurfaceGenerationFromSecondD() {
  int i;

  ZeroCrossingVolume_mf = new float[this->WHD_mi];

  printf("Initializing the zero-crossing volume\n");
  for (i = 0; i < this->WHD_mi; i++)
    ZeroCrossingVolume_mf[i] = this->MinSecond_mf - 1.0;
  //	for (i=0; i<this->WHD_mi; i++) ZeroCrossingVolume_mf[i] =
  //this->MaxSecond_mf;

  for (i = 0; i < NumMaterials_mi; i++) {
    // Computing Boundary Voxels from the classification results and
    // Saving the second derivatives at the boundary voxel locations
    // to ZeroCrossingVolume_mf[]
    BoundaryVoxelExtraction(i, Ranges_mi[i * 2], Ranges_mi[i * 2 + 1],
                            BoundaryLocs_map[i]);
  }

  // New data assignment
  //	this->Data_mT = ZeroCrossingVolume_mf;

  /*
          // Quantization Error
          for (i=0; i<this->WHD_mi; i++) ZeroCrossingVolume_mf[i] =
     this->SecondDerivative_mf[i]; int		Tempi; for (i=0;
     i<this->WHD_mi; i++) { Tempi = (int)((this->SecondDerivative_mf[i] -
     this->MinSecond_mf)/(this->MaxSecond_mf - this->MinSecond_mf)*255.0);
                  ZeroCrossingVolume_mf[i] = Tempi;
          }
          float		IsoValue;
          Tempi = (int)((0.0 - this->MinSecond_mf)/(this->MaxSecond_mf -
     this->MinSecond_mf)*255.0); IsoValue = Tempi;
          ExtractingIsosurfacesFromSecondD(IsoValue, Ranges_mi[i*2],
     Ranges_mi[i*2 + 1]);
  */

  // Extracting Iso-surfaces
  ExtractingIsosurfacesFromSecondD((float)0.0, Ranges_mi[i * 2],
                                   Ranges_mi[i * 2 + 1]);
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::ExtractingIsosurfacesFromSecondD(
    float IsoValue, float RangeBegin, float RangeEnd) {
  int i, j, k, n, loc[8], ConfigIndex;
  int NumAmbiguities;
  _DataType DataCube[8];

#ifdef DEBUG_MC
  printf("Start to Extract Isosurfaces...\n");
  fflush(stdout);
#endif
  printf("RangeBegin = %f RangeEnd = %f\n", RangeBegin, RangeEnd);

  this->IsoValue_mf = IsoValue;
  for (k = 0; k < this->Depth_mi - 1; k++) {     // Z
    for (j = 0; j < this->Height_mi - 1; j++) {  // Y
      for (i = 0; i < this->Width_mi - 1; i++) { // X

        loc[0] = k * this->WtimesH_mi + (j + 1) * this->Width_mi + i;
        loc[1] = k * this->WtimesH_mi + (j + 1) * this->Width_mi + i + 1;
        loc[2] = k * this->WtimesH_mi + j * this->Width_mi + i;
        loc[3] = k * this->WtimesH_mi + j * this->Width_mi + i + 1;

        loc[4] = (k + 1) * this->WtimesH_mi + (j + 1) * this->Width_mi + i;
        loc[5] =
            (k + 1) * this->WtimesH_mi + (j + 1) * this->Width_mi + i + 1;
        loc[6] = (k + 1) * this->WtimesH_mi + j * this->Width_mi + i;
        loc[7] = (k + 1) * this->WtimesH_mi + j * this->Width_mi + i + 1;
        ConfigIndex = 0;
        for (n = 7; n >= 0; n--) {
          ConfigIndex <<= 1;
          DataCube[n] = this->Data_mT[loc[n]];
          if (this->Data_mT[loc[n]] >= IsoValue)
            ConfigIndex |= 1;
        }

        NumAmbiguities = ConfigurationTable[ConfigIndex][0][1];
        ComputeAndAddTriangles(ConfigIndex, i, j, k, DataCube);
        /*
        if (NumAmbiguities==0) ComputeAndAddTriangles(ConfigIndex, i, j, k,
        DataCube); else {

                int AmbiguityCase = CheckAmbiguity(ConfigIndex, DataCube);

                ComputeAndAddTriangles(ConfigIndex, i, j, k, DataCube,
                                                        LevelOneUnConfiguration_gi[ConfigIndex],
        AmbiguityCase);

        }
        */
      }
    }
    this->CopyVertexBuffer();
  }

  this->ComputeVertexNormals(); // using three vertices and counter-clock wise
                                // direction from the top
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::CheckAmbiguity(int ConfigIndex,
                                                         _DataType DataCube) {
  int i, LevelOneConfig;
  int ConnectedAmbigousSurfaces[6];

  printf("DataCube = %f\n", (float)DataCube);
  // Initializing the variable, ConnectedAmbigousSurfaces[].
  // 1=connected, 0=not-connecdted, -1=Don't need to be considered
  for (i = 0; i < 6; i++)
    ConnectedAmbigousSurfaces[i] = -1;
  LevelOneConfig = LevelOneUnConfiguration_gi[ConfigIndex];
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::setNumMaterials(int NumMat) {
  NumMaterials_mi = NumMat;
  Ranges_mi = new int[NumMaterials_mi * 2];
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::setAMaterialRange(int MatNum,
                                                            int Intensity1,
                                                            int Intensity2) {
  Ranges_mi[MatNum * 2 + 0] = Intensity1;
  Ranges_mi[MatNum * 2 + 1] = Intensity2;
}

// Boundary Extraction in 3D space of data value, 1st and 2nd derivative
template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::BoundaryVoxelExtraction(
    int MatNum, _DataType MatMin, _DataType MatMax,
    map<int, unsigned char> &BoundaryLocs_map) {
  int i, j, loc[8], DataCoor_i[3];

  printf("BoundaryLocs_map.Size() = %d\n", (int)BoundaryLocs_map.size());

  // Finding the initial boundary from the classification result
  cStack<int> StackInitBoundaryLocs;
  StackInitBoundaryLocs.Clear();

  for (i = 0; i < this->WHD_mi; i++) {
    if (IsMaterialBoundaryUsingMinMax(i, MatMin, MatMax)) {

      DataCoor_i[2] = i / this->WtimesH_mi;
      DataCoor_i[1] = (i - DataCoor_i[2] * this->WtimesH_mi) / this->Width_mi;
      DataCoor_i[0] = i % this->Width_mi;

      if (DataCoor_i[2] >= 24 && DataCoor_i[2] <= 26 && DataCoor_i[1] >= 96 &&
          DataCoor_i[1] <= 104 && DataCoor_i[0] >= 111 &&
          DataCoor_i[0] <= 121) {
      } else
        continue;

      StackInitBoundaryLocs.Push(i);
    }
  }

  //--------------------------------------------------------------------
  // Saving the Initial Boundary Volume
  unsigned char *InitBoundaryVolume = new unsigned char[this->WHD_mi];
  for (j = 0; j < this->WHD_mi; j++)
    InitBoundaryVolume[j] = (unsigned char)0;

  for (j = 0; j < StackInitBoundaryLocs.Size(); j++) {
    StackInitBoundaryLocs.IthValue(j, loc[0]);
    InitBoundaryVolume[loc[0]] = (unsigned char)255;
  }
  printf("Saving the Initial Boundary Volume \n");
  char Postfix[200];
  sprintf(Postfix, "Mat%d_InitBoundary", MatNum);
  SaveVolume(InitBoundaryVolume, 0, 255, Postfix); // Control.cpp
  printf("The End of the Saving\n");
  delete[] InitBoundaryVolume;
  //--------------------------------------------------------------------

#ifdef DEBUG_MC_GEOM
  printf("MC_Geom: Searching the Boundary....\n");
  printf("MC_Geom: Size of the initial boundary map = %d \n",
         (int)StackInitBoundaryLocs.Size());
  fflush(stdout);
#endif

  // Finding the maximum gradient magnitudes and removing unnecessarily
  // classified voxels
  SearchingBoundary(StackInitBoundaryLocs);

#ifdef DEBUG_MC_GEOM
  printf("MC_Geom: The End of the Searching \n");
  fflush(stdout);
#endif

  if (ZeroCrossingVolume_mf == NULL) {
    printf("ZeroCrossingVolume_mf is NULL\n");
    exit(1);
  }
  if (this->SecondDerivative_mf == NULL) {
    printf("SecondDerivative_mf is NULL\n");
    exit(1);
  }

  //--------------------------------------------------------------------
  // Saving the Searched Final Boundary Volume
  unsigned char *FinalBoundaryVolume = new unsigned char[this->WHD_mi];
  for (j = 0; j < this->WHD_mi; j++)
    FinalBoundaryVolume[j] = (unsigned char)0;
  for (j = 0; j < this->WHD_mi; j++) {
    if (ZeroCrossingVolume_mf[j] >= this->MinSecond_mf)
      FinalBoundaryVolume[j] = 255;
  }
  printf("Saving the Final Boundary Volume \n");
  char Postfix2[200];
  sprintf(Postfix2, "Mat%d_FinalBoundary", MatNum);
  SaveVolume(FinalBoundaryVolume, 0, 255, Postfix2); // Control.cpp
  printf("The End of the Saving\n");
  delete[] FinalBoundaryVolume;
  //--------------------------------------------------------------------
}

int EdgeListT[12][2] = {{0, 1}, {2, 3}, {6, 7}, {4, 5}, // Around the X Axis
                        {0, 4}, {1, 5}, {3, 7}, {2, 6},
                        {0, 2}, {1, 3}, {5, 7}, {4, 6}};

int EdgeToNeighborT[8][8][3] = {
    {
        {
            0,
            0,
            0,
        },
        {
            10,
            1,
            4,
        },
        {
            4,
            3,
            12,
        },
        {
            0,
            0,
            0,
        },
        {
            10,
            9,
            12,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            10,
            1,
            4,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            4,
            5,
            14,
        },
        {
            0,
            0,
            0,
        },
        {
            10,
            17,
            14,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            4,
            3,
            12,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            4,
            7,
            16,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            12,
            15,
            16,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            4,
            5,
            14,
        },
        {
            4,
            7,
            16,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            14,
            17,
            16,
        },
    },
    {
        {
            10,
            9,
            12,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            22,
            19,
            10,
        },
        {
            12,
            21,
            22,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            10,
            17,
            14,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            22,
            19,
            10,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            14,
            23,
            22,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            12,
            15,
            16,
        },
        {
            0,
            0,
            0,
        },
        {
            12,
            21,
            22,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            16,
            25,
            22,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            14,
            17,
            16,
        },
        {
            0,
            0,
            0,
        },
        {
            14,
            23,
            22,
        },
        {
            16,
            25,
            22,
        },
        {
            0,
            0,
            0,
        },
    },
};

int FaceIndexT[6][5] = {{1, 0, 4, 5, 1}, {0, 2, 6, 4, 0}, {7, 6, 2, 3, 7},
                        {1, 5, 7, 3, 1}, {5, 4, 6, 7, 5}, {1, 3, 2, 0, 1}};

int EdgeDirT[8][8][3] = {
    {
        {
            0,
            0,
            0,
        },
        {
            1,
            0,
            0,
        },
        {
            0,
            1,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            1,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            -1,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            1,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            1,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            0,
            -1,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            1,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            1,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            0,
            -1,
            0,
        },
        {
            -1,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            1,
        },
    },
    {
        {
            0,
            0,
            -1,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            1,
            0,
            0,
        },
        {
            0,
            1,
            0,
        },
        {
            0,
            0,
            0,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            -1,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            -1,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            1,
            0,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            -1,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            -1,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            1,
            0,
            0,
        },
    },
    {
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            0,
            -1,
        },
        {
            0,
            0,
            0,
        },
        {
            0,
            -1,
            0,
        },
        {
            -1,
            0,
            0,
        },
        {
            0,
            0,
            0,
        },
    },
};

int RelativeLocT[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                          {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

// ----------------------------------------------------------------------
// Searching the boundary of zero-crossing locations by shooting a ray
// from the initial boundary voxels to the gradient directions
// Input:	Initial Boundary Voxels
// Output:	Searched Boundary Voxels which contain the zero-crossings
//
// ----------------------------------------------------------------------
template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::SearchingBoundary(
    cStack<int> &StackInitBLocs) {
  int i, j, k, l, m, n, loc[8], DataCoor_i[3], EI[2], NI[3];
  double CurrLoc_d[3], GradVec_d[3], ZeroCrossingLoc_d[3], Length_d;
  double FirstDAtTheLoc_d, DataPosFromZeroCrossingLoc_d;
  int NeighborsLocs[28], Num_i, IsFound;
  cStack<int> StackUntouchedLocs;
  double SD_d[5], Pt_d[3], ZCPt_d[3], t_d;

  //	StackInitBLocs.setDataPointer(10);

  StackUntouchedLocs.Clear();

  printf("Computing the Real Boundary \n");
  fflush(stdout);

  // Computing and Finding Real Boundary using the boundary from the
  // classification
  for (i = 0; i < StackInitBLocs.Size(); i++) {
    StackInitBLocs.IthValue(i, loc[0]);
    DataCoor_i[2] = loc[0] / this->WtimesH_mi;
    DataCoor_i[1] =
        (loc[0] - DataCoor_i[2] * this->WtimesH_mi) / this->Width_mi;
    DataCoor_i[0] = loc[0] % this->Width_mi;
    for (k = 0; k < 3; k++) {
      CurrLoc_d[k] = (double)DataCoor_i[k];
      GradVec_d[k] = (double)this->GradientVec_mf[loc[0] * 3 + k];
    }
    Length_d =
        sqrt(GradVec_d[0] * GradVec_d[0] + GradVec_d[1] * GradVec_d[1] +
             GradVec_d[2] * GradVec_d[2]);
    for (k = 0; k < 3; k++)
      GradVec_d[k] /= Length_d;

    IsFound = this->PED_m.FindZeroCrossingLocation(
        CurrLoc_d, GradVec_d, ZeroCrossingLoc_d, FirstDAtTheLoc_d,
        DataPosFromZeroCrossingLoc_d);
    if (!IsFound)
      continue;
    for (k = 0; k < 3; k++)
      DataCoor_i[k] = (int)ZeroCrossingLoc_d[k];

    ZCPt_d[0] = ZeroCrossingLoc_d[0];
    ZCPt_d[1] = ZeroCrossingLoc_d[1];
    ZCPt_d[2] = ZeroCrossingLoc_d[2];

    /*
    int FaceIndexT [6][5] = {
            {1, 0, 4, 5, 1}, {0, 2, 6, 4, 0}, {7, 6, 2, 3, 7},
            {1, 5, 7, 3, 1}, {5, 4, 6, 7, 5}, {1, 3, 2, 0, 1}
    };
    */
    if (DataCoor_i[2] >= 24 && DataCoor_i[2] <= 26 && DataCoor_i[1] >= 96 &&
        DataCoor_i[1] <= 104 && DataCoor_i[0] >= 111 &&
        DataCoor_i[0] <= 121) {
    } else
      continue;

    MarkingVoxelEdges(DataCoor_i, GradVec_d, ZCPt_d);
    StackUntouchedLocs.Push(loc[0]);
  }

  //------------------------------------------------------------------------------
  // Saving the Computed Real Boundary Volume
  unsigned char *ComputedBoundaryVolume = new unsigned char[this->WHD_mi];
  for (j = 0; j < this->WHD_mi; j++)
    ComputedBoundaryVolume[j] = (unsigned char)0;
  printf("\n# Computed Boundary Voxels = %d\n", StackUntouchedLocs.Size());
  for (j = 0; j < StackUntouchedLocs.Size(); j++) {
    if (!StackUntouchedLocs.IthValue(j, loc[0])) {
      printf("Stack Error\n");
      break;
    }

    DataCoor_i[2] = loc[0] / this->WtimesH_mi;
    DataCoor_i[1] =
        (loc[0] - DataCoor_i[2] * this->WtimesH_mi) / this->Width_mi;
    DataCoor_i[0] = loc[0] % this->Width_mi;

    loc[1] = DataCoor_i[2] * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + (DataCoor_i[0] + 1);
    loc[2] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
    loc[3] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + (DataCoor_i[0] + 1);

    loc[4] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0];
    loc[5] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + (DataCoor_i[0] + 1);
    loc[6] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
    loc[7] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + (DataCoor_i[0] + 1);

    if (DataCoor_i[0] >= this->Width_mi)
      loc[0] = loc[2] = loc[4] = loc[6] = 0;
    if (DataCoor_i[1] >= this->Height_mi)
      loc[0] = loc[1] = loc[4] = loc[5] = 0;
    if (DataCoor_i[2] >= this->Depth_mi)
      loc[0] = loc[1] = loc[2] = loc[3] = 0;

    if (DataCoor_i[0] + 1 >= this->Width_mi)
      loc[1] = loc[3] = loc[5] = loc[7] = 0;
    if (DataCoor_i[1] + 1 >= this->Height_mi)
      loc[2] = loc[3] = loc[6] = loc[7] = 0;
    if (DataCoor_i[2] + 1 >= this->Depth_mi)
      loc[4] = loc[5] = loc[6] = loc[7] = 0;

    for (k = 0; k < 8; k++)
      ComputedBoundaryVolume[loc[k]] = (unsigned char)255;
  }
  printf("Saving the Computed Boundary Volume \n");
  char Postfix[200];
  sprintf(Postfix, "Mat00_ComputedBoundary");
  SaveVolume(ComputedBoundaryVolume, 0, 255, Postfix); // Control.cpp
  printf("The End of the Saving\n\n");
  fflush(stdout);
  delete[] ComputedBoundaryVolume;
  // The End of the Saving
  //------------------------------------------------------------------------------

  int Count = 0, NewVoxel;
  ZeroCrossingVolume_mf[0] = this->MinSecond_mf + 1.0;
  printf("Searching the connected boundary from the computed boundary\n");
  // Searching the connected boundary from the computed boundary
  do {

    if (!StackUntouchedLocs.Pop(loc[0]))
      break;
    DataCoor_i[2] = loc[0] / this->WtimesH_mi;
    DataCoor_i[1] =
        (loc[0] - DataCoor_i[2] * this->WtimesH_mi) / this->Width_mi;
    DataCoor_i[0] = loc[0] % this->Width_mi;

    if (DataCoor_i[2] >= 24 && DataCoor_i[2] <= 26 && DataCoor_i[1] >= 96 &&
        DataCoor_i[1] <= 104 && DataCoor_i[0] >= 111 &&
        DataCoor_i[0] <= 121) {
    } else
      continue;

    for (n = 0; n < 27; n++)
      NeighborsLocs[n] = 0;
    Num_i = 0;
    for (n = DataCoor_i[2] - 1; n <= DataCoor_i[2] + 1; n++) {
      if (n < 0 || n >= this->Depth_mi - 2)
        continue;
      for (m = DataCoor_i[1] - 1; m <= DataCoor_i[1] + 1; m++) {
        if (m < 0 || m >= this->Height_mi - 2)
          continue;
        for (l = DataCoor_i[0] - 1; l <= DataCoor_i[0] + 1; l++) {
          if (l < 0 || l >= this->Width_mi - 2)
            continue;
          loc[1] = n * this->WtimesH_mi + m * this->Width_mi + l;
          NeighborsLocs[Num_i++] = loc[1];
        }
      }
    }

    loc[1] = DataCoor_i[2] * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + (DataCoor_i[0] + 1);
    loc[2] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
    loc[3] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + (DataCoor_i[0] + 1);

    loc[4] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0];
    loc[5] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + (DataCoor_i[0] + 1);
    loc[6] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
    loc[7] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + (DataCoor_i[0] + 1);

    if (DataCoor_i[0] >= this->Width_mi)
      loc[0] = loc[2] = loc[4] = loc[6] = 0;
    if (DataCoor_i[1] >= this->Height_mi)
      loc[0] = loc[1] = loc[4] = loc[5] = 0;
    if (DataCoor_i[2] >= this->Depth_mi)
      loc[0] = loc[1] = loc[2] = loc[3] = 0;

    if (DataCoor_i[0] + 1 >= this->Width_mi)
      loc[1] = loc[3] = loc[5] = loc[7] = 0;
    if (DataCoor_i[1] + 1 >= this->Height_mi)
      loc[2] = loc[3] = loc[6] = loc[7] = 0;
    if (DataCoor_i[2] + 1 >= this->Depth_mi)
      loc[4] = loc[5] = loc[6] = loc[7] = 0;

    // ZeroCrossingVolume_mf[] is initialized with (this->MinSecond_mf-1.0)
    for (i = 0; i < 12; i++) {
      EI[0] = EdgeListT[i][0];
      EI[1] = EdgeListT[i][1];
      SD_d[0] = ZeroCrossingVolume_mf[loc[EI[0]]];
      SD_d[1] = ZeroCrossingVolume_mf[loc[EI[1]]];

      if (SD_d[0] * SD_d[1] <= 0.0) {

        for (k = 0; k < 3; k++)
          NI[k] = EdgeToNeighborT[EI[0]][EI[1]][k];
        for (m = 0; m < 3; m++) {
          if (NeighborsLocs[NI[m]] > 0) {

            DataCoor_i[2] = NeighborsLocs[NI[m]] / this->WtimesH_mi;
            DataCoor_i[1] =
                (NeighborsLocs[NI[m]] - DataCoor_i[2] * this->WtimesH_mi) /
                this->Width_mi;
            DataCoor_i[0] = NeighborsLocs[NI[m]] % this->Width_mi;

            if (DataCoor_i[0] >= this->Width_mi - 1)
              continue;
            if (DataCoor_i[1] >= this->Height_mi - 1)
              continue;
            if (DataCoor_i[2] >= this->Depth_mi - 1)
              continue;

            //						printf ("DataCoor =
            //%3d %3d %3d\n", DataCoor_i[0], DataCoor_i[1], DataCoor_i[2]);

            t_d = fabs(SD_d[0]) / fabs(SD_d[0] - SD_d[1]);
            for (k = 0; k < 3; k++) {
              Pt_d[k] = (double)DataCoor_i[k];
              Pt_d[k] += (double)RelativeLocT[EI[0]][k];
              Pt_d[k] += (double)EdgeDirT[EI[0]][EI[1]][k] * t_d;
            }

            //						printf ("Pt_d = %.2f
            //%.2f %.2f\n", Pt_d[0], Pt_d[1], Pt_d[2]);
            this->GradVecInterpolation(Pt_d, GradVec_d);

            NewVoxel = MarkingVoxelEdges(DataCoor_i, GradVec_d, Pt_d);
            if (NewVoxel)
              StackUntouchedLocs.Push(NeighborsLocs[NI[m]]);

#ifdef DEBUG_MC_GEOM
            if (NewVoxel) {
              printf("DataCoor = %3d %3d %3d, ", DataCoor_i[0], DataCoor_i[1],
                     DataCoor_i[2]);
              printf("GradVec = %7.2f %7.2f %7.2f, ", GradVec_d[0],
                     GradVec_d[1], GradVec_d[2]);
              printf("This voxel is marked ");
              printf("\n\n");
            }
#endif
          }
        }
      }
    }

    if (Count++ % 10000 == 0) {
      printf("The Size of StackUntouchedLocs = %d\n",
             StackUntouchedLocs.Size());
      fflush(stdout);
    }

  } while (StackUntouchedLocs.Size() > 0);
  printf("The searching is done\n\n");
  fflush(stdout);

  StackUntouchedLocs.Destroy();
}

// Checking ambiguity and Marcking the voxel edges
template <class _DataType, class _ExtraDataType>
int cMC_Geom<_DataType, _ExtraDataType>::MarkingVoxelEdges(
    int *DataCoor3, double *GradVec3, double *ZeroCrossingPt3) {
  int i, j, k, ConfigIndex, loc[8], Processed;
  double SD_d[5], Pt_d[3], Vec_d[4][3], DotP_d[4], t_d[4], Length_d;
  double MaxDotP_d, B_st;
  int MaxDotPI_i, CongifIndex;

  loc[0] = DataCoor3[2] * this->WtimesH_mi + DataCoor3[1] * this->Width_mi +
           DataCoor3[0];
  loc[1] = DataCoor3[2] * this->WtimesH_mi + DataCoor3[1] * this->Width_mi +
           (DataCoor3[0] + 1);
  loc[2] = DataCoor3[2] * this->WtimesH_mi +
           (DataCoor3[1] + 1) * this->Width_mi + DataCoor3[0];
  loc[3] = DataCoor3[2] * this->WtimesH_mi +
           (DataCoor3[1] + 1) * this->Width_mi + (DataCoor3[0] + 1);

  loc[4] = (DataCoor3[2] + 1) * this->WtimesH_mi +
           DataCoor3[1] * this->Width_mi + DataCoor3[0];
  loc[5] = (DataCoor3[2] + 1) * this->WtimesH_mi +
           DataCoor3[1] * this->Width_mi + (DataCoor3[0] + 1);
  loc[6] = (DataCoor3[2] + 1) * this->WtimesH_mi +
           (DataCoor3[1] + 1) * this->Width_mi + DataCoor3[0];
  loc[7] = (DataCoor3[2] + 1) * this->WtimesH_mi +
           (DataCoor3[1] + 1) * this->Width_mi + (DataCoor3[0] + 1);

  // Copying the second derivatives to the zero-crossing volume
  Processed = true;
  for (k = 0; k < 8; k++) {
    if (ZeroCrossingVolume_mf[loc[k]] < this->MinSecond_mf) {
      ZeroCrossingVolume_mf[loc[k]] = this->SecondDerivative_mf[loc[k]];
      Processed = false;
    }
  }
  if (Processed)
    return false;
/*
int FaceIndexT[6][5] = {
        {1, 0, 4, 5, 1}, {0, 2, 6, 4, 0}, {7, 6, 2, 3, 7}, {1, 5, 7, 3, 1},
{5, 4, 6, 7, 5}, {1, 3, 2, 0, 1}
};
*/
#ifdef DEBUG_MC_GEOM
  int m;
  printf("Zero-Crossing Volume = ");
  for (i = 0; i < 8; i++) {
    printf("%7.2f ", ZeroCrossingVolume_mf[loc[i]]);
  }
  printf("\n");

#endif

  // Checking the Ambiguity
  for (i = 0; i < 6; i++) { // i = Face Index
    ConfigIndex = 0;
    for (j = 3; j >= 0; j--) {
      ConfigIndex <<= 1;
      SD_d[j] = (double)ZeroCrossingVolume_mf[loc[FaceIndexT[i][j]]];
      if (SD_d[j] >= 0)
        ConfigIndex |= 1;
      else
        ConfigIndex |= 0;
    }
    SD_d[4] = SD_d[0];

    // 10 = Binary(1010), 5 = Binary(0101)
    if (ConfigIndex == 10 || CongifIndex == 5) {
      // The case is ambiguous

      for (k = 0; k < 4; k++)
        t_d[k] = fabs(SD_d[k]) / fabs(SD_d[k] - SD_d[k + 1]);
      for (j = 0; j < 4; j++) {
        for (k = 0; k < 3; k++) {
          Pt_d[k] = (double)DataCoor3[k];
          Pt_d[k] += (double)RelativeLocT[FaceIndexT[i][j]][k];
          Pt_d[k] +=
              (double)EdgeDirT[FaceIndexT[i][j]][FaceIndexT[i][j + 1]][k] *
              t_d[j];
        }
        for (k = 0; k < 3; k++)
          Vec_d[j][k] = Pt_d[k] - ZeroCrossingPt3[k];
        Length_d =
            sqrt(Vec_d[j][0] * Vec_d[j][0] + Vec_d[j][1] * Vec_d[j][1] +
                 Vec_d[j][2] * Vec_d[j][2]);
        for (k = 0; k < 3; k++)
          Vec_d[j][k] /= Length_d;

#ifdef DEBUG_MC_GEOM
/*
                                printf ("Pt_d = ");
                                for (m=0; m<3; m++) printf ("%.4f ", Pt_d[m]);
                                printf ("Vec = ");
                                for (m=0; m<3; m++) printf ("%7.4f ",
   Vec_d[j][m]); printf ("\n");
*/
#endif
      }

      B_st = (SD_d[0] * SD_d[3] + SD_d[2] * SD_d[1]) /
             (SD_d[0] + SD_d[3] - SD_d[1] - SD_d[2]);
      if (B_st < 0) { // Separated
        for (j = 0; j < 4; j++)
          DotP_d[j] = 0.0;
        for (j = 0; j < 4; j++) {
          for (k = 0; k < 3; k++)
            DotP_d[j] += GradVec3[k] * Vec_d[j][k];
          DotP_d[j] = fabs(DotP_d[j]);
        }
        MaxDotP_d = -FLT_MAX;
        for (j = 0; j < 4; j++) {
          if (MaxDotP_d < DotP_d[j]) {
            MaxDotP_d = DotP_d[j];
            MaxDotPI_i = j;
          }
        }

#ifdef DEBUG_MC_GEOM
        printf("ZeroC = ");
        for (m = 0; m < 8; m++)
          printf("%7.2f ", ZeroCrossingVolume_mf[loc[m]]);
        printf("DataCoor = %3d %3d %3d, ", DataCoor3[0], DataCoor3[1],
               DataCoor3[2]);
        printf("2ndD = ");
        for (m = 0; m < 4; m++)
          printf("%7.2f ", SD_d[m]);
        printf("\n");

#endif

        if (ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i]]] > 0)
          ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i]]] =
              this->MinSecond_mf;
        if (ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i + 1]]] > 0)
          ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i]]] =
              this->MinSecond_mf;

#ifdef DEBUG_MC_GEOM
        printf("ZeroC = ");
        for (m = 0; m < 8; m++)
          printf("%7.2f ", ZeroCrossingVolume_mf[loc[m]]);
        printf("\n");

        printf("ZeroPt = %7.2f %7.2f %7.2f, ", ZeroCrossingPt3[0],
               ZeroCrossingPt3[1], ZeroCrossingPt3[2]);
        printf("GradV = %.4f %.4f %.4f, ", GradVec3[0], GradVec3[1],
               GradVec3[2]);
        printf("DotP = ");
        for (m = 0; m < 4; m++)
          printf("%.7f ", DotP_d[m]);
        printf("MaxDotPI_i = %d, ", MaxDotPI_i);
        printf("\n");
        printf("B_st = %7.2f, ", B_st);
        printf("\n");
        printf("\n");
#endif

      } else { // Not Separated

        for (j = 0; j < 4; j++)
          DotP_d[j] = 0.0;
        for (j = 0; j < 4; j++) {
          for (k = 0; k < 3; k++)
            DotP_d[j] += fabs(GradVec3[k] * Vec_d[j][k]);
        }
        MaxDotP_d = -FLT_MAX;
        for (j = 0; j < 4; j++) {
          if (MaxDotP_d < DotP_d[j]) {
            MaxDotP_d = DotP_d[j];
            MaxDotPI_i = j;
          }
        }
        if (ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i]]] < 0)
          ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i]]] =
              this->MaxSecond_mf;
        if (ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i + 1]]] < 0)
          ZeroCrossingVolume_mf[loc[FaceIndexT[i][MaxDotPI_i]]] =
              this->MaxSecond_mf;
      }
    } else {
      // The case is Not ambiguous
    }
  }
  return true;
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::SearchingBoundary2(
    map<int, unsigned char> &BoundaryLocs_map) {
  int i, k, l, m, n, loc[8], DataCoor_i[3], EI[2], NI[3];
  map<int, unsigned char> BoundaryUntouched_map;
  map<int, unsigned char>::iterator Boundary_it;
  double CurrLoc_d[3], GradVec_d[3], ZeroCrossingLoc_d[3];
  double FirstDAtTheLoc_d, DataPosFromZeroCrossingLoc_d;
  int NeighborsLocs[28], Num_i, IsFound;

  printf("Computing the Real Boundary \n");
  fflush(stdout);
  // Computing and Finding Real Boundary using the boundary from the
  // classification
  BoundaryUntouched_map.clear();
  Boundary_it = BoundaryLocs_map.begin();
  for (i = 0; i < (int)BoundaryLocs_map.size(); i++, Boundary_it++) {
    loc[0] = (*Boundary_it).first;
    DataCoor_i[2] = loc[0] / this->WtimesH_mi;
    DataCoor_i[1] =
        (loc[0] - DataCoor_i[2] * this->WtimesH_mi) / this->Width_mi;
    DataCoor_i[0] = loc[0] % this->Width_mi;
    for (k = 0; k < 3; k++) {
      CurrLoc_d[k] = (double)DataCoor_i[k];
      GradVec_d[k] = (double)this->GradientVec_mf[loc[0] * 3 + k];
    }
    IsFound = this->PED_m.FindZeroCrossingLocation(
        CurrLoc_d, GradVec_d, ZeroCrossingLoc_d, FirstDAtTheLoc_d,
        DataPosFromZeroCrossingLoc_d);
    if (!IsFound)
      continue;
    for (k = 0; k < 3; k++) {
      DataCoor_i[k] =
          (int)(CurrLoc_d[k] + GradVec_d[k] * DataPosFromZeroCrossingLoc_d);
    }

    loc[1] = DataCoor_i[2] * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0];
    BoundaryUntouched_map[loc[1]] = (unsigned char)FirstDAtTheLoc_d;
  }

  printf("Copying the computed real boundary to boundary map buffer\n");
  // Copying BoundaryUntouched_map to BoundaryLocs_map
  BoundaryLocs_map.clear();
  Boundary_it = BoundaryUntouched_map.begin();
  for (i = 0; i < (int)BoundaryUntouched_map.size(); i++, Boundary_it++) {
    BoundaryLocs_map[(*Boundary_it).first] = (unsigned char)0;
  }

  int Count = 0;
  ZeroCrossingVolume_mf[0] = this->MinSecond_mf + 1.0;
  printf("Searching2 the connected boundary from the computed boundary\n");
  // Searching2 the connected boundary from the computed boundary
  do {

    Boundary_it = BoundaryUntouched_map.begin();
    loc[0] = (*Boundary_it).first;
    DataCoor_i[2] = loc[0] / this->WtimesH_mi;
    DataCoor_i[1] =
        (loc[0] - DataCoor_i[2] * this->WtimesH_mi) / this->Width_mi;
    DataCoor_i[0] = loc[0] % this->Width_mi;
    BoundaryUntouched_map.erase(loc[0]);

    for (n = 0; n < 27; n++)
      NeighborsLocs[n] = 0;
    Num_i = 0;
    for (n = DataCoor_i[2] - 1; n <= DataCoor_i[2] + 1; n++) {
      if (n < 0 || n >= this->Depth_mi - 2)
        continue;
      for (m = DataCoor_i[1] - 1; m <= DataCoor_i[1] + 1; m++) {
        if (m < 0 || m >= this->Height_mi - 2)
          continue;
        for (l = DataCoor_i[0] - 1; l <= DataCoor_i[0] + 1; l++) {
          if (l < 0 || l >= this->Width_mi - 2)
            continue;
          loc[1] = n * this->WtimesH_mi + m * this->Width_mi + l;
          NeighborsLocs[Num_i++] = loc[1];
        }
      }
    }

    loc[1] = DataCoor_i[2] * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + (DataCoor_i[0] + 1);
    loc[2] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
    loc[3] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + (DataCoor_i[0] + 1);

    loc[4] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0];
    loc[5] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + (DataCoor_i[0] + 1);
    loc[6] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
    loc[7] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + (DataCoor_i[0] + 1);

    if (DataCoor_i[0] >= this->Width_mi)
      loc[0] = loc[2] = loc[4] = loc[6] = 0;
    if (DataCoor_i[1] >= this->Height_mi)
      loc[0] = loc[1] = loc[4] = loc[5] = 0;
    if (DataCoor_i[2] >= this->Depth_mi)
      loc[0] = loc[1] = loc[2] = loc[3] = 0;

    if (DataCoor_i[0] + 1 >= this->Width_mi)
      loc[1] = loc[3] = loc[5] = loc[7] = 0;
    if (DataCoor_i[1] + 1 >= this->Height_mi)
      loc[2] = loc[3] = loc[6] = loc[7] = 0;
    if (DataCoor_i[2] + 1 >= this->Depth_mi)
      loc[4] = loc[5] = loc[6] = loc[7] = 0;

    if ((int)BoundaryUntouched_map.size() <= 13646) {
      printf("(%d %d %d) ", DataCoor_i[0], DataCoor_i[1], DataCoor_i[2]);
      fflush(stdout);
    }

    // ZeroCrossingVolume_mf[] is initialized with (this->MinSecond_mf-1.0)
    for (i = 0; i < 12; i++) {
      EI[0] = EdgeListT[i][0];
      EI[1] = EdgeListT[i][1];
      if ((ZeroCrossingVolume_mf[loc[EI[0]]] < this->MinSecond_mf ||
           ZeroCrossingVolume_mf[loc[EI[1]]] < this->MinSecond_mf) &&
          this->SecondDerivative_mf[loc[EI[0]]] *
                  this->SecondDerivative_mf[loc[EI[1]]] <=
              0.0) {

        ZeroCrossingVolume_mf[loc[EI[0]]] =
            this->SecondDerivative_mf[loc[EI[0]]];
        ZeroCrossingVolume_mf[loc[EI[1]]] =
            this->SecondDerivative_mf[loc[EI[1]]];
        for (k = 0; k < 3; k++)
          NI[k] = EdgeToNeighborT[EI[0]][EI[1]][k];
        for (k = 0; k < 3; k++) {
          if (NeighborsLocs[NI[k]] > 0)
            BoundaryUntouched_map[NeighborsLocs[NI[k]]] = (unsigned char)0;
        }
      }
    }

    if (Count++ % 10000 == 0) {
      printf("The Size of BoundaryUntouched_map = %d\n",
             (int)BoundaryUntouched_map.size());
      fflush(stdout);
    }

  } while ((int)BoundaryUntouched_map.size() > 0);
  printf("The searching is done\n");
  fflush(stdout);
}

/*
        Neigbor # Table

 X  Y  Z   Num
-1 -1 -1 =  0
 0 -1 -1 =  1
 1 -1 -1 =  2
-1  0 -1 =  3
 0  0 -1 =  4
 1  0 -1 =  5
-1  1 -1 =  6
 0  1 -1 =  7
 1  1 -1 =  8
-1 -1  0 =  9
 0 -1  0 = 10
 1 -1  0 = 11
-1  0  0 = 12
 0  0  0 = 13
 1  0  0 = 14
-1  1  0 = 15
 0  1  0 = 16
 1  1  0 = 17
-1 -1  1 = 18
 0 -1  1 = 19
 1 -1  1 = 20
-1  0  1 = 21
 0  0  1 = 22
 1  0  1 = 23
-1  1  1 = 24
 0  1  1 = 25
 1  1  1 = 26
*/

// Adjusting the boundary acoording to gradient magnitudes
// Removing the false classified voxels
template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::AdjustingBoundary(
    map<int, unsigned char> &BoundaryLocs_map) {
  int i, m, loc[7], DataCoor_i[3];
  map<int, unsigned char> BoundaryUntouched_map;
  map<int, unsigned char>::iterator Boundary_it;
  int *UntouchedCoor_i, NumUntouchedVoxels;

  UntouchedCoor_i = new int[(int)BoundaryLocs_map.size() * 3];

  // Copy the boundary map
  NumUntouchedVoxels = (int)BoundaryLocs_map.size();
  Boundary_it = BoundaryLocs_map.begin();
  for (i = 0; i < (int)BoundaryLocs_map.size(); i++, Boundary_it++) {
    loc[0] = (*Boundary_it).first;
    UntouchedCoor_i[i * 3 + 2] = loc[0] / this->WtimesH_mi;
    UntouchedCoor_i[i * 3 + 1] =
        (loc[0] - UntouchedCoor_i[i * 3 + 2] * this->WtimesH_mi) /
        this->Width_mi;
    UntouchedCoor_i[i * 3 + 0] = loc[0] % this->Width_mi;
  }

#ifdef DEBUG_MC_GEOM
  printf("Start to find the local maximum gradient magnitudes ... \n");
  printf("# Untouched Voxels = %d\n", NumUntouchedVoxels);
  fflush(stdout);
  int Iteration = 0;
#endif

  do {

    BoundaryUntouched_map.clear();

    for (i = 0; i < NumUntouchedVoxels; i++) {

      DataCoor_i[2] = UntouchedCoor_i[i * 3 + 2];
      DataCoor_i[1] = UntouchedCoor_i[i * 3 + 1];
      DataCoor_i[0] = UntouchedCoor_i[i * 3 + 0];

      loc[0] = DataCoor_i[2] * this->WtimesH_mi +
               DataCoor_i[1] * this->Width_mi + DataCoor_i[0];

      loc[1] = DataCoor_i[2] * this->WtimesH_mi +
               DataCoor_i[1] * this->Width_mi + DataCoor_i[0] + 1;
      loc[2] = DataCoor_i[2] * this->WtimesH_mi +
               DataCoor_i[1] * this->Width_mi + DataCoor_i[0] - 1;
      loc[3] = DataCoor_i[2] * this->WtimesH_mi +
               (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
      loc[4] = DataCoor_i[2] * this->WtimesH_mi +
               (DataCoor_i[1] - 1) * this->Width_mi + DataCoor_i[0];
      loc[5] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
               DataCoor_i[1] * this->Width_mi + DataCoor_i[0];
      loc[6] = (DataCoor_i[2] - 1) * this->WtimesH_mi +
               DataCoor_i[1] * this->Width_mi + DataCoor_i[0];
      for (m = 1; m <= 6; m++) {
        if (this->GradientMag_mf[loc[m]] > this->GradientMag_mf[loc[0]]) {
          BoundaryUntouched_map[loc[m]] = (unsigned char)0;
          BoundaryLocs_map[loc[m]] = (unsigned char)0;
          //					BoundaryLocs_map.erase(loc[0]);
        }
      }

      /*
                              int		l, n;
                              for (n=DataCoor_i[2]-1; n<=DataCoor_i[2]+1; n++)
      { if (n<0 || n>=this->Depth_mi) continue; for (m=DataCoor_i[1]-1;
      m<=DataCoor_i[1]+1; m++) { if (m<0 || m>=this->Height_mi) continue; for
      (l=DataCoor_i[0]-1; l<=DataCoor_i[0]+1; l++) { if (l<0 ||
      l>=this->Width_mi) continue; loc[1] = n*this->WtimesH_mi +
      m*this->Width_mi + l;

                                                      if
      (this->GradientMag_mf[loc[1]] >  this->GradientMag_mf[loc[0]]) {
                                                              BoundaryUntouched_map[loc[1]]
      = (unsigned char)0;
      //
      BoundaryLocs_map[loc[1]] = (unsigned char)0;
                                                              BoundaryLocs_map.erase(loc[0]);
                                                      }
                                              }
                                      }
                              }
      */
    }

    delete[] UntouchedCoor_i;
    NumUntouchedVoxels = (int)BoundaryUntouched_map.size();
    if (NumUntouchedVoxels <= 0)
      break;
    UntouchedCoor_i = new int[NumUntouchedVoxels * 3];

    Boundary_it = BoundaryUntouched_map.begin();
    for (i = 0; i < (int)BoundaryUntouched_map.size(); i++, Boundary_it++) {
      loc[0] = (*Boundary_it).first;
      UntouchedCoor_i[i * 3 + 2] = loc[0] / this->WtimesH_mi;
      UntouchedCoor_i[i * 3 + 1] =
          (loc[0] - UntouchedCoor_i[i * 3 + 2] * this->WtimesH_mi) /
          this->Width_mi;
      UntouchedCoor_i[i * 3 + 0] = loc[0] % this->Width_mi;
    }

#ifdef DEBUG_MC_GEOM
    printf("The # Iterations of the boundary adjusting = %d\n", Iteration++);
    fflush(stdout);
#endif

  } while (1);

  delete[] UntouchedCoor_i;

  printf("Copying the boundary map\n");
  fflush(stdout);

  // Copy the boundary map (BoundaryUntouched_map <-- BoundaryLocs_map)
  BoundaryUntouched_map.clear();
  Boundary_it = BoundaryLocs_map.begin();
  for (m = 0; m < (int)BoundaryLocs_map.size(); m++, Boundary_it++) {

    /*
                    loc[0] = (*Boundary_it).first;
                    DataCoor_i[2] = loc[0]/this->WtimesH_mi;
                    DataCoor_i[1] = (loc[0] -
       DataCoor_i[2]*this->WtimesH_mi)/this->Width_mi; DataCoor_i[0] = loc[0]
       % this->Width_mi;

                    for (k=DataCoor_i[2]-1; k<=DataCoor_i[2]+1; k++) {
                            if (k<0 || k>=this->Depth_mi) continue;
                            for (j=DataCoor_i[1]-1; j<=DataCoor_i[1]+1; j++) {
                                    if (j<0 || j>=this->Height_mi) continue;
                                    for (i=DataCoor_i[0]-1;
       i<=DataCoor_i[0]+1; i++) { if (i<0 || i>=this->Width_mi) continue;
                                            loc[1] = k*this->WtimesH_mi +
       j*this->Width_mi + i; BoundaryUntouched_map[loc[1]] = (unsigned char)0;
                                    }
                            }
                    }
    */

    BoundaryUntouched_map[(*Boundary_it).first] = (unsigned char)0;
  }
  BoundaryLocs_map.clear();

  printf("The end of the Copying the boundary map\n");
  fflush(stdout);

  Boundary_it = BoundaryUntouched_map.begin();
  do {

    loc[0] = (*Boundary_it).first;
    DataCoor_i[2] = loc[0] / this->WtimesH_mi;
    DataCoor_i[1] =
        (loc[0] - DataCoor_i[2] * this->WtimesH_mi) / this->Width_mi;
    DataCoor_i[0] = loc[0] % this->Width_mi;
    BoundaryUntouched_map.erase(loc[0]);

    loc[1] = DataCoor_i[2] * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0] + 1;
    loc[2] = DataCoor_i[2] * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0] - 1;
    loc[3] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] + 1) * this->Width_mi + DataCoor_i[0];
    loc[4] = DataCoor_i[2] * this->WtimesH_mi +
             (DataCoor_i[1] - 1) * this->Width_mi + DataCoor_i[0];
    loc[5] = (DataCoor_i[2] + 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0];
    loc[6] = (DataCoor_i[2] - 1) * this->WtimesH_mi +
             DataCoor_i[1] * this->Width_mi + DataCoor_i[0];

    /*
                    if
       (this->SecondDerivative_mf[loc[1]]*this->SecondDerivative_mf[loc[2]]<0
       ||
                            this->SecondDerivative_mf[loc[3]]*this->SecondDerivative_mf[loc[4]]<0
       ||
                            this->SecondDerivative_mf[loc[5]]*this->SecondDerivative_mf[loc[6]]<0)
       {

                            for (k=DataCoor_i[2]-1; k<=DataCoor_i[2]+1; k++) {
                                    if (k<0 || k>=this->Depth_mi) continue;
                                    for (j=DataCoor_i[1]-1;
       j<=DataCoor_i[1]+1; j++) { if (j<0 || j>=this->Height_mi) continue; for
       (i=DataCoor_i[0]-1; i<=DataCoor_i[0]+1; i++) { if (i<0 ||
       i>=this->Width_mi) continue; loc[0] = k*this->WtimesH_mi +
       j*this->Width_mi + i; BoundaryLocs_map[loc[0]] = (unsigned char)0;
                                            }
                                    }
                            }
                    }
    */

    for (i = 1; i <= 6; i++) {
      if (this->SecondDerivative_mf[loc[0]] *
              this->SecondDerivative_mf[loc[i]] <
          0) {
        BoundaryLocs_map[loc[0]] = (unsigned char)0;
        BoundaryLocs_map[loc[i]] = (unsigned char)0;
      }
    }

    //		BoundaryLocs_map[loc[0]] = (unsigned char)0;

    Boundary_it = BoundaryUntouched_map.begin();

  } while ((int)BoundaryUntouched_map.size() > 0);

#ifdef DEBUG_MC_GEOM
  printf("The end of removing the maximum gradient magnitudes ... \n");
  fflush(stdout);
#endif
}

template <class _DataType, class _ExtraDataType>
int cMC_Geom<_DataType, _ExtraDataType>::IsMaterialBoundaryUsingMinMax(
    int DataLoc, _DataType MatMin, _DataType MatMax) {
  int i, j, k, loc[3];
  int XCoor, YCoor, ZCoor;

  // The given location should be between the min and max values
  if (this->PED_m.getDataAt(DataLoc) < MatMin ||
      MatMax < this->PED_m.getDataAt(DataLoc))
    return false;

  ZCoor = DataLoc / this->WtimesH_mi;
  YCoor = (DataLoc - ZCoor * this->WtimesH_mi) / this->Height_mi;
  XCoor = DataLoc % this->Width_mi;

  // Checking all 26 neighbors, whether at least one of them is a different
  // material
  for (k = ZCoor - 1; k <= ZCoor + 1; k++) {
    if (k < 0 || k >= this->Depth_mi)
      return true;
    for (j = YCoor - 1; j <= YCoor + 1; j++) {
      if (j < 0 || j >= this->Height_mi)
        return true;
      for (i = XCoor - 1; i <= XCoor + 1; i++) {
        if (i < 0 || i >= this->Width_mi)
          return true;

        loc[0] = k * this->WtimesH_mi + j * this->Width_mi + i;
        if (this->PED_m.getDataAt(loc[0]) < MatMin ||
            MatMax < this->PED_m.getDataAt(loc[0]))
          return true;
      }
    }
  }

  return false;
}

int TempMatNum;

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::TriangleClassification() {
  int i, j, VertIdx[4], TriIndex[2], MatNum;

  for (i = 0; i < NumMaterials_mi; i++) {
    BoundaryVoxelExtraction(i, Ranges_mi[i * 2], Ranges_mi[i * 2 + 1],
                            BoundaryLocs_map[i]);

    // Save Boundary Volume
    int loc[4];
    unsigned char *BoundaryVolume = new unsigned char[this->WHD_mi];
    for (j = 0; j < this->WHD_mi; j++)
      BoundaryVolume[j] = (unsigned char)0;
    map<int, unsigned char>::iterator BoundaryLoc_it =
        BoundaryLocs_map[i].begin();
    for (j = 0; j < (int)BoundaryLocs_map[i].size(); j++, BoundaryLoc_it++) {
      loc[0] = (*BoundaryLoc_it).first;
      BoundaryVolume[loc[0]] = (unsigned char)255;
    }
    printf("Save Boundary Volume \n");
    char Postfix[200];
    sprintf(Postfix, "Mat%d_Boundary", i);
    SaveVolume(BoundaryVolume, 0, 255, Postfix); // Control.cpp
    printf("The End of the Saving\n");
    fflush(stdout);
    delete[] BoundaryVolume;
  }

  BuildVertexToTriangleLink();

  MatNumOfTriangle_mi = new int[this->NumTriangles_mi];
  for (i = 0; i < this->NumTriangles_mi; i++)
    MatNumOfTriangle_mi[i] = -1;

  map<int, unsigned char> ConnectedTriangleIndex_map;
  map<int, unsigned char> VertexIndex_map;
  map<int, unsigned char>::iterator VertexIndex_it;
  map<int, unsigned char>::iterator TriangleIndex_it;
  ConnectedTriangleIndex_map.clear();
  VertexIndex_map.clear();

  int NumDisconnectedObjects;

  for (i = 0; i < this->NumTriangles_mi; i++) {

    if (MatNumOfTriangle_mi[i] == -1) { // -1 means that it is not touched

      NumDisconnectedObjects++;

      VertexIndex_map[this->VertexIndex_mi[i * 3 + 0]] = (unsigned char)0;
      VertexIndex_map[this->VertexIndex_mi[i * 3 + 1]] = (unsigned char)0;
      VertexIndex_map[this->VertexIndex_mi[i * 3 + 2]] = (unsigned char)0;

      ConnectedTriangleIndex_map.clear();
      ConnectedTriangleIndex_map[i] = (unsigned char)0;

      do {

        VertexIndex_it = VertexIndex_map.begin();
        VertIdx[0] = (*VertexIndex_it).first;
        VertexIndex_map.erase(VertIdx[0]);

        for (j = 0; j < MaxNumVToTLinks_mi; j++) {
          if (VertexToTriangleIndex_mi[VertIdx[0] * MaxNumVToTLinks_mi + j] >
              0) {
            TriIndex[0] =
                VertexToTriangleIndex_mi[VertIdx[0] * MaxNumVToTLinks_mi + j];
            TriangleIndex_it = ConnectedTriangleIndex_map.find(TriIndex[0]);

            if (TriangleIndex_it == ConnectedTriangleIndex_map.end()) {
              ConnectedTriangleIndex_map[TriIndex[0]] = (unsigned char)0;

              VertexIndex_map[this->VertexIndex_mi[TriIndex[0] * 3 + 0]] =
                  (unsigned char)0;
              VertexIndex_map[this->VertexIndex_mi[TriIndex[0] * 3 + 1]] =
                  (unsigned char)0;
              VertexIndex_map[this->VertexIndex_mi[TriIndex[0] * 3 + 2]] =
                  (unsigned char)0;
            }
          }
        }

      } while ((int)VertexIndex_map.size() > 0);

      //			if (ConnectedTriangleIndex_map.size()>=500)
      //MatNum = TempMatNum++; 			else MatNum = -99999;

      if (ConnectedTriangleIndex_map.size() <= 16)
        MatNum = MAT_NUM_DOES_NOT_EXIST;
      else
        MatNum = DecidingMaterialNum(ConnectedTriangleIndex_map);

      TriangleIndex_it = ConnectedTriangleIndex_map.begin();
      for (j = 0; j < (int)ConnectedTriangleIndex_map.size();
           j++, TriangleIndex_it++) {
        TriIndex[0] = (*TriangleIndex_it).first;
        MatNumOfTriangle_mi[TriIndex[0]] = MatNum;
      }

      //	if ((int)ConnectedTriangleIndex_map.size()>=500) {

#ifdef DEBUG_MC_GEOM
      if (MatNum >= 0)
        printf(
            "============================================================\n");
      if ((int)ConnectedTriangleIndex_map.size() > 16) {
        printf("Disconnedted object # = %d, ", NumDisconnectedObjects);
        printf("Mat # = %3d, ", MatNum);
        printf("# Connected Triangles = %d\n",
               (int)ConnectedTriangleIndex_map.size());
      }
      if (MatNum >= 0)
        printf(
            "============================================================\n");
      fflush(stdout);
#endif

      //	}
    }

#ifdef DEBUG_MC_GEOM
    if (i % 10000 == 0) {
      printf("Progressed Triangles = %d / %d\n", i, this->NumTriangles_mi);
    }
#endif
  }

  printf("The Total # of disconnedted objects = %d\n",
         NumDisconnectedObjects);
}

template <class _DataType, class _ExtraDataType>
int cMC_Geom<_DataType, _ExtraDataType>::DecidingMaterialNum(
    map<int, unsigned char> &TriangleIndex_map) {
  int i, j, k, VertexIndex, TriangleIndex, loc[4];
  map<int, unsigned char> TriangleVertex_map;
  map<int, unsigned char>::iterator TriangleVertex_it;
  TriangleVertex_map.clear();

  map<int, unsigned char>::iterator TriangleIndex_it;
  map<int, unsigned char>::iterator BoundaryLocs_it;
  int *MatHittingRecords = new int[NumMaterials_mi];
  float *Vertices = this->getVerticesPtsf();

  for (i = 0; i < NumMaterials_mi; i++)
    MatHittingRecords[i] = 0;

  TriangleIndex_it = TriangleIndex_map.begin();
  for (i = 0; i < (int)TriangleIndex_map.size(); i++, TriangleIndex_it++) {

    TriangleIndex = (*TriangleIndex_it).first;
    for (j = 0; j < 3; j++) {
      VertexIndex = this->VertexIndex_mi[TriangleIndex * 3 + j];

      loc[0] = ((int)(Vertices[VertexIndex * 3 + 2])) * this->WtimesH_mi +
               ((int)(Vertices[VertexIndex * 3 + 1])) * this->Width_mi +
               ((int)(Vertices[VertexIndex * 3 + 0]));
      TriangleVertex_map[loc[0]] = (unsigned char)0;
    }
  }

  TriangleVertex_it = TriangleVertex_map.begin();
  for (j = 0; j < (int)TriangleVertex_map.size(); j++, TriangleVertex_it++) {
    loc[0] = (*TriangleVertex_it).first;
    for (k = 0; k < NumMaterials_mi; k++) {
      BoundaryLocs_it = BoundaryLocs_map[k].find(loc[0]);
      if (BoundaryLocs_it != BoundaryLocs_map[k].end())
        MatHittingRecords[k]++;
    }
  }

  int MaxHitting = 0, MaxHittingMatNum = -1;
  for (i = 0; i < NumMaterials_mi; i++) {
    if (MatHittingRecords[i] > MaxHitting) {
      MaxHittingMatNum = i;
      MaxHitting = MatHittingRecords[i];
    }
  }

  delete[] MatHittingRecords;
  if (MaxHittingMatNum >= 0 &&
      MatHittingRecords[MaxHittingMatNum] >=
          (int)TriangleVertex_map.size() / (4 * 2 * 2 * 2) &&
      (int)TriangleVertex_map.size() >=
          (int)((float)BoundaryLocs_map[MaxHittingMatNum].size() / 27. * 3. /
                2.)) {

#ifdef DEBUG_MC_GEOM
    printf("\n(Hitting # = %d, ", MatHittingRecords[MaxHittingMatNum]);
    printf("# Boundary Locs of Mat %d = %d, ", MaxHittingMatNum,
           (int)BoundaryLocs_map[MaxHittingMatNum].size());
    printf("# Triangle Vertices = %d)\n", (int)TriangleVertex_map.size());
#endif
    delete[] MatHittingRecords;
    TriangleVertex_map.clear();
    return MaxHittingMatNum;
  } else {

#ifdef DEBUG_MC_GEOM
    if (MatHittingRecords[MaxHittingMatNum] > 16) {
      printf("\n(Hitting # = %d, ", MatHittingRecords[MaxHittingMatNum]);
      printf("# Triangle Vertices = %d, ", (int)TriangleVertex_map.size());
      printf("# Boundary Locs of Mat %d = %d) ", MaxHittingMatNum,
             (int)BoundaryLocs_map[MaxHittingMatNum].size());
    }
#endif

    delete[] MatHittingRecords;
    TriangleVertex_map.clear();
    return MAT_NUM_DOES_NOT_EXIST;
  }
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::BuildVertexToTriangleLink() {
  int i, j, k, VertexIdx;

  VertexToTriangleIndex_mi =
      new int[this->NumVertices_mi * MaxNumVToTLinks_mi];

  for (i = 0; i < this->NumVertices_mi * MaxNumVToTLinks_mi; i++) {
    VertexToTriangleIndex_mi[i] = -1;
  }

  // i = triangle index
  for (i = 0; i < this->NumTriangles_mi; i++) {
    for (j = 0; j < 3; j++) {
      VertexIdx = this->VertexIndex_mi[i * 3 + j];

#ifdef DEBUG_MC_GEOM
      if (VertexToTriangleIndex_mi[VertexIdx * MaxNumVToTLinks_mi +
                                   MaxNumVToTLinks_mi - 1] > 0) {
        printf(
            "MaxNumVToTLinks_mi should be bigger than %d at vertex # %d of\n",
            MaxNumVToTLinks_mi, VertexIdx);
      }
#endif
      for (k = 0; k < MaxNumVToTLinks_mi; k++) {

#ifdef DEBUG_MC_GEOM
        if (VertexToTriangleIndex_mi[VertexIdx * MaxNumVToTLinks_mi + k] ==
            i) {
          printf("The triangle index(%d) is overlapped at verex %d\n", i,
                 VertexIdx);
          continue;
        }
#endif

        if (VertexToTriangleIndex_mi[VertexIdx * MaxNumVToTLinks_mi + k] <
            0) {
          VertexToTriangleIndex_mi[VertexIdx * MaxNumVToTLinks_mi + k] = i;
          break;
        }
      }
    }
  }
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::SaveMatGeometry_RAW(char *filename,
                                                              int MatNum) {
  FILE *fp_out;
  int i;
  char OutFileName[500];
  float *Vertices = this->getVerticesPtsf();

  printf("Saving the geometry using the raw format...\n");

  sprintf(OutFileName, "%s_Mat%02d_Geom.raw", filename, MatNum);
  fp_out = fopen(OutFileName, "w");
  if (fp_out == NULL) {
    fprintf(stderr, "Cannot open the file: %s\n", OutFileName);
    exit(1);
  }

  int NumTriangleOfTheMaterial = 0;
  for (i = 0; i < this->NumTriangles_mi; i++) {
    if (MatNumOfTriangle_mi[i] == MatNum)
      NumTriangleOfTheMaterial++;
  }

  if (NumTriangleOfTheMaterial == 0)
    return;

  //	fprintf (fp_out, "%d %d\n", this->NumVertices_mi,
  //this->NumTriangles_mi);
  fprintf(fp_out, "%d %d\n", this->NumVertices_mi, NumTriangleOfTheMaterial);

  for (i = 0; i < this->NumVertices_mi; i++) {
    fprintf(fp_out, "%f %f %f\n", Vertices[i * 3 + 0], Vertices[i * 3 + 1],
            Vertices[i * 3 + 2]);
  }
  for (i = 0; i < this->NumTriangles_mi; i++) {
    if (MatNumOfTriangle_mi[i] == MatNum) {
      fprintf(fp_out, "%d %d %d\n", this->VertexIndex_mi[i * 3 + 0],
              this->VertexIndex_mi[i * 3 + 1],
              this->VertexIndex_mi[i * 3 + 2]);
    }
  }
  fprintf(fp_out, "\n");
  fclose(fp_out);
}

template <class _DataType, class _ExtraDataType>
void cMC_Geom<_DataType, _ExtraDataType>::Destroy() {
  delete[] VertexToTriangleIndex_mi;
  VertexToTriangleIndex_mi = NULL;

  delete[] MatNumOfTriangle_mi;
  MatNumOfTriangle_mi = NULL;

  delete[] Ranges_mi;
  Ranges_mi = NULL;

  for (int i = 0; i < MaxNumMaterials; i++) {
    BoundaryLocs_map[i].clear();
  }
  delete[] ZeroCrossingVolume_mf;
  ZeroCrossingVolume_mf = NULL;
}

template class cMC_Geom<unsigned char, unsigned char>;
// template class cMC_Geom<unsigned char, unsigned short>;
// template class cMC_Geom<unsigned char, int>;
// template class cMC_Geom<unsigned char, float>;

// template class cMC_Geom<unsigned short, unsigned char>;
// template class cMC_Geom<unsigned short, unsigned short>;
// template class cMC_Geom<unsigned short, int>;
// template class cMC_Geom<unsigned short, float>;

// template class cMC_Geom<int, unsigned char>;
// template class cMC_Geom<int, unsigned short>;
// template class cMC_Geom<int, int>;
// template class cMC_Geom<int, float>;

// template class cMC_Geom<float, unsigned char>;
// template class cMC_Geom<float, unsigned short>;
// template class cMC_Geom<float, int>;
// template class cMC_Geom<float, float>;
