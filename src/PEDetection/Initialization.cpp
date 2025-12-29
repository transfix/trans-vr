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
#include <PEDetection/Initialization.h>
#include <iostream.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

template <class _DataType> cInitialValue<_DataType>::cInitialValue() {
  gvf_m = NULL;
}

template <class _DataType> cInitialValue<_DataType>::~cInitialValue() {
  delete[] InitialValues_mT;
  delete[] UpperBoundInitialValues_mT;
  delete[] LowerBoundInitialValues_mT;
  delete[] InitialValueLocations_mi;
  // Data_mT. Gradient_mf should not be removed,
  // since they are defined outside of this class
}
template <class _DataType>
void cInitialValue<_DataType>::setHistogram(int *Histogram) {
  Histogram_mi = Histogram;
}

template <class _DataType>
void cInitialValue<_DataType>::setWHD(int W, int H, int D) {
  Width_mi = W, Height_mi = H, Depth_mi = D;

  WtimesH_mi = W * H;
  WHD_mi = W * H * D;

  MaxNumInitialValues_mi = 100;
  InitialValues_mT = new _DataType[MaxNumInitialValues_mi];
  LowerBoundInitialValues_mT = new _DataType[MaxNumInitialValues_mi];
  UpperBoundInitialValues_mT = new _DataType[MaxNumInitialValues_mi];
  InitialValueLocations_mi = new int[MaxNumInitialValues_mi * 3];
  NumInitialValues_mi = 0;
}

template <class _DataType>
void cInitialValue<_DataType>::getInitalValues(_DataType *InitValues) {
  for (int i = 0; i < NumInitialValues_mi; i++) {
    InitValues[i] = InitialValues_mT[i];
  }
}

template <class _DataType>
void cInitialValue<_DataType>::getInitialValueLocations(int *InitLoc) {
  for (int i = 0; i < NumInitialValues_mi; i++) {
    InitLoc[i * 3] = InitialValueLocations_mi[i * 3];
    InitLoc[i * 3 + 1] = InitialValueLocations_mi[i * 3 + 1];
    InitLoc[i * 3 + 2] = InitialValueLocations_mi[i * 3 + 2];
  }
}

// Using GVF Seed Points as initial values
template <class _DataType>
int cInitialValue<_DataType>::FindInitialValues(int NumSeedPts,
                                                int *SeedPtsLocations) {
  int i, Xi, Yi, Zi;
  int ReturnNumMaterials;
  _DataType LocalMin, LocalMax, Average;
  unsigned char *SegImage;
  char SegImageName[512];

  for (i = 0; i < NumSeedPts; i++) {
    Xi = SeedPtsLocations[i * 3];
    Yi = SeedPtsLocations[i * 3 + 1];
    Zi = SeedPtsLocations[i * 3 + 2];

    FindMinMaxNeighbor(Xi, Yi, Zi, LocalMin, LocalMax);
    AddInitialValue(Xi, Yi, Zi, LocalMin, LocalMax);
  }

  // Step 1.
  // Merge two initial values, if a range is totally included by another
  // Take one which has higher frequency
  QuickSortInitialValues(); // Swap Upper and Lower bound and Locations also
  ReturnNumMaterials = RangeHistogramMergeInitialValues();

  cout << "Range Merged : The new number of materials = "
       << ReturnNumMaterials << endl;

  QuickSortInitialValues(); // Swap Upper and Lower bound and Locations also
  cout << "After Range Histogram Merge InitialValues (1st Step)" << endl;
  DisplayInitialValuesRange();
  if (NumInitialValues_mi <= 2)
    return NumInitialValues_mi;

  /*
          i=0;
          InitialValueLocations_mi[0] = 218;
          InitialValueLocations_mi[1] = 242;
          InitialValueLocations_mi[2] = 0;

          i++;
          InitialValueLocations_mi[i*3 + 0] = 207;
          InitialValueLocations_mi[i*3 + 1] = 250;
          InitialValueLocations_mi[i*3 + 2] = 0;

          i++;
          InitialValueLocations_mi[i*3 + 0] = 224;
          InitialValueLocations_mi[i*3 + 1] = 256;
          InitialValueLocations_mi[i*3 + 2] = 0;

          i++;
          InitialValueLocations_mi[i*3 + 0] = 205;
          InitialValueLocations_mi[i*3 + 1] = 268;
          InitialValueLocations_mi[i*3 + 2] = 0;

          i++;
          InitialValueLocations_mi[i*3 + 0] = 247;
          InitialValueLocations_mi[i*3 + 1] = 259;
          InitialValueLocations_mi[i*3 + 2] = 0;
  */

  // Step 2.
  // Recompute the min & max values using GVF
  for (i = 0; i < NumInitialValues_mi; i++) {
    LocalMin = (_DataType)999999;
    LocalMax = (_DataType)0;

    if (gvf_m != NULL) {
      //			cout << endl << "Checking Min Max using GVF"
      //<< endl; 			cout << "Initial Value Num = " << i << endl;
      SegImage = gvf_m->BoundaryMinMax(1, &InitialValueLocations_mi[i * 3],
                                       LocalMin, LocalMax, Average);
      //			cout << "Num Locations of Inside Boundary = "
      //<< gvf_m->getNumInsideBoundaryLocations() << endl;
      if (SegImage != NULL) {
        sprintf(SegImageName, "%s_Segmented_%03d.ppm", TargetName_gc, i);
        SaveImage(Width_mi, Height_mi, SegImage, SegImageName);
      }
    } else {
      cout << "Error!: GVF is NULL" << endl;
      exit(1);
    }

    Xi = InitialValueLocations_mi[i * 3];
    Yi = InitialValueLocations_mi[i * 3 + 1];
    Zi = InitialValueLocations_mi[i * 3 + 2];
    UpdateMinMaxInitialValue(Xi, Yi, Zi, LocalMin, LocalMax, Average);
  }
  cout << endl;

  cout << "After GVF Min Max Value Calculation (2nd Step) " << endl;
  QuickSortInitialValues(); // Swap Upper and Lower bound and Locations also
  DisplayInitialValuesRange();
  ReturnNumMaterials = RangeHistogramMergeInitialValues();
  QuickSortInitialValues(); // Swap Upper and Lower bound and Locations also
  cout
      << "After Range Histogram  Merging Initial Values using the GVF Results"
      << endl
      << endl;
  DisplayInitialValuesRange();

  /*
          // Step 4.
          // Merging using histogram, distances, and gradient magnitudes
          // Agglomerative Clustering
          QuickSortInitialValues(); // Swap Upper and Lower bound and
     Locations also

          DistanceHistogramMergeInitialValues(NumMaterials);

          cout << "Distance Histogram Merge Initial Values() -- Agglomerative
     Clustering (Final 3rd Step) " << endl << endl; QuickSortInitialValues();
     // Swap Upper and Lower bound and Locations also
          DisplayInitialValuesRange();
  */

  return ReturnNumMaterials;
}

template <class _DataType>
int cInitialValue<_DataType>::AgglomerativeMerging(int NumMaterials) {

  // Step 4.
  // Merging using histogram, distances, and gradient magnitudes
  // Agglomerative Clustering
  QuickSortInitialValues(); // Swap Upper and Lower bound and Locations also

  DistanceHistogramMergeInitialValues(NumMaterials);

  cout << "Distance Histogram Merge Initial Values() -- Agglomerative "
          "Clustering (Final 3rd Step) "
       << endl
       << endl;
  QuickSortInitialValues(); // Swap Upper and Lower bound and Locations also
  DisplayInitialValuesRange();

  return getNumInitialValues();
}

template <class _DataType>
int cInitialValue<_DataType>::IsLocalMin(int xi, int yj, int zk,
                                         _DataType &Min, _DataType &Max) {
  int WindowSize = 5;
  int loc[2], n, l, m, i;
  float CurrGradient, NeighborGrad;

  if (Depth_mi == 1) {

    // Do not consider the boundary of data
    for (i = 0; i < WindowSize / 2; i++) {
      if (xi == i || yj == i)
        return FALSE;
      if (xi == Width_mi - 1 - i || yj == Height_mi - 1 - i)
        return FALSE;
    }
    CurrGradient = Gradient_mf[yj * Width_mi + xi];

    for (l = -WindowSize / 2 + yj; l <= WindowSize / 2 + yj; l++) {
      for (m = -WindowSize / 2 + xi; m <= WindowSize / 2 + xi; m++) {

        loc[0] = l * Width_mi + m;
        NeighborGrad = Gradient_mf[loc[0]];
        //				if (Min > Data_mT[loc[0]]) Min =
        //Data_mT[loc[0]]; // Min & Max in 5x5 				if (Max < Data_mT[loc[0]]) Max
        //= Data_mT[loc[0]];

        if (CurrGradient > NeighborGrad) {
          return FALSE;
        }
      }
    }
  } else {
    // Do not consider the boundary of data
    for (i = 0; i < WindowSize / 2; i++) {
      if (xi == i || yj == i || zk == i)
        return FALSE;
      if (xi == Width_mi - 1 - i || yj == Height_mi - 1 - i ||
          zk == Depth_mi - 1 - i)
        return FALSE;
    }
    CurrGradient = Gradient_mf[zk * WtimesH_mi + yj * Width_mi + xi];

    for (n = -WindowSize / 2 + zk; n <= WindowSize / 2 + zk; n++) {
      for (l = -WindowSize / 2 + yj; l <= WindowSize / 2 + yj; l++) {
        for (m = -WindowSize / 2 + xi; m <= WindowSize / 2 + xi; m++) {

          loc[0] = n * WtimesH_mi + l * Width_mi + m;
          NeighborGrad = Gradient_mf[loc[0]];
          //					if (Min > Data_mT[loc[0]]) Min
          //= Data_mT[loc[0]]; // Min & Max in 5x5 					if (Max < Data_mT[loc[0]])
          //Max = Data_mT[loc[0]];

          if (CurrGradient > NeighborGrad) {
            return FALSE;
          }
        }
      }
    }
  }

  WindowSize = 3;
  Min = (_DataType)99999;
  Max = (_DataType)0;

  if (Depth_mi == 1) {
    for (l = -WindowSize / 2 + yj; l <= WindowSize / 2 + yj; l++) {
      for (m = -WindowSize / 2 + xi; m <= WindowSize / 2 + xi; m++) {
        loc[0] = l * Width_mi + m;
        if (Min > Data_mT[loc[0]])
          Min = Data_mT[loc[0]]; // Min & Max in 3x3
        if (Max < Data_mT[loc[0]])
          Max = Data_mT[loc[0]];
      }
    }
  } else {
    for (n = -WindowSize / 2 + zk; n <= WindowSize / 2 + zk; n++) {
      for (l = -WindowSize / 2 + yj; l <= WindowSize / 2 + yj; l++) {
        for (m = -WindowSize / 2 + xi; m <= WindowSize / 2 + xi; m++) {
          loc[0] = n * WtimesH_mi + l * Width_mi + m;
          if (Min > Data_mT[loc[0]])
            Min = Data_mT[loc[0]]; // Min & Max in 3x3
          if (Max < Data_mT[loc[0]])
            Max = Data_mT[loc[0]];
        }
      }
    }
  }
  return TRUE;
}

template <class _DataType>
int cInitialValue<_DataType>::FindMinMaxNeighbor(int xi, int yj, int zk,
                                                 _DataType &Min,
                                                 _DataType &Max) {
  int WindowSize = 5;
  int loc[2], n, l, m;

  WindowSize = 3;
  Min = (_DataType)99999;
  Max = (_DataType)0;

  if (Depth_mi == 1) {
    for (l = -WindowSize / 2 + yj; l <= WindowSize / 2 + yj; l++) {
      for (m = -WindowSize / 2 + xi; m <= WindowSize / 2 + xi; m++) {
        loc[0] = l * Width_mi + m;
        if (Min > Data_mT[loc[0]])
          Min = Data_mT[loc[0]]; // Min & Max in 3x3
        if (Max < Data_mT[loc[0]])
          Max = Data_mT[loc[0]];
      }
    }
  } else {
    for (n = -WindowSize / 2 + zk; n <= WindowSize / 2 + zk; n++) {
      for (l = -WindowSize / 2 + yj; l <= WindowSize / 2 + yj; l++) {
        for (m = -WindowSize / 2 + xi; m <= WindowSize / 2 + xi; m++) {
          loc[0] = n * WtimesH_mi + l * Width_mi + m;
          if (Min > Data_mT[loc[0]])
            Min = Data_mT[loc[0]]; // Min & Max in 3x3x3
          if (Max < Data_mT[loc[0]])
            Max = Data_mT[loc[0]];
        }
      }
    }
  }
  return TRUE;
}

template <class _DataType>
void cInitialValue<_DataType>::AddInitialValue(int xi, int yj, int zk,
                                               _DataType LocalMin,
                                               _DataType LocalMax) {
  int i;
  _DataType Data;

  if (Depth_mi == 1)
    Data = Data_mT[yj * Width_mi + xi];
  else
    Data = Data_mT[zk * WtimesH_mi + yj * Width_mi + xi];

  for (i = 0; i < NumInitialValues_mi; i++) {
    if (Data == InitialValues_mT[i])
      return;
    //		if (LocalMin >= LowerBoundInitialValues_mT[i] && LocalMax <=
    //UpperBoundInitialValues_mT[i]) return; 		if (Data >=
    //LowerBoundInitialValues_mT[i] && Data <= UpperBoundInitialValues_mT[i])
    //return;
  }

  if (NumInitialValues_mi < MaxNumInitialValues_mi) {
    InitialValues_mT[NumInitialValues_mi] = Data;
    LowerBoundInitialValues_mT[NumInitialValues_mi] = LocalMin;
    UpperBoundInitialValues_mT[NumInitialValues_mi] = LocalMax;
    InitialValueLocations_mi[NumInitialValues_mi * 3] = xi;
    InitialValueLocations_mi[NumInitialValues_mi * 3 + 1] = yj;
    InitialValueLocations_mi[NumInitialValues_mi * 3 + 2] = zk;
    NumInitialValues_mi++;
  } else {
    MaxNumInitialValues_mi *= 2;
    cout << "MaxNumInitialValues_mi = " << MaxNumInitialValues_mi << endl;

    _DataType *InitialValues = new _DataType[MaxNumInitialValues_mi];
    _DataType *UpperBound = new _DataType[MaxNumInitialValues_mi];
    _DataType *LowerBound = new _DataType[MaxNumInitialValues_mi];
    int *InitialValueLocations = new int[MaxNumInitialValues_mi * 3];

    for (i = 0; i < NumInitialValues_mi; i++) {
      InitialValues[i] = InitialValues_mT[i];
      UpperBound[i] = LowerBoundInitialValues_mT[i];
      LowerBound[i] = UpperBoundInitialValues_mT[i];
      InitialValueLocations[i * 3] = InitialValueLocations_mi[i * 3];
      InitialValueLocations[i * 3 + 1] = InitialValueLocations_mi[i * 3 + 1];
      InitialValueLocations[i * 3 + 2] = InitialValueLocations_mi[i * 3 + 2];
    }

    delete[] InitialValues_mT;
    delete[] LowerBoundInitialValues_mT;
    delete[] UpperBoundInitialValues_mT;
    delete[] InitialValueLocations_mi;

    InitialValues_mT = InitialValues;
    LowerBoundInitialValues_mT = UpperBound;
    UpperBoundInitialValues_mT = LowerBound;
    InitialValueLocations_mi = InitialValueLocations;

    InitialValues_mT[NumInitialValues_mi] = Data;
    LowerBoundInitialValues_mT[NumInitialValues_mi] = LocalMin;
    UpperBoundInitialValues_mT[NumInitialValues_mi] = LocalMax;
    InitialValueLocations_mi[NumInitialValues_mi * 3] = xi;
    InitialValueLocations_mi[NumInitialValues_mi * 3 + 1] = yj;
    InitialValueLocations_mi[NumInitialValues_mi * 3 + 2] = zk;
    NumInitialValues_mi++;
  }
}

template <class _DataType>
void cInitialValue<_DataType>::UpdateMinMaxInitialValue(int xi, int yi,
                                                        int zi, _DataType Min,
                                                        _DataType Max) {
  for (int i = 0; i < NumInitialValues_mi; i++) {
    if (InitialValueLocations_mi[i * 3 + 0] == xi &&
        InitialValueLocations_mi[i * 3 + 1] == yi &&
        InitialValueLocations_mi[i * 3 + 2] == zi) {

      LowerBoundInitialValues_mT[i] = Min;
      UpperBoundInitialValues_mT[i] = Max;
      break;
    }
  }
}

template <class _DataType>
void cInitialValue<_DataType>::UpdateMinMaxInitialValue(int xi, int yi,
                                                        int zi, _DataType Min,
                                                        _DataType Max,
                                                        _DataType Ave) {
  for (int i = 0; i < NumInitialValues_mi; i++) {
    if (InitialValueLocations_mi[i * 3 + 0] == xi &&
        InitialValueLocations_mi[i * 3 + 1] == yi &&
        InitialValueLocations_mi[i * 3 + 2] == zi) {

      LowerBoundInitialValues_mT[i] = Min;
      UpperBoundInitialValues_mT[i] = Max;
      InitialValues_mT[i] = Ave;
      break;
    }
  }
}

// Merge two initial values, if a range is totally included by another
template <class _DataType>
void cInitialValue<_DataType>::RangeMergeInitialValues() {
  int NewNumInitialValues, i, j, k, Insert;
  _DataType *InitialValues;
  _DataType *UpperBound;
  _DataType *LowerBound;
  int *InitialValueLocations;

  NewNumInitialValues = NumInitialValues_mi;
  do {

    cout << "RangeMergeInitialValues: NewNumInitialValues = "
         << NewNumInitialValues << endl;
    DisplayInitialValuesRange(NewNumInitialValues);

    InitialValues = new _DataType[NewNumInitialValues];
    LowerBound = new _DataType[NewNumInitialValues];
    UpperBound = new _DataType[NewNumInitialValues];
    InitialValueLocations = new int[NewNumInitialValues * 3];

    for (j = 0, i = 0; i < NewNumInitialValues; i++) {
      Insert = TRUE;
      for (k = i + 1; k < NewNumInitialValues; k++) {

        if (LowerBoundInitialValues_mT[i] >= LowerBoundInitialValues_mT[k] &&
            UpperBoundInitialValues_mT[i] <= UpperBoundInitialValues_mT[k]) {

          InitialValues_mT[k] = InitialValues_mT[i];
          InitialValueLocations_mi[k * 3] = InitialValueLocations_mi[i * 3];
          InitialValueLocations_mi[k * 3 + 1] =
              InitialValueLocations_mi[i * 3 + 1];
          InitialValueLocations_mi[k * 3 + 2] =
              InitialValueLocations_mi[i * 3 + 2];
          Insert = FALSE;
          break;
        } else if (LowerBoundInitialValues_mT[i] <=
                       LowerBoundInitialValues_mT[k] &&
                   UpperBoundInitialValues_mT[i] >=
                       UpperBoundInitialValues_mT[k]) {

          InitialValues_mT[i] = InitialValues_mT[k];
          InitialValueLocations_mi[i * 3] = InitialValueLocations_mi[k * 3];
          InitialValueLocations_mi[i * 3 + 1] =
              InitialValueLocations_mi[k * 3 + 1];
          InitialValueLocations_mi[i * 3 + 2] =
              InitialValueLocations_mi[k * 3 + 2];
          Insert = TRUE;
          break;
        }
      }
      if (Insert) {
        InitialValues[j] = InitialValues_mT[i];
        LowerBound[j] = LowerBoundInitialValues_mT[i];
        UpperBound[j] = UpperBoundInitialValues_mT[i];
        InitialValueLocations[j * 3] = InitialValueLocations_mi[i * 3];
        InitialValueLocations[j * 3 + 1] =
            InitialValueLocations_mi[i * 3 + 1];
        InitialValueLocations[j * 3 + 2] =
            InitialValueLocations_mi[i * 3 + 2];
        j++;
      }
    }

    delete[] InitialValues_mT;
    delete[] LowerBoundInitialValues_mT;
    delete[] UpperBoundInitialValues_mT;
    delete[] InitialValueLocations_mi;

    InitialValues_mT = InitialValues;
    LowerBoundInitialValues_mT = LowerBound;
    UpperBoundInitialValues_mT = UpperBound;
    InitialValueLocations_mi = InitialValueLocations;

    if (NewNumInitialValues == j)
      break;
    else
      NewNumInitialValues = j;

  } while (1);

  NumInitialValues_mi = NewNumInitialValues;
}

// Merge two initial values, if a range is totally included by another
template <class _DataType>
int cInitialValue<_DataType>::RangeHistogramMergeInitialValues() {
  int NewNumInitialValues, i, j, k; //, Insert;
  _DataType *InitialValues;
  _DataType *UpperBound;
  _DataType *LowerBound;
  int *InitialValueLocations;

  NewNumInitialValues = NumInitialValues_mi;

  do {

    InitialValues = new _DataType[NewNumInitialValues];
    LowerBound = new _DataType[NewNumInitialValues];
    UpperBound = new _DataType[NewNumInitialValues];
    InitialValueLocations = new int[NewNumInitialValues * 3];

    for (j = 0, i = 0; i < NewNumInitialValues; i++) {

      // Check whether the i-th initial value is already inserted or not
      if (InitialValueLocations_mi[i * 3] == 0 &&
          InitialValueLocations_mi[i * 3 + 1] == 0 &&
          InitialValueLocations_mi[i * 3 + 2] == 0)
        continue;

      // Insert the value, if the lower bound and the upper bound are the same
      // as the initial values. The initial value has high probability to be
      // background
      if (InitialValues_mT[i] == LowerBoundInitialValues_mT[i] &&
          InitialValues_mT[i] == UpperBoundInitialValues_mT[i]) {
        InitialValues[j] = InitialValues_mT[i];
        LowerBound[j] = LowerBoundInitialValues_mT[i];
        UpperBound[j] = UpperBoundInitialValues_mT[i];
        InitialValueLocations[j * 3] = InitialValueLocations_mi[i * 3];
        InitialValueLocations[j * 3 + 1] =
            InitialValueLocations_mi[i * 3 + 1];
        InitialValueLocations[j * 3 + 2] =
            InitialValueLocations_mi[i * 3 + 2];
        j++;

        // Mark that i-th initial value is inserted
        InitialValueLocations_mi[i * 3] = 0;
        InitialValueLocations_mi[i * 3 + 1] = 0;
        InitialValueLocations_mi[i * 3 + 2] = 0;
        continue;
      }

      for (k = i + 1; k < NewNumInitialValues; k++) {
        // if (k==i) continue;

        // Check whether the two initial values are the same
        if (InitialValues_mT[i] == InitialValues_mT[k]) {

          if (LowerBoundInitialValues_mT[i] > LowerBoundInitialValues_mT[k])
            LowerBoundInitialValues_mT[i] = LowerBoundInitialValues_mT[k];
          if (UpperBoundInitialValues_mT[i] < UpperBoundInitialValues_mT[k])
            UpperBoundInitialValues_mT[i] = UpperBoundInitialValues_mT[k];
          InitialValueLocations[i * 3] = InitialValueLocations_mi[k * 3];
          InitialValueLocations[i * 3 + 1] =
              InitialValueLocations_mi[k * 3 + 1];
          InitialValueLocations[i * 3 + 2] =
              InitialValueLocations_mi[k * 3 + 2];

          // Mark that i-th initial value is inserted
          InitialValueLocations_mi[k * 3] = 0;
          InitialValueLocations_mi[k * 3 + 1] = 0;
          InitialValueLocations_mi[k * 3 + 2] = 0;
          break;
        }

        // Check whether the i-th initial value is already inserted or not
        if (InitialValueLocations_mi[k * 3] == 0 &&
            InitialValueLocations_mi[k * 3 + 1] == 0 &&
            InitialValueLocations_mi[k * 3 + 2] == 0)
          continue;

        if (LowerBoundInitialValues_mT[i] >= LowerBoundInitialValues_mT[k] &&
            UpperBoundInitialValues_mT[i] <= UpperBoundInitialValues_mT[k]) {

          // Use the InitialValues_mT[k]'s range
          LowerBoundInitialValues_mT[i] = LowerBoundInitialValues_mT[k];
          UpperBoundInitialValues_mT[i] = UpperBoundInitialValues_mT[k];

          if (Histogram_mi[(int)((float)InitialValues_mT[i] - MinData_mf)] <=
              Histogram_mi[(int)((float)InitialValues_mT[k] - MinData_mf)]) {
            // Insert the k-th value
            InitialValues_mT[i] = InitialValues_mT[k];
            InitialValueLocations_mi[i * 3] = InitialValueLocations_mi[k * 3];
            InitialValueLocations_mi[i * 3 + 1] =
                InitialValueLocations_mi[k * 3 + 1];
            InitialValueLocations_mi[i * 3 + 2] =
                InitialValueLocations_mi[k * 3 + 2];
          }
          // Make that the k-th initial value is merged with the i-th value
          // So, the k-th value should not be inserted next time
          InitialValueLocations_mi[k * 3] = 0;
          InitialValueLocations_mi[k * 3 + 1] = 0;
          InitialValueLocations_mi[k * 3 + 2] = 0;
          break;
        } else if (LowerBoundInitialValues_mT[i] <=
                       LowerBoundInitialValues_mT[k] &&
                   UpperBoundInitialValues_mT[i] >=
                       UpperBoundInitialValues_mT[k]) {

          // Use the InitialValues_mT[i]'s range

          if (Histogram_mi[(int)((float)InitialValues_mT[i] - MinData_mf)] <
              Histogram_mi[(int)((float)InitialValues_mT[k] - MinData_mf)]) {

            // Insert the k-th value
            InitialValues_mT[i] = InitialValues_mT[k];
            InitialValueLocations_mi[i * 3] = InitialValueLocations_mi[k * 3];
            InitialValueLocations_mi[i * 3 + 1] =
                InitialValueLocations_mi[k * 3 + 1];
            InitialValueLocations_mi[i * 3 + 2] =
                InitialValueLocations_mi[k * 3 + 2];
          }
          // Make that the k-th initial value is merged with the i-th value
          // So, the k-th value should not be inserted next time
          InitialValueLocations_mi[k * 3] = 0;
          InitialValueLocations_mi[k * 3 + 1] = 0;
          InitialValueLocations_mi[k * 3 + 2] = 0;
          break;
        }
      }

      InitialValues[j] = InitialValues_mT[i];
      LowerBound[j] = LowerBoundInitialValues_mT[i];
      UpperBound[j] = UpperBoundInitialValues_mT[i];
      InitialValueLocations[j * 3] = InitialValueLocations_mi[i * 3];
      InitialValueLocations[j * 3 + 1] = InitialValueLocations_mi[i * 3 + 1];
      InitialValueLocations[j * 3 + 2] = InitialValueLocations_mi[i * 3 + 2];
      j++;

      // Mark that i-th initial value is inserted
      InitialValueLocations_mi[i * 3] = 0;
      InitialValueLocations_mi[i * 3 + 1] = 0;
      InitialValueLocations_mi[i * 3 + 2] = 0;
    }

    delete[] InitialValues_mT;
    delete[] LowerBoundInitialValues_mT;
    delete[] UpperBoundInitialValues_mT;
    delete[] InitialValueLocations_mi;

    InitialValues_mT = InitialValues;
    LowerBoundInitialValues_mT = LowerBound;
    UpperBoundInitialValues_mT = UpperBound;
    InitialValueLocations_mi = InitialValueLocations;

    if (NewNumInitialValues == j)
      break;
    else
      NewNumInitialValues = j;

  } while (1);

  NumInitialValues_mi = NewNumInitialValues;

  return NumInitialValues_mi;
}

// Merge two initial values using gradient magnitude values and Histogram
template <class _DataType>
void cInitialValue<_DataType>::DistanceHistogramMergeInitialValues(
    int NumMaterials) {
  int NewNumInitialValues, i, j;
  _DataType *InitialValues;
  _DataType *UpperBound;
  _DataType *LowerBound;
  double *Distance, MinDistance, Tempd;
  int *InitialValueLocations, MinDistLoc = 0;

  NewNumInitialValues = NumInitialValues_mi;
  do {

    //		cout << "NumMaterials = " << NumMaterials << endl;
    //		cout << "NewNumInitialValues = " << NewNumInitialValues <<
    //endl;
    if (NewNumInitialValues <= NumMaterials)
      break;

    //		cout <<"DistanceMergeInitialValues: NewNumInitialValues = " <<
    //NewNumInitialValues << endl;
    //		DisplayInitialValuesRange(NewNumInitialValues);

    InitialValues = new _DataType[NewNumInitialValues];
    LowerBound = new _DataType[NewNumInitialValues];
    UpperBound = new _DataType[NewNumInitialValues];
    InitialValueLocations = new int[NewNumInitialValues * 3];
    Distance = new double[NewNumInitialValues];

    MinDistance = (double)999999;
    for (i = 0; i < NewNumInitialValues - 1; i++) {
      Distance[i] =
          fabs((double)InitialValues_mT[i] - InitialValues_mT[i + 1]);
      if (MinDistance > Distance[i]) {
        MinDistance = Distance[i];
        MinDistLoc = i;
      }
    }

    // Considering gradient magnitude value and Histogram
    float Grad1, Grad2;
    int xi, yi, zi, xj, yj, zj;

    xi = InitialValueLocations_mi[MinDistLoc * 3];
    yi = InitialValueLocations_mi[MinDistLoc * 3 + 1];
    zi = InitialValueLocations_mi[MinDistLoc * 3 + 2];

    xj = InitialValueLocations_mi[(MinDistLoc + 1) * 3];
    yj = InitialValueLocations_mi[(MinDistLoc + 1) * 3 + 1];
    zj = InitialValueLocations_mi[(MinDistLoc + 1) * 3 + 2];

    Grad1 = Gradient_mf[zi * WtimesH_mi + yi * Width_mi + xi];
    Grad2 = Gradient_mf[zj * WtimesH_mi + yj * Width_mi + xj];

    // Computing the average of the two nearest values
    Tempd = ((double)InitialValues_mT[MinDistLoc] +
             InitialValues_mT[MinDistLoc + 1]) /
            2.0;
    InitialValues_mT[MinDistLoc] = (_DataType)Tempd;
    InitialValues_mT[MinDistLoc + 1] = (_DataType)Tempd;

    if (Grad1 == Grad2) {
      if (Histogram_mi[(int)((float)InitialValues_mT[MinDistLoc] -
                             MinData_mf)] >
          Histogram_mi[(int)((float)InitialValues_mT[MinDistLoc + 1] -
                             MinData_mf)])
        MinDistLoc += 1;
    } else if (Grad1 < Grad2)
      MinDistLoc += 1;

    // An initial value, which is in MinDistLoc, will be removed
    for (j = 0, i = 0; i < NewNumInitialValues; i++) {
      if (i == MinDistLoc)
        continue;
      InitialValues[j] = InitialValues_mT[i];
      LowerBound[j] = LowerBoundInitialValues_mT[i];
      UpperBound[j] = UpperBoundInitialValues_mT[i];
      InitialValueLocations[j * 3] = InitialValueLocations_mi[i * 3];
      InitialValueLocations[j * 3 + 1] = InitialValueLocations_mi[i * 3 + 1];
      InitialValueLocations[j * 3 + 2] = InitialValueLocations_mi[i * 3 + 2];
      j++;
    }

    delete[] InitialValues_mT;
    delete[] LowerBoundInitialValues_mT;
    delete[] UpperBoundInitialValues_mT;
    delete[] InitialValueLocations_mi;

    InitialValues_mT = InitialValues;
    LowerBoundInitialValues_mT = LowerBound;
    UpperBoundInitialValues_mT = UpperBound;
    InitialValueLocations_mi = InitialValueLocations;

    NewNumInitialValues = j;
    //		if (NewNumInitialValues == NumMaterials) break;

  } while (1);

  NumInitialValues_mi = NewNumInitialValues;
}

template <class _DataType>
void cInitialValue<_DataType>::QuickSortInitialValues() {
  QuickSort(InitialValues_mT, 0, NumInitialValues_mi - 1);
}

template <class _DataType>
void cInitialValue<_DataType>::QuickSort(_DataType *data, int p, int r) {
  int q;

  if (p < r) {
    q = Partition(data, p, r);
    QuickSort(data, p, q - 1);
    QuickSort(data, q + 1, r);
  }
}

template <class _DataType>
void cInitialValue<_DataType>::Swap(unsigned char &x, unsigned char &y) {
  unsigned char Temp;
  Temp = x;
  x = y;
  y = Temp;
}

template <class _DataType>
void cInitialValue<_DataType>::Swap(unsigned short &x, unsigned short &y) {
  unsigned short Temp;
  Temp = x;
  x = y;
  y = Temp;
}

template <class _DataType>
void cInitialValue<_DataType>::Swap(float &x, float &y) {
  float Temp;
  Temp = x;
  x = y;
  y = Temp;
}

template <class _DataType>
void cInitialValue<_DataType>::Swap(int &x, int &y) {
  int Temp;
  Temp = x;
  x = y;
  y = Temp;
}

template <class _DataType>
int cInitialValue<_DataType>::Partition(_DataType *data, int low, int high) {
  int left, right, Pivot_Loc[3];
  _DataType pivot_item, Pivot_Upper, Pivot_Lower;

  pivot_item = data[low];
  Pivot_Lower = LowerBoundInitialValues_mT[low];
  Pivot_Upper = UpperBoundInitialValues_mT[low];
  Pivot_Loc[0] = InitialValueLocations_mi[low * 3];
  Pivot_Loc[1] = InitialValueLocations_mi[low * 3 + 1];
  Pivot_Loc[2] = InitialValueLocations_mi[low * 3 + 2];

  left = low;
  right = high;

  while (left < right) {

    while (data[left] <= pivot_item && left <= high)
      left++;
    while (data[right] > pivot_item && right >= low)
      right--;
    if (left < right) {
      Swap(data[left], data[right]);
      Swap(LowerBoundInitialValues_mT[left],
           LowerBoundInitialValues_mT[right]);
      Swap(UpperBoundInitialValues_mT[left],
           UpperBoundInitialValues_mT[right]);

      Swap(InitialValueLocations_mi[left * 3],
           InitialValueLocations_mi[right * 3]);
      Swap(InitialValueLocations_mi[left * 3 + 1],
           InitialValueLocations_mi[right * 3 + 1]);
      Swap(InitialValueLocations_mi[left * 3 + 2],
           InitialValueLocations_mi[right * 3 + 2]);
    }
  }

  data[low] = data[right];
  data[right] = pivot_item;

  LowerBoundInitialValues_mT[low] = LowerBoundInitialValues_mT[right];
  UpperBoundInitialValues_mT[low] = UpperBoundInitialValues_mT[right];
  InitialValueLocations_mi[low * 3] = InitialValueLocations_mi[right * 3];
  InitialValueLocations_mi[low * 3 + 1] =
      InitialValueLocations_mi[right * 3 + 1];
  InitialValueLocations_mi[low * 3 + 2] =
      InitialValueLocations_mi[right * 3 + 2];

  LowerBoundInitialValues_mT[right] = Pivot_Lower;
  UpperBoundInitialValues_mT[right] = Pivot_Upper;
  InitialValueLocations_mi[right * 3] = Pivot_Loc[0];
  InitialValueLocations_mi[right * 3 + 1] = Pivot_Loc[1];
  InitialValueLocations_mi[right * 3 + 2] = Pivot_Loc[2];

  return right;
}

//-------------------------------------------------------------------------------------------

// Merge two initial values using histogram and gradient magnitudes
template <class _DataType>
void cInitialValue<_DataType>::DistanceMergeInitialValues(int NumMaterials) {
  int NewNumInitialValues, i, j;
  _DataType *InitialValues;
  _DataType *UpperBound;
  _DataType *LowerBound;
  double *Distance, MinDistance;
  int *InitialValueLocations, MinDistLoc = 0;

  if (NumInitialValues_mi <= NumMaterials)
    return;

  NewNumInitialValues = NumInitialValues_mi;
  do {

    if (NewNumInitialValues <= NumMaterials)
      break;

    //		cout <<"DistanceMergeInitialValues: NewNumInitialValues = " <<
    //NewNumInitialValues << endl;
    //		DisplayInitialValuesRange(NewNumInitialValues);

    InitialValues = new _DataType[NewNumInitialValues];
    LowerBound = new _DataType[NewNumInitialValues];
    UpperBound = new _DataType[NewNumInitialValues];
    InitialValueLocations = new int[NewNumInitialValues * 3];
    Distance = new double[NewNumInitialValues];

    MinDistance = (double)999999;
    for (i = 0; i < NewNumInitialValues - 1; i++) {
      Distance[i] =
          fabs((double)InitialValues_mT[i] - InitialValues_mT[i + 1]);
      if (MinDistance > Distance[i]) {
        MinDistance = Distance[i];
        MinDistLoc = i;
      }
    }

    // Considering gradient magnitude value
    //		float	MaxGradient = (float)-999999.0;
    float Grad1, Grad2;
    int xi, yi, zi, xj, yj, zj;

    xi = InitialValueLocations_mi[MinDistLoc * 3];
    yi = InitialValueLocations_mi[MinDistLoc * 3 + 1];
    zi = InitialValueLocations_mi[MinDistLoc * 3 + 2];

    xj = InitialValueLocations_mi[(MinDistLoc + 1) * 3];
    yj = InitialValueLocations_mi[(MinDistLoc + 1) * 3 + 1];
    zj = InitialValueLocations_mi[(MinDistLoc + 1) * 3 + 2];

    Grad1 = Gradient_mf[zi * WtimesH_mi + yi * Width_mi + xi];
    Grad2 = Gradient_mf[zj * WtimesH_mi + yj * Width_mi + xj];

    if (Grad1 == Grad2) {
      if (Histogram_mi[(int)((float)InitialValues_mT[MinDistLoc] -
                             MinData_mf)] >
          Histogram_mi[(int)((float)InitialValues_mT[MinDistLoc + 1] -
                             MinData_mf)])
        MinDistLoc += 1;
    } else if (Grad1 < Grad2)
      MinDistLoc += 1;

    for (j = 0, i = 0; i < NewNumInitialValues; i++) {
      if (i == MinDistLoc)
        continue;
      InitialValues[j] = InitialValues_mT[i];
      LowerBound[j] = LowerBoundInitialValues_mT[i];
      UpperBound[j] = UpperBoundInitialValues_mT[i];
      InitialValueLocations[j * 3] = InitialValueLocations_mi[i * 3];
      InitialValueLocations[j * 3 + 1] = InitialValueLocations_mi[i * 3 + 1];
      InitialValueLocations[j * 3 + 2] = InitialValueLocations_mi[i * 3 + 2];
      j++;
    }

    delete[] InitialValues_mT;
    delete[] LowerBoundInitialValues_mT;
    delete[] UpperBoundInitialValues_mT;
    delete[] InitialValueLocations_mi;

    InitialValues_mT = InitialValues;
    LowerBoundInitialValues_mT = LowerBound;
    UpperBoundInitialValues_mT = UpperBound;
    InitialValueLocations_mi = InitialValueLocations;

    NewNumInitialValues = j;
    //		if (NewNumInitialValues == NumMaterials) break;

  } while (1);

  NumInitialValues_mi = NewNumInitialValues;
}

// Merge two initial values, if a range is totally included by another
template <class _DataType>
void cInitialValue<_DataType>::MaxGradientMagnitudeMerge(int NumMaterials) {
  int NewNumInitialValues, i, j;
  _DataType *InitialValues;
  _DataType *UpperBound;
  _DataType *LowerBound;
  int *InitialValueLocations, MinGradLoc = 0;
  int xi, yi, zi;
  float MaxGradient, CurrGradient;

  if (NumInitialValues_mi <= NumMaterials)
    return;

  NewNumInitialValues = NumInitialValues_mi;
  do {

    InitialValues = new _DataType[NewNumInitialValues];
    LowerBound = new _DataType[NewNumInitialValues];
    UpperBound = new _DataType[NewNumInitialValues];
    InitialValueLocations = new int[NewNumInitialValues * 3];

    MaxGradient = (float)-999999.0;
    for (i = 0; i < NewNumInitialValues; i++) {

      xi = InitialValueLocations_mi[i * 3];
      yi = InitialValueLocations_mi[i * 3 + 1];
      zi = InitialValueLocations_mi[i * 3 + 2];

      CurrGradient = Gradient_mf[zi * WtimesH_mi + yi * Width_mi + xi];

      if (MaxGradient < CurrGradient) {
        MaxGradient = CurrGradient;
        MinGradLoc = i;
      }
    }

    for (j = 0, i = 0; i < NewNumInitialValues; i++) {
      if (i == MinGradLoc)
        continue;
      InitialValues[j] = InitialValues_mT[i];
      LowerBound[j] = LowerBoundInitialValues_mT[i];
      UpperBound[j] = UpperBoundInitialValues_mT[i];
      InitialValueLocations[j * 3] = InitialValueLocations_mi[i * 3];
      InitialValueLocations[j * 3 + 1] = InitialValueLocations_mi[i * 3 + 1];
      InitialValueLocations[j * 3 + 2] = InitialValueLocations_mi[i * 3 + 2];
      j++;
    }

    delete[] InitialValues_mT;
    delete[] LowerBoundInitialValues_mT;
    delete[] UpperBoundInitialValues_mT;
    delete[] InitialValueLocations_mi;

    InitialValues_mT = InitialValues;
    LowerBoundInitialValues_mT = LowerBound;
    UpperBoundInitialValues_mT = UpperBound;
    InitialValueLocations_mi = InitialValueLocations;

    NewNumInitialValues = j;
    if (NewNumInitialValues == NumMaterials)
      break;

  } while (1);

  NumInitialValues_mi = NewNumInitialValues;
}

template <class _DataType>
void cInitialValue<_DataType>::DisplayInitialValuesRange() {
  DisplayInitialValuesRange(NumInitialValues_mi);
}

template <class _DataType>
void cInitialValue<_DataType>::DisplayInitialValuesRange(int FirstN) {
  int i, xi, yi, zi;

  cout << "Num Initial Values = " << NumInitialValues_mi << endl;
  cout << "Intensity (RangeMin, Max) (LocationX, Y, Z)  Frequency  Gradient"
       << endl;
  for (i = 0; i < FirstN; i++) {
    cout.setf(ios::fixed); // used oridnary decimal notation
    cout.setf(ios::right);
    cout.width(4);
    cout << i << " ";

    cout.width(6);
    cout << (double)InitialValues_mT[i];

    cout << " (";
    cout.width(6);
    cout << (double)LowerBoundInitialValues_mT[i] << ",";
    cout.width(6);
    cout << (double)UpperBoundInitialValues_mT[i] << ")  ";

    cout << " (";
    cout.width(3);
    cout << InitialValueLocations_mi[i * 3] << ",";
    cout.width(3);
    cout << InitialValueLocations_mi[i * 3 + 1] << ",";
    cout.width(3);
    cout << InitialValueLocations_mi[i * 3 + 2] << ") ";

    cout.width(10);
    cout << Histogram_mi[(int)((float)InitialValues_mT[i] - MinData_mf)];

    xi = InitialValueLocations_mi[i * 3];
    yi = InitialValueLocations_mi[i * 3 + 1];
    zi = InitialValueLocations_mi[i * 3 + 2];

    cout.width(12);
    cout << Gradient_mf[zi * WtimesH_mi + yi * Width_mi + xi];

    cout << endl;
  }
  cout << endl;
}

template class cInitialValue<unsigned char>;
template class cInitialValue<unsigned short>;
template class cInitialValue<int>;
template class cInitialValue<float>;
