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
#include <PEDetection/Evaluation.h>
#include <iostream.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

template <class _DataType> cEvaluation<_DataType>::cEvaluation() {}

template <class _DataType> cEvaluation<_DataType>::~cEvaluation() {}

template <class _DataType>
void cEvaluation<_DataType>::setData(_DataType *Data, float Min, float Max) {
  Data_mT = Data;
  MinData_mf = Min;
  MaxData_mf = Max;
}

template <class _DataType>
void cEvaluation<_DataType>::setGradient(float *Grad, float Min, float Max) {
  GradientMag_mf = Grad;
  MinGradMag_mf = Min;
  MaxGradMag_mf = Max;
}

template <class _DataType>
void cEvaluation<_DataType>::setSecondDerivative(float *SecondD, float Min,
                                                 float Max) {
  SecondDerivative_mf = SecondD;
  MinSecond_mf = Min;
  MaxSecond_mf = Max;
}

template <class _DataType>
void cEvaluation<_DataType>::setHistogram(int *Histo, float HistoF) {
  Histogram_mi = Histo;
  HistogramFactor_mf = HistoF;
}

template <class _DataType>
void cEvaluation<_DataType>::setWHD(int W, int H, int D) {
  Width_mi = W, Height_mi = H, Depth_mi = D;

  WtimesH_mi = W * H;
  WHD_mi = W * H * D;
}

template <class _DataType>
void cEvaluation<_DataType>::FindAndEvaluateRanges(float *Material_Prob,
                                                   int NumClusters) {
  int i, j, ithCluster, loc[2];
  _DataType DataMin = (_DataType)999999, DataMax = (_DataType)0;
  int FoundARange;

  NumMatRanges_mi = 0;
  // Finding material ranges
  cout << "Find and Evaluate Ranges ... " << endl;
  for (ithCluster = 0; ithCluster < NumClusters; ithCluster++) {
    for (j = 0; j < (int)((MaxData_mf - MinData_mf) * HistogramFactor_mf);
         j++) {

      FoundARange = false;
      // Find DataMin
      for (i = j; i < (int)((MaxData_mf - MinData_mf) * HistogramFactor_mf);
           i++) {
        if (Histogram_mi[i] > 0) {
          loc[0] = i * NumClusters + ithCluster;
          if (Material_Prob[loc[0]] >= 0.1) {
            DataMin = (_DataType)i;
            FoundARange = true;
            break;
          }
        }
      }
      i++;
      // Find DataMax
      for (; i < (int)((MaxData_mf - MinData_mf) * HistogramFactor_mf); i++) {
        if (FoundARange) {
          loc[0] = i * NumClusters + ithCluster;
          if (Material_Prob[loc[0]] < 0.1) {
            DataMax = (_DataType)(i - 1);
            break;
          }
        } else
          break;
      }
      // Considering the final data value
      if (i >= (int)((MaxData_mf - MinData_mf) * HistogramFactor_mf)) {
        if (FoundARange)
          DataMax = (_DataType)(i);
      }

      if (FoundARange) {
        MatRanges_ms[NumMatRanges_mi].DataMin = DataMin;
        MatRanges_ms[NumMatRanges_mi].DataMax = DataMax;
        MatRanges_ms[NumMatRanges_mi].AveFirstD = 0.0;
        MatRanges_ms[NumMatRanges_mi].AveSecondD = 0.0;
        MatRanges_ms[NumMatRanges_mi].NumBoundaries = 0;
        MatRanges_ms[NumMatRanges_mi].ithMaterial = (unsigned char)ithCluster;
        MatRanges_ms[NumMatRanges_mi].TotalNumMaterials =
            (unsigned char)NumClusters;
        NumMatRanges_mi++;
        if (NumMatRanges_mi > MAX_MATRANGES) {
          cout << "The maximum number of material ranges is greater than "
                  "MAX_MATRANGES: ";
          cout << MAX_MATRANGES << endl;
          exit(1);
        }

#ifdef DEBUG_EVALUATION
        cout << " Min & Max = "
             << (int)MatRanges_ms[NumMatRanges_mi - 1].DataMin << " ";
        cout << (int)MatRanges_ms[NumMatRanges_mi - 1].DataMax << ", ";
        cout << "1st & 2nd D = "
             << MatRanges_ms[NumMatRanges_mi - 1].AveFirstD << " ";
        cout << MatRanges_ms[NumMatRanges_mi - 1].AveSecondD << ", ";
        cout << "# Boundaries = "
             << MatRanges_ms[NumMatRanges_mi - 1].NumBoundaries << endl;
#endif
      }
      j = i + 1;
    }
  }

  // Compute Boundary Fitness
  cout << "Compute Boundary Fitness ... " << endl;
  BoundaryFitting(NumMatRanges_mi);

#ifdef DEBUG_EVALUATION
  cout << "The Results of Bounary Fitness" << endl;
  for (i = 0; i < NumMatRanges_mi; i++) {
    printf("Range %2d ", i);
    printf("MinMax = %3d %3d, ", (int)MatRanges_ms[i].DataMin,
           (int)MatRanges_ms[i].DataMax);
    printf("1st & 2nd D = %7.3f %7.3f, ", MatRanges_ms[i].AveFirstD,
           MatRanges_ms[i].AveSecondD);
    printf("# B = %8d", MatRanges_ms[i].NumBoundaries);
    printf("\n");
  }

#endif

  // Add Each Material Ranges
  cout << "Add each material ranges" << endl;
  for (i = 0; i < NumMatRanges_mi; i++) {
    //		int NumAddedMatRanges = AddAMatRange(MatRanges_ms[i],
    //MatRanges_ms[i].AveFirstD);
    AddAMatRange(MatRanges_ms[i], MatRanges_ms[i].AveFirstD);
  }
}

// Compute the fitness of the material boundary
template <class _DataType>
void cEvaluation<_DataType>::BoundaryFitting(int NumRanges) {
  int i, j;
  int TrueBoundary = false;
  double Tempd;

  for (i = 0; i < WHD_mi; i++) {
    for (j = 0; j < NumRanges; j++) {
      if (MatRanges_ms[j].DataMin <= Data_mT[i] &&
          Data_mT[i] <= MatRanges_ms[j].DataMax) {
        TrueBoundary = IsMaterialBoundary(i, MatRanges_ms[j].DataMin,
                                          MatRanges_ms[j].DataMax);
        if (TrueBoundary) {
          Tempd = ((double)GradientMag_mf[i] - MinGradMag_mf) /
                  ((double)MaxGradMag_mf - MinGradMag_mf);
          MatRanges_ms[j].AveFirstD += Tempd * 1000.0;
          if (SecondDerivative_mf[i] < 0.0) {
            Tempd = (double)SecondDerivative_mf[i] / (double)MinSecond_mf;
            MatRanges_ms[j].AveSecondD += fabs(Tempd) * 1000.0;
          } else {
            Tempd = (double)SecondDerivative_mf[i] / (double)MaxSecond_mf;
            MatRanges_ms[j].AveSecondD += fabs(Tempd) * 1000.0;
          }
          MatRanges_ms[j].NumBoundaries++;
        }
        TrueBoundary = false;
      }
    }
  }

  for (j = 0; j < NumRanges; j++) {
    if (MatRanges_ms[j].NumBoundaries > 0) {
      MatRanges_ms[j].AveFirstD /= (double)MatRanges_ms[j].NumBoundaries;
      MatRanges_ms[j].AveSecondD /= (double)MatRanges_ms[j].NumBoundaries;
    } else {
      MatRanges_ms[j].AveFirstD = 0.0;
      MatRanges_ms[j].AveSecondD = 0.0;
    }
  }
}

template <class _DataType>
int cEvaluation<_DataType>::AddAMatRange(
    sMatRangeInfo<_DataType> &MatRange_input, double FirstD) {
  sMatRangeInfo<_DataType> *MatRange_conflict, *MatRange_new;
  class map<double, sMatRangeInfo<_DataType> *>::iterator currPt_it =
      MatRange_mm.find(FirstD);

  MatRange_new = new sMatRangeInfo<_DataType>;
  MatRange_new->DataMin = MatRange_input.DataMin;
  MatRange_new->DataMax = MatRange_input.DataMax;
  MatRange_new->AveFirstD = MatRange_input.AveFirstD;
  MatRange_new->AveSecondD = MatRange_input.AveSecondD;
  MatRange_new->NumBoundaries = MatRange_input.NumBoundaries;
  MatRange_new->ithMaterial = MatRange_input.ithMaterial;
  MatRange_new->TotalNumMaterials = MatRange_input.TotalNumMaterials;

  if (currPt_it == MatRange_mm.end()) {
    MatRange_mm[FirstD] = MatRange_new; // Add the material range structure
  } else {
    MatRange_conflict = (*currPt_it).second;
    if (MatRange_conflict->DataMin == MatRange_new->DataMin &&
        MatRange_conflict->DataMax == MatRange_new->DataMax) {
    } else {
      MatRange_new->AveSecondD += 1e10 - 6;
      MatRange_mm[FirstD] = MatRange_new;
    }
  }

  return MatRange_mm.size();
}

// i = Data Location, j = Material Number
template <class _DataType>
int cEvaluation<_DataType>::IsMaterialBoundary(int DataLoc, _DataType DataMin,
                                               _DataType DataMax) {
  int i, j, k, loc[2];
  int XCoor, YCoor, ZCoor;

  ZCoor = DataLoc / WtimesH_mi;
  YCoor = (DataLoc - ZCoor * WtimesH_mi) / Height_mi;
  XCoor = DataLoc % Width_mi;

  // Checking all 26 neighbors, whether at least one of them is a different
  // material
  for (k = ZCoor - 1; k <= ZCoor + 1; k++) {
    if (k < 0 || k >= Depth_mi)
      return true;
    for (j = YCoor - 1; j <= YCoor + 1; j++) {
      if (j < 0 || j >= Height_mi)
        return true;
      for (i = XCoor - 1; i <= XCoor + 1; i++) {
        if (i < 0 || i >= Width_mi)
          return true;

        loc[0] = k * WtimesH_mi + j * Width_mi + i;
        if (Data_mT[loc[0]] < DataMin || DataMax < Data_mT[loc[0]])
          return true;
      }
    }
  }

  return false;
}

template <class _DataType>
void cEvaluation<_DataType>::DisplayMatRangeInfo() {
  int i, TotalNumMat, ithMat;
  sMatRangeInfo<_DataType> *MatRange;
  class map<double, sMatRangeInfo<_DataType> *>::iterator currPt_it =
      MatRange_mm.begin();
  double WeightRGB[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  cout << "Display All Elements" << endl;
  for (i = 0; i < (int)MatRange_mm.size(); i++, currPt_it++) {
    MatRange = (*currPt_it).second;
    cout << "Range " << i << " ";
    if (sizeof(MatRange->DataMin) == 1)
      cout << "MinMax = " << (int)MatRange->DataMin << " "
           << (int)MatRange->DataMax << " ";
    else
      cout << "MinMax = " << MatRange->DataMin << " " << MatRange->DataMax
           << " ";
    cout << "1st & 2nd D = " << MatRange->AveFirstD << " ";
    cout << MatRange->AveSecondD << " ";
    cout << "# B = " << MatRange->NumBoundaries << " ";
    cout << "ith = " << (int)MatRange->ithMaterial << "/";
    cout << (int)MatRange->TotalNumMaterials;

    TotalNumMat = (int)MatRange->TotalNumMaterials;
    ithMat = (int)MatRange->ithMaterial;

    cout << "Color (RGB) = ";
    cout << (ithMat + 1) / (TotalNumMat * WeightRGB[ithMat % 3][0]) << " ";
    cout << (ithMat + 1) / (TotalNumMat * WeightRGB[ithMat % 3][1]) << " ";
    cout << (ithMat + 1) / (TotalNumMat * WeightRGB[ithMat % 3][2]) << " ";
    cout << endl;
  }
  cout.flush();
}

// Formated Display for the unsigned char data type
template <class _DataType>
void cEvaluation<_DataType>::DisplayMatRangeInfoFormated() {
  int i, j, TotalNumMat, ithMat;
  sMatRangeInfo<_DataType> *MatRange;
  class map<double, sMatRangeInfo<_DataType> *>::iterator currPt_it =
      MatRange_mm.begin();
  double WeightRGB[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  printf("Display All Elements\n");
  for (i = 0; i < (int)MatRange_mm.size(); i++, currPt_it++) {
    MatRange = (*currPt_it).second;
    printf("Range %2d ", i);
    printf("MinMax = %3d %3d, ", (int)MatRange->DataMin,
           (int)MatRange->DataMax);
    printf("1st & 2nd D = %7.3f %7.3f, ", MatRange->AveFirstD,
           MatRange->AveSecondD);
    printf("# B = %8d ", MatRange->NumBoundaries);
    printf("ith Mat = %2d/%2d ", MatRange->ithMaterial,
           MatRange->TotalNumMaterials);
    printf("Color(RGB) = ");

    TotalNumMat = (int)MatRange->TotalNumMaterials;
    ithMat = (int)MatRange->ithMaterial;

    printf("(%3d,", (int)(((double)ithMat + 1) * 255 / TotalNumMat *
                          WeightRGB[ithMat % 3][0]));
    printf("%3d,", (int)(((double)ithMat + 1) * 255 / TotalNumMat *
                         WeightRGB[ithMat % 3][1]));
    printf("%3d)", (int)(((double)ithMat + 1) * 255 / TotalNumMat *
                         WeightRGB[ithMat % 3][2]));
    printf("\n");
  }
  fflush(stdout);

  // Print the numbers from 0 to 255 with 4 width of each
  printf("   ");
  for (i = 0; i < 256; i += 10)
    printf("%-10d", i / 10);
  printf("\n");
  printf("   ");
  for (i = 0; i < 256; i++)
    printf("%1d", i % 10);
  printf("\n");

  currPt_it = MatRange_mm.begin();
  for (i = 0; i < (int)MatRange_mm.size(); i++, currPt_it++) {
    MatRange = (*currPt_it).second;
    printf("%2d ", i);
    for (j = 0; j < (int)MatRange->DataMin; j++)
      printf(" ");
    for (; j <= (int)MatRange->DataMax; j++)
      printf("-");
    printf("\n");
  }
  fflush(stdout);
}

template class cEvaluation<unsigned char>;
template class cEvaluation<unsigned short>;
template class cEvaluation<int>;
template class cEvaluation<float>;
