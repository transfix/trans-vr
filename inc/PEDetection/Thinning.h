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

#ifndef FILE_THINNING_H
#define FILE_THINNING_H

#include <map.h>

#define MAX_NUM_MATERIALS 30
#define MAXNUM_HISTOGRAM_ELEMENTS 5000

// Marking MatNumVolume_muc[]
#define MAT_ZERO_CROSSING 0x80      // 1000 0000
#define MAT_BOUNDARY_HAS_ALPHA 0x40 // 0100 0000
#define MAT_BOUNDARY 0x20           // 0010 0000
#define MAT_INSIDE_BOUNDARY 0x10    // 0001 0000 Read Material Body

#define MAT_OTHER_MATERIALS 0x08 // 0000 1000
#define MAT_CHECKED_VOXEL 0x01   // 0000 0001

template <class _DataType> class cThinning {
protected:
  int NumMaterial_mi; // the number of Clusters
  int *Histogram_mi;
  float HistogramFactorI_mf;
  float HistogramFactorG_mf;

  int Width_mi, Height_mi, Depth_mi;
  int WtimesH_mi, WHD_mi;

  _DataType *Data_mT;
  float MinData_mf, MaxData_mf;

  float *MaterialProb_mf;

  unsigned char *ThreeColorVolume_muc;
  unsigned char *ConnectedComponentVolume_muc;

public:
  map<int, int> EndVoxelStack_mm; // First = Start Point, Second = End Point

public:
  cThinning();
  ~cThinning();

  void setData(_DataType *Data, float Minf, float Maxf);
  void setGradient(float *Grad, float Minf, float Maxf);
  void setGradientVectors(float *GradVec);
  void setWHD(int W, int H, int D);
  void setProbabilityHistogram(float *Prob, int NumMaterial, int *Histo,
                               float HistoF);
  unsigned char *getCCVolume();

public:
  void ConnectedComponents(char *OutFileName, _DataType MatMin,
                           _DataType MatMax);
  void ConnectedComponents2(char *OutFileName, _DataType MatMin,
                            _DataType MatMax);
  void Thinning4Subfields(char *OutFileName, _DataType MatMin,
                          _DataType MatMax);
  void Skeletonize(char *OutFileName, _DataType MatMin, _DataType MatMax);

private:
  void Reduce_ORTH_DIAG_Nonend_Voxels(int Px, int Py, int Pz,
                                      int &NumRepeat_Ret);
  int Is_DiametricallyAdjacent(int Xi, int Yi, int Zi,
                               unsigned char *Cube_uc);
  int Is_DiametricallyAdjacent(unsigned char *Cube_uc);
  int Is_ORTH_Reducible(int Xi, int Yi, int Zi, unsigned char *Cube_uc);
  int Is_DIAG_Reducible(int Px, int Py, int Pz, unsigned char *Cube_uc);
  int Is_ORTH_Edge(int Xi, int Yi, int Zi, unsigned char *Cube_uc);
  void RotateCube_ORTH(int RotationNumber, unsigned char *Cube27);
  void RotateCube_DIAG(int RotationNumber, unsigned char *Cube27);
  void RotateAroundX(unsigned char *Cube27);
  void RotateAroundY(unsigned char *Cube27);
  void RotateAroundZ(unsigned char *Cube27);
  void SymmetryAlongX(unsigned char *Cube27);
  void SymmetryAlongY(unsigned char *Cube27);
  void SymmetryAlongZ(unsigned char *Cube27);
  int Index(int X, int Y, int Z);
  void InitThreeColorVolume(_DataType MatMin, _DataType MatMax);
  int ConnectedTo_lwp(unsigned char *Cube27);
  int ConnectedTo_lwp2(unsigned char *Cube27);
  void Remove_All_Weakly_End_Voxels(int Px, int Py, int Pz,
                                    int &NumRepeat_Ret);
  int Is_Adjacent_Two_1_Voxels(int Px, int Py, int Pz);
  int Is_Adjacent_Three_1_Voxels(int Px, int Py, int Pz);
  void All_Upper_End_Voxels_Even(int Px, int Py, int Pz, int &NumRepeat_Ret);
  void All_Lower_End_Voxels_Odd(int Px, int Py, int Pz, int &NumRepeat_Ret);

public:
  void SaveThreeColorVolume(char *OutFileName, _DataType MatMin,
                            _DataType MatMax);
  void SaveThreeColorVolume(char *OutFileName, _DataType MatMin,
                            _DataType MatMax, unsigned char *SkeletonVolume);
  void SaveThickVolume(char *OutFileName, _DataType MatMin, _DataType MatMax,
                       unsigned char *VolumeInput);
  void SaveThickVolume(char *OutFileName, _DataType MatMin, _DataType MatMax,
                       unsigned char *VolumeInput, unsigned char Threshold);
  void SaveThickSkeletons(char *OutFileName, _DataType MatMin,
                          _DataType MatMax);
  void SaveVolume(char *OutFileName, _DataType MatMin, _DataType MatMax,
                  unsigned char *VolumeInput);

  void DisplayCube(unsigned char *Cube27);
  void Destroy();
};

template <class _DataType> void QuickSort(_DataType *data, int p, int r);

template <class _DataType> void Swap(_DataType &x, _DataType &y);

template <class _DataType> int Partition(_DataType *data, int low, int high);

template <class T>
T *Bilateral_Filter(T *data, int WindowSize, double &Min, double &Max,
                    double S1, double S2);

#endif
