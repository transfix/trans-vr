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

#ifndef FILE_GVF_H
#define FILE_GVF_H

// Gradient Vector Field Class
#include <PEDetection/Geometric.h>
#include <map.h>

#define GAUSSIAN_WINDOW 5
#define SIGMA 1.5
#define GVF_ITERATION 10

template <class _DataType> class cGVF {
protected:
  _DataType *Data_mT;
  float *TimeValues_mf;
  float *Gradient_mf;
  Vector3f *Velocity_mvf;
  float MinData_mf, MaxData_mf;
  int Width_mi, Height_mi, Depth_mi;
  int WtimesH_mi, WHD_mi;
  float Gauss_1D[2 * GAUSSIAN_WINDOW + 1];
  float DoG_1D[2 * GAUSSIAN_WINDOW + 1];
  int CurrTime_mi;
  int *SeedPoint_mi;
  int Computing_GradMVector_Done_mi;

  // Found Seed Locations
  map<int, _DataType> FoundSeedPtsLocations_mm;

  // Inside + Outside Boundary Locations
  map<int, _DataType> InsideBoundaryLocations_mm;
  int NumInsideBoundaryLocations_mi;

  // Stopped Boundary Locations
  map<int, _DataType> StoppedOutsideBoundaryLocations_mm;

  // Outside Real Boundary Locations
  map<int, _DataType> OutsideBoundaryLocations_mm;

  // Active Boundary Locations
  map<int, _DataType> CurrBoundaryLocations_mm;

  // Active Boundary Locations
  map<int, _DataType> FlowedAreaLocations_mm;

public:
  cGVF();
  ~cGVF();

public:
  void setData(_DataType *Data, float Min, float Max);
  void setGradient(float *Gradient) { Gradient_mf = Gradient; };
  // Calculate Gradient Vectors and GVF
  void setWHD(int W, int H, int D);
  void setWHD(int W, int H, int D, char *GVFFileName);

  int getNumBoundaryPoints() { return CurrBoundaryLocations_mm.size(); };

public:
  int getWidth() { return Width_mi; };
  int getHeight() { return Height_mi; };
  int getDepth() { return Depth_mi; };
  // Compute Gradient Vectors from image: Coordinates are in the image space
  float *getGradientVectors();
  float *getGradientMVectors(char *GVFFileName);

public:
  int *getFlowedAreaLocations();   // Compute the flowed area without Boundary
  int getNumFlowedAreaLocations(); // Without Boundary
  int getNumInsideBoundaryLocations(); // The # locations of inside boundary
private:
  void AddFlowedAreaLocations(int *FlowedLoc);
  void InitTimeValues();

public:
  unsigned char *BoundaryMinMax(int NumSeedPts, int *Pts, _DataType &Min,
                                _DataType &Max, _DataType &Ave);
  unsigned char *CWaterFlowMinMax(int NumSeedPts, int *Pts, _DataType &Min,
                                  _DataType &Max);
  void SetSeedPts(int NumSeedPts, int *PtsXYZ);

protected:
  void setTimeValueAt(int *Loc, float value);
  float getTimeValueAt(int *Loc);
  void FindNeighbors4Or6(int *CenterLoc, int *Neighbors);
  void FindNeighbors8Or26(int *CenterLoc, int *Neighbors);
  void AddActiveBoundary(int *BoundaryLoc);
  void AddInsideAndOnBoundary(int *BoundaryLoc);
  void UpdateTimeValueAt(int *CenterLoc, float AddedTime);

  void ComputeGradientVFromGradientM();
  void ComputeGradientVFromData();

  void GV_Diffusion();                              // for 2D
  void Anisotropic_Diffusion_3D(int NumIterations); // for 3D
  void Gaussian();
  // Find the minimum time value and remove it from the neighbor list
  int FindMinTimeValueLocation(int *MinTimeLoc);
  _DataType getDataAt(int *Loc);
  void AddStoppedOutsideBoundary(int *BoundaryLoc);
  void AddOutsideBoundary(int *BoundaryLoc);

public:
  int FindSeedPts();
  int *getFoundSeedPtsLocations();

private:
  void AddFoundSeedPts(int *SeedPtsLoc);

public:
  void SaveGradientVectors(char *filename);

private:
  int IsBoundary(int X, int Y, int Z);
  int IsSeedPoint(int X, int Y, int Z);
  int IsInsideBoundary(int X, int Y, int Z);
  int IsOutsideBoundary(int Xi, int Yi, int Zi);
  int IsStoppedOutsideBoundary(int Xi, int Yi, int Zi);

  // Quick Sort Algorithm for locations and Related Functions
private:
  void QuickSortLocations(int *Locs, int NumLocs, char Axis1, char Axis2,
                          char Axis3);
  void QuickSortLocations(int *Locs, int NumLocs, char Axis);
  void QuickSortLocs(int *Locs, int p, int r);
  void SwapLocs(int &x, int &y);
  int PartitionLocs(int *Locs, int low, int high);

  // Quick Sort Algorithm for Intensity Values and Related Functions
private:
  void QuickSortIntensities(_DataType *Intensities, int NumData);
  void QuickSortIntensities(_DataType *Intensities, int p, int r);
  void SwapIntensities(_DataType &x, _DataType &y);
  int PartitionIntensities(_DataType *Intensities, int low, int high);

private:
  void ComputeBoundary();
  void DisplayLocations(int *Locs, int NumLocs, char *Str);
  void DisplayLocationsFormated(int *Locs, int NumLocs, char *Str);

public:
  void SaveSeedPtImages(int *TempSeedPts, int NumSeedPts, _DataType Min,
                        _DataType Max);
  int Index(int X, int Y, int Z);
  void Destroy();
};

// extern char				*InputFileName_gc, *TargetName_gc;
extern char TargetName_gc[512], InputFileName_gc[512];
void SaveImage(int Width, int Height, unsigned char *Image, char *PPMFile);

#endif
