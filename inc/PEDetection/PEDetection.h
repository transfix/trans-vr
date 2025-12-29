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

#ifndef FILE_PEDETECTION_H
#define FILE_PEDETECTION_H

#include <PEDetection/Geometric.h>
#include <PEDetection/TFGeneration.h>
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

// class cMarchingCubes2nd;

template <class _DataType>
class cPEDetection : public cTFGeneration<_DataType> {

protected:
  unsigned char *
      CCVolume_muc; // CCVolume_muc has the location of the cThinning CCVolume
  map<int, int> EndVoxelStack_mm;

public:
  cPEDetection();
  ~cPEDetection();
  void setCCVolume(unsigned char *CCVolume);
  void CopyEndVoxelLocationsFromThinning(map<int, int> &EndVoxels);

public:
  void VesselQuantification(char *OutFileName, _DataType MatMin,
                            _DataType MatMax);
  void VesselQuantification_Manual(char *OutFileName, _DataType MatMin,
                                   _DataType MatMax);

private:
  double VesselTracking_Auto(_DataType MatMin, _DataType MatMax,
                             float *StartPt, float *EndPt);
  void VesselTracking_BoundaryExtraction(_DataType MatMin, _DataType MatMax,
                                         float *StartPt, float *EndPt);
  void VesselTracking_BoundaryExtraction2(_DataType MatMin, _DataType MatMax,
                                          float *StartPt, float *EndPt);
  void VesselTracking_backup(_DataType MatMin, _DataType MatMax,
                             float *StartPt, float *EndPt);
  int getANearestBoundary(double *StartPt, double *Ray, double Increase,
                          _DataType MatMin, _DataType MatMax);
  double FindBranches(_DataType MatMin, _DataType MatMax, int *StartPt,
                      int *EndPt);
  void ExtractingCC(map<int, int> &CCElements_m, int *StartPt, int *EndPt,
                    int *BranchStartPt_Ret, map<int, int> &Branches_m_Ret);

  int getANearestBoundary_BasedOnSecondD(double *StartPt, double *Ray,
                                         double Increase, _DataType MatMin,
                                         _DataType MatMax, double MinRange);
  int getANearestBoundary_BasedOnClassifiation(double *StartPt, double *Ray,
                                               double Increase,
                                               _DataType MatMin,
                                               _DataType MatMax,
                                               double MinRange);

  void FindNewCenterLoc(map<int, unsigned char>::iterator &Locs_it, int Size,
                        double *CenterLoc);
  void FindNewCenterLoc(map<int, unsigned char>::iterator &Locs_it, int Size,
                        double *CenterLoc, int *Min, int *Max);

  void getBoundaryVoxels(double *StartPt, int NumRays, double *Rays,
                         _DataType MatMin, _DataType MatMax,
                         map<int, unsigned char> &VoxelLocs_map);
  double ComputeAveRadius(double *CurrPt, double *NextPt, _DataType MatMin,
                          _DataType MatMax);
  void ComputePerpendicular8Rays(float *StartPt, float *Dir, double *Rays);
  void ComputePerpendicular8Rays(double *StartPt, double *Dir, double *Rays);
  void UpdateMinMax(int *CurrMinXYZ, int *CurrMaxXYZ, int *NextMinXYZ,
                    int *NextMaxXYZ, int *MinXYZ_Ret, int *MaxXYZ_Ret);
  void ComputeGradientMeanStd(map<int, unsigned char> &Locs_map,
                              double &Mean_Ret, double &Std_Ret);
  double ComputeGaussianValue(double Mean, double Std, double P_x);

  void ClearAndMergeTwoMaps(map<int, unsigned char> &Map_Ret,
                            map<int, unsigned char> &Map1,
                            map<int, unsigned char> &Map2);
  void ClearAndCopyMap(map<int, unsigned char> &Dest_map,
                       map<int, unsigned char> &Source_map);
  void ArbitraryRotate(int NumRays, double *Rays, double Theta,
                       double *StartPt, double *EndPt);
  double Normalize(double *Vec);
  double Distance(double *Pt1, double *Pt2);
  int Index(int X, int Y, int Z);

  void SaveZeroCrossingVoxels(map<int, unsigned char>::iterator &Locs_it,
                              int Size);
  void SaveVesselCenterLoc(double *Center);

public:
  float *BoundaryVoxelExtraction(char *OutFilename, _DataType MatMin,
                                 _DataType MatMax);
  //		void BoundaryVoxelExtraction(char *OutFilename, _DataType
  //MatMin, _DataType MatMax, cMarchingCubes2nd& MC2nd);
private:
  void AdjustingBoundary(map<int, unsigned char> &BoundaryLocs_map);
};

#endif
