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

#ifndef FILE_FRONT_PLANE_H
#define FILE_FRONT_PLANE_H

#include <PEDetection/Stack.h>

#define CLASS_UNKNOWN 0x0000 // 0001 0000 0000 0000 U

#define CLASS_HEART 0x0001   // 0000 0000 0000 0001 H
#define CLASS_ARTERY 0x0002  // 0000 0000 0000 0010 A
#define CLASS_VEIN 0x0004    // 0000 0000 0000 0100 V
#define CLASS_REMOVED 0x0008 // 0000 0000 0000 1000 R

#define CLASS_ARTERY_ROOT 0x0010 // 0000 0000 0001 0000 O
#define CLASS_ARTERY_DOWN 0x0020 // 0000 0000 0010 0000 O
#define CLASS_ARTERY_LEFT 0x0040 // 0000 0000 0100 0000 O
#define CLASS_ARTERY_RGHT 0x0080 // 0000 0000 1000 0000 O

#define CLASS_DEADEND 0x0100     // 0000 0001 0000 0000 D
#define CLASS_HEARTBRANCH 0x0200 // 0000 0010 0000 0000 B
#define CLASS_LIVEEND 0x0400     // 0000 0100 0000 0000 T
#define CLASS_HITHEART 0x0800    // 0000 1000 0000 0000 M

#define CLASS_NOT_CONNECTED                                                  \
  0x1000 // 0001 0000 0000 0000 N --> Does not connected to the heart
#define CLASS_NEW_SPHERE 0x1000 // 0001 0000 0000 0000 N
#define CLASS_JUNCTION 0x2000   // 0010 0000 0000 0000 J
#define CLASS_LEFT_LUNG 0x4000  // 0100 0000 0000 0000 F
#define CLASS_RIGHT_LUNG 0x8000 // 1000 0000 0000 0000 R

#define VOXEL_ZERO_0 0
#define VOXEL_ZERO_TEMP_10 10
#define VOXEL_EMPTY_TEMP_30 30
#define VOXEL_EMPTY_50 50
#define VOXEL_BOUNDARY_HEART_BV_60 60 // 60 - 90
#define VOXEL_BOUNDARY_HEART_BV_90 90
#define VOXEL_LUNG_100 100
#define VOXEL_HEART_OUTER_SURF_120 120
#define VOXEL_HEART_SURF_130 130
#define VOXEL_HEART_TEMP_140 140
#define VOXEL_HEART_150 150
#define VOXEL_MUSCLES_170 170
#define VOXEL_TEMP_JCT1_180 180
#define VOXEL_TEMP_JCT2_190 190
#define VOXEL_STUFFEDLUNG_200 200
#define VOXEL_VESSEL_OUTER_SURF_220 220
#define VOXEL_VESSEL_LUNG_230 230
#define VOXEL_VESSEL_INIT_255 255

#define SKELETON_MAX 255
#define SKELETON_LINE 254

#define SKELETON_ARTERY_MAX 253
#define SKELETON_ARTERY_LINE 252
#define SKELETON_ARTERY_BRANCH 251
#define SKELETON_ARTERY_TEMP 250

#define SKELETON_VEIN_MAX 245
#define SKELETON_VEIN_LINE 244
#define SKELETON_VEIN_BRANCH 243
#define SKELETON_VEIN_TEMP 242

// Six Planes
// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

// void cVesselSeg<_DataType>::Stuffing_Lungs()
// LungSegmented_muc[]
// 255 = Blood Vessels --> Inside lung (200) or Outside Lungs (50)
// 200 = Inside Lung Blood Vessels + Bronchia
// 150 = Heart
// 100 = Two Lungs

#define MAXNUM_NEXT 5

class cSkeletonVoxel {
public:
  int Xi, Yi, Zi;
  int Generation_i;
  char AV_c;
  char LR_c;
  char End_c;
  int PrevVoxel_i;
  int NextVoxels_i[MAXNUM_NEXT];

  int BVCenter_LocXYZ_i[3];  // The center that maximize the radius
  int BVMaxR_i;              // Max size in the blood vessel ranges
  int BVMCenter_LocXYZ_i[3]; // The center in blood vessel and muscle ranges
  int BVMMaxR_i;             // Max size in the blood vessel and muscle ranges

public:
  cSkeletonVoxel();
  ~cSkeletonVoxel();

  void set(int X, int Y, int Z);
  void set(int X, int Y, int Z, int Loc);
  void set(int X, int Y, int Z, char End);
  void setStartEnd(char StarOrEnd_c) { End_c = StarOrEnd_c; };
  void setAVType(char AVType_c) { AV_c = AVType_c; };
  void setLRRype(char LRType_c) { LR_c = LRType_c; };
  void getXYZ(int &X, int &Y, int &Z);
  void getXYZ(int *XYZ3) {
    XYZ3[0] = Xi;
    XYZ3[1] = Yi;
    XYZ3[2] = Zi;
  };
  void getXYZ(float &Xf, float &Yf, float &Zf);

  int getNumNext();
  int DoesExistNext(int Loc);
  int DoesMatchType(cSkeletonVoxel *source);
  void AddNextVoxel(int Loc);
  void RemoveNextID(int Loc);

  void Copy(cSkeletonVoxel *Source);
  void Display();
};

class cSeedPtsInfo {
public:
  int MovedCenterXYZ_i[3];
  float Ave_f, Std_f, Median_f;
  float Min_f, Max_f;
  int MaxSize_i, Type_i;

  int BVCenter_LocXYZ_i[3]; // The center that maximize the radius
  int BVMaxR_i;             // Max size in the blood vessel ranges

  int BVMCenter_LocXYZ_i[3]; // The center in blood vessel and muscle ranges
  int BVMMaxR_i;             // Max size in the blood vessel and muscle ranges

  cStack<int> ConnectedNeighbors_s; // Seed point indexes
  int TowardHeart_SpID_i;
  int NumOpenVoxels_i;
  int CCID_i;
  int LoopID_i;
  float Direction_f[3];
  unsigned char LungSegValue_uc;
  int Traversed_i;
  int NumBloodVessels_i;
  int Generations_i;
  int General_i;

  float AveR_f, AveBVR_f, AveBVMR_f;

public:
  cSeedPtsInfo();
  ~cSeedPtsInfo();

  int getNextNeighbor(int PrevSphereID);
  int getNextNeighbors(int PrevSphereID, cStack<int> &Neighbors_ret);
  int getNextNeighbors(int PrevSpID1, int PrevSpID2,
                       cStack<int> &Neighbors_ret);
  int getNextNeighbors(int PrevSpID1, int PrevSpID2);
  void Copy(cSeedPtsInfo &src);
  void Copy(cSeedPtsInfo *src);

  void Init();
  void Init(int Type);
  void DisplayType();
  int getType(char Type_ret[]);

  int ComputeDistance(cSeedPtsInfo &src);

  void Display();
};

class cCCInfo {
public:
  int CCID_i;
  cStack<int> ConnectedPts_s;     // Seed point indexes
  int NumLines_i;                 // # of line structures
  int NumOutsideLungs_i;          // # of outside lungs
  int MinR_i, MaxR_i;             // Radius
  int MinN_i, MaxN_i;             // Number of Neighbors
  int MinLungSeg_i, MaxLungSeg_i; // 200 = Inside Lungs
                                  // 100 = Two Lungs
                                  //  50 = Outside Lungs

public:
  cCCInfo();
  ~cCCInfo();
  void Copy(cCCInfo *Src);
  void Copy(cCCInfo &Src);
};

class cFrontPlane {

public:
  int WaveTime_i;
  int TotalSize;
  float CenterPt_f[6 * 3];
  cStack<int> VoxelLocs_s[6];
  int Histogram_i[256];
  float AveIntensity_f;
  float Std_f;

public:
  cFrontPlane();
  ~cFrontPlane();

  void getAveStd(float &Ave_ret, float &Std_ret);

  void Init();
  void AddToHistogram(int Value);
  void ComputeAveStd();
  void Copy(cFrontPlane *Source);
  void CopyIthStack(int ith, cStack<int> &Stack);
  void DeleteStack();

  void Display();
};

#endif
