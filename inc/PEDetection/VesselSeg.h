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

#ifndef FILE_VESSEL_SEG_H
#define FILE_VESSEL_SEG_H

#include <PEDetection/FrontPlane.h>
#include <PEDetection/Geometric.h>
#include <PEDetection/MarchingCubes.h>
#include <PEDetection/Stack.h>
#include <PEDetection/TFGeneration.h>
#include <PEDetection/Timer.h>
#include <deque.h>
#include <map.h>
// #include "SphereIndex.h"

// SphereR01_i[] --> R = 1 # voxels = 6  	int NumSphereR01_gi = 6  ;
// SphereR02_i[] --> R = 2 # voxels = 26 	int NumSphereR02_gi = 26 ;
// SphereR03_i[] --> R = 3 # voxels = 90 	int NumSphereR03_gi = 90 ;
// SphereR04_i[] --> R = 4 # voxels = 134 	int NumSphereR04_gi = 134;
// SphereR05_i[] --> R = 5 # voxels = 258 	int NumSphereR05_gi = 258;
// SphereR06_i[] --> R = 6 # voxels = 410 	int NumSphereR06_gi = 410;
// SphereR07_i[] --> R = 7 # voxels = 494 	int NumSphereR07_gi = 494;
// SphereR08_i[] --> R = 8 # voxels = 690 	int NumSphereR08_gi = 690;

// Debugging Options

// #define DEBUG_SEEDPTS_SEG
#define DEBUG_ARTERY_VEIN

#define DEBUG_SKELETONIZATION
#define DEBUG_REGION_GROWING
#define DEBUG_MERGING

// The minimum value of gradient magnitude
// to compute the second derivative and for wave propagation
#define MIN_GM 5
#define GM_THRESHOLD 11.565
#define SEARCH_MAX_RADIUS_5 5
#define WINDOW_ST_5 5 // Window size for structure tensor

struct sLineS {
  int NumHits;
  float AveDist_f;
  float AveGM_f;
  float Direction_f[3];
  float CenterPt_f[3];
};

template <class _DataType> class cVesselSeg {

protected:
  int NumMaterial_mi; // the number of clusters
  int *Histogram_mi;
  float HistogramFactorI_mf;
  float HistogramFactorG_mf;

  int Width_mi, Height_mi, Depth_mi;
  int WtimesH_mi, WHD_mi;

  _DataType *Data_mT, *ClassifiedData_mT;
  float MinData_mf, MaxData_mf;
  float *MaterialProb_mf;
  float *GradientMag_mf, MinGrad_mf, MaxGrad_mf;
  float *GradientVec_mf;
  float *SecondDerivative_mf, MinSecond_mf, MaxSecond_mf;
  char *OutFileName_mc;
  float *GradMVectors_mf;
  int FlipX_mi;

  double GaussianKernel3_md[3][3][3], GaussianKernel5_md[5][5][5];
  double GaussianKernel7_md[7][7][7], GaussianKernel9_md[9][9][9];
  double GaussianKernel11_md[11][11][11], GaussianKernel13_md[13][13][13];
  double GaussianKernel15_md[15][15][15], GaussianKernel17_md[17][17][17];
  double GaussianKernel19_md[19][19][19], GaussianKernel21_md[21][21][21];
  double CubeData5_md[5 * 5 * 5], CubeData7_md[7 * 7 * 7],
      CubeData9_md[9 * 9 * 9];
  double CubeData11_md[11 * 11 * 11], CubeData13_md[13 * 13 * 13];
  double CubeData15_md[15 * 15 * 15], CubeData17_md[17 * 17 * 17];
  double CubeData19_md[19 * 19 * 19], CubeData21_md[21 * 21 * 21];

  float SpanX_mf, SpanY_mf, SpanZ_mf;

  int *Distance_mi;
  unsigned char *LungSegmented_muc;
  int *Wave_mi;

  // Geometric Data
  int NumVertices_mi, NumTriangles_mi;
  float *Vertices_mf;
  int *Triangles_mi;

  // Seed Point Locations(X, Y, Z) and the number of seed points
  int *SeedPts_mi, NumSeedPts_m;
  map<int, unsigned char> SeedPts_mm;

  int NumSegmentedVoxels_mi, NumLineStructureVoxels_mi;
  int NumCenterVoxels_mi;
  int MaxNumBranches_mi, CurrNumBranches_mi;

  cStack<int> NextPlane_mstack[6];
  int Range_BloodVessels_mi[2];
  int Range_Soft_mi[2];
  int Range_Muscles_mi[2];
  int Range_Lungs_mi[2];
  int MaxNumSeedPts_mi, NumSeedPts_mi, *SeedPtsXYZ_mi;
  cStack<int> SeedPts_mstack; // Seed Pts which are inside lungs
  float MeanBloodVessels_mf, StdBooldVessels_mf;
  cSeedPtsInfo *SeedPtsInfo_ms;
  int CCID_mi;
  unsigned char *LineSegments_muc;
  cCCInfo *CCInfo_ms;
  int CurrEmptySphereID_mi;
  int LoopID_mi;
  int MaxRadius_mi;
  int ArteryRootSpID_mi;
  int ArteryLeftBranchSpID_mi[20], ArteryRightBranchSpID_mi[20];
  Vector3f ArteryLeftDir_mvf, ArteryRightDir_mvf;

  unsigned char ColorTable_muc[256][3];
  int OutputFileNum_mi;
  double DotMapCount_mi;

  // SAVE_Geometry_data
  cMarchingCubes<float> MC_m;
  // PE_DETECTION
  unsigned char *PE_Volume_muc;

  // For debugging
  cStack<int> Line_mstack, Line1_mstack, Line2_mstack, Line3_mstack;
  cStack<int> MultiBRemoved_mstack, MultiBCenters_mstack;
  cStack<int> CheckSpIDs_mstack;
  int BranchHisto_mi[20];

public:
  cVesselSeg();
  ~cVesselSeg();

  void setData(_DataType *Data, float Minf, float Maxf, int FlipX);
  void setGradient(float *Grad, float Minf, float Maxf);
  void setGradientVectors(float *GradVec);
  void setWHD(int W, int H, int D);
  void setProbabilityHistogram(float *Prob, int NumMaterial, int *Histo,
                               float HistoF);
  void setSecondDerivative(float *SecondD, float Min, float Max);
  void setXYZSpans(float SpanX, float SpanY, float SpanZ);

  unsigned char *getPEVolume() { return &PE_Volume_muc[0]; };

public:
  void VesselExtraction(char *OutFileName, _DataType LungMatMin,
                        _DataType LungMatMax, _DataType MSoftMatMin,
                        _DataType MSoftMatMax, _DataType MMuscleMatMin,
                        _DataType MMuscleMatMax, _DataType VesselMatMin,
                        _DataType VesselMatMax);

protected:
  void BiggestDistanceVoxels(cStack<int> &BDistVoxels_s);

  int ExtractLungOnly(int LocX, int LocY, int LocZ);
  void LungBinarySegment(_DataType LungMatMin, _DataType LungMatMax);
  void LungBinarySegment2(_DataType LungMatMin, _DataType LungMatMax);
  void MedianFilterForLungSegmented(int NumRepeat);
  void Vessel_BinarySegment(_DataType VesselMatMin, _DataType VesselMatMax);
  void Removing_Outside_Lung();
  void MovingPlanesX(int *MovingPlanePosX, int *MovingPlaneNegX, int EndPos_i,
                     int EndNeg_i, _DataType Range1_i, _DataType Range2_i,
                     int GridSize_i);
  void MovingPlanesY(int *MovingPlanePosY, int *MovingPlaneNegY, int EndPos_i,
                     int EndNeg_i, _DataType Range1_i, _DataType Range2_i,
                     int GridSize_i);
  int ComputingNumRemovedVoxels(_DataType Range1_i, _DataType Range2_i,
                                double &Ratio_d_ret);

  void Complementary_Union_Lung_BloodVessels();

  void SaveSecondD_Lung();
  void ComputeSaddlePoint(float *T8, float &DistCT, float &GT);
  void RemoveSecondDFragments(int MinNumNegVoxels, float Sign_PN);
  void Cleaning_Phantom_Edges();
  void Remove_Phantom_Triangles(_DataType LungMatMin, _DataType LungMatMax,
                                _DataType VesselMatMin,
                                _DataType VesselMatMax);
  void Rearrange_Triangle_Indexes();
  void Remove_Triangle_Fragments();
  void Finding_Triangle_Neighbors();
  int ClassifyBranch(int PrevSpID, int CurrSpID, Vector3f &Direction,
                     int MinZ, cStack<int> &Line_stack_ret, Vector3f &Ave_ret,
                     float &MinDot_ret);
  void Removing_Multi_Branches();

protected:
  void ComputeDistanceTransform(int *DistVolume_i);
  void SmoothingClassifiedData(int WindowSize);
  void GaussianSmoothing3D(_DataType *data, int WindowSize);
  void ComputeGaussianKernel();

protected:
  int IsLineStructure(int Xi, int Yi, int Zi, float *DirVec_ret,
                      int WindowSize);
  int IsLineStructureDist(int Xi, int Yi, int Zi, float *DirVec_ret,
                          int WindowSize);
  int IsLineStructureOnlyCC(int Xi, int Yi, int Zi, float *DirVec_ret,
                            int WindowSize);

  void EigenDecomposition(float *Mat, float *Eigenvalues,
                          float *Eigenvectors);
  int SkeletonTracking_Marking(int EndLoc);

protected:
  void ComputeDistanceVolume(double GradThreshold);
  void ComputeGVF();
  void ComputeSkeletons(int RootLoc, int EndLoc, int Threshold_Dist);

private:
  void ComputePerpendicular8Rays(double *StartPt, double *Direction,
                                 double *Rays);
  void ComputeRadius(double *StartPt, double *Rays8, double *HitLocs8,
                     double *Radius8);
  int getANearestZeroCrossing(double *CurrLoc, double *DirVec,
                              double *ZeroCrossingLoc_Ret,
                              double &FirstDAtTheLoc_Ret,
                              double &DataPosFromZeroCrossingLoc_Ret);
  void ComputingCCInfo();

  float Normalize(float *Vec);
  double Normalize(double *Vec);
  double DataInterpolation(double *LocXYZ);
  double DataInterpolation(double LocX, double LocY, double LocZ);
  double GradientInterpolation(double *LocXYZ);
  double GradientInterpolation(double LocX, double LocY, double LocZ);
  double GradientInterpolation2(double LocX, double LocY, double LocZ);
  double SecondDInterpolation(double *LocXYZ);
  double SecondDInterpolation(double LocX, double LocY, double LocZ);
  double SecondDInterpolation2(double LocX, double LocY, double LocZ);
  void MakeCube8Indexes(int Xi, int Yi, int Zi, int *Locs8);
  void SaveVoxels_Volume(unsigned char *Data, int LocX, int LocY, int LocZ,
                         char *Postfix, int VoxelResolution,
                         int TotalResolution);
  int GradVecInterpolation(double LocX, double LocY, double LocZ,
                           double *GradVec_Ret);
  double TrilinearInterpolation(unsigned char *Data, double LocX, double LocY,
                                double LocZ);

private:
  int Index(int X, int Y, int Z, int ith, int NumElements);
  int Index(int X, int Y, int Z);
  int Index(int *Loc3);
  void IndexInverse(int Loc, int &X, int &Y, int &Z);
  void IndexInverse(int Loc, int *Center3);
  void SaveInitVolume(_DataType Min, _DataType Max);
  void SaveDistanceVolume();
  void Display_Distance(int ZPlane);
  void Display_Distance2(int ZPlane);
  void DisplayKernel(int WindowSize);
  void Display_XYZLocs(int CurrSpID_i, cStack<int> *Loc_stack);

private:
  void FollowDirection(float *FlowVector3, float Direction_f, int CurrSpID_i);
  float FlowDot(int JCT1SpID_i, int JCT2SpID_i);
  double ComputeDistance(int *Center13, int *Center23);
  float ComputeDistance_f(int *Center13, int *Center23);
  void ComputePrevToCurrVector(int CurrSpID, Vector3f &Dir_ret,
                               int DoWeightR_i);
  void ComputePrevToCurrVector(cStack<int> &PrevSpIDs_stack, int CurrSpID_i,
                               Vector3f &Dir_ret);
  void ComputePrevToCurrVector2(cStack<int> &PrevSpIDs_stack, int CurrSpID_i,
                                Vector3f &Dir_ret);
  int NumContinuousBV_Voxels(int *CurrCenter3, int *NextCenter3, int Length);
  int NumContinuousPureBV_Voxels(int *CurrCenter3, int *NextCenter3,
                                 int Length);
  int FindNextAvaiableCenters(int CurrSpID_i, int ExstSpID_i,
                              int MaxNumCenters, cStack<int> &Interval_stack,
                              cStack<int> &CentersFromCurr_stack);
  int FindNextAvaiableCentersNearHeart(int CurrSpID_i, int ExstSpID_i,
                                       int MaxNumCenters,
                                       cStack<int> &Interval_stack,
                                       cStack<int> &CentersFromCurr_stack);
  int ComputeContinuousBranch(int PrevSpID_i, int CurrSpID_i, int NextSpID_i);
  void ComputeIntervals(int CurrSpID_i, cStack<int> &CentersFromCurr_stack,
                        cStack<int> &Interval_stack_ret);
  void ComputeHighProb(int CurrSpID_i, Vector3f &Dir_vec,
                       cStack<int> &NewCenter_stack, int MaxNumBV_ret,
                       cStack<int> &HighProbCenter_stack_ret);
  int SearchingSingleB(map<int, cStack<int> *> &SpID_SingleB_map,
                       map<int, cStack<int> *> &SpID_MultiB_map);
  int MoveBiggestDegreeMultiB_To_SingleB(
      map<int, cStack<int> *> &SpID_SingleB_map,
      map<int, cStack<int> *> &SpID_MultiB_map, int MinR_i);
  void SearchAndConnectMissingBranches(int CurrSpID_i, int MinDegLoc_i,
                                       int MaxDegLoc_i);
  int Computing_NumOfBackwardVoxels(int *Center3, int CurrR,
                                    cStack<int> &PrevSpIDs_stack,
                                    cStack<int> *NextLocs_stack_ret);
  int Computing_NumOfBranches(int CurrSpID_i, cStack<int> *NextLocs_stack_ret,
                              int &HitTheHeart_ret);
  int Computing_NumOfBranches(int CurrSpID_i, cStack<int> *NextLocs_stack_ret,
                              int &HitTheHeart_ret, int DoSaveNumVoxels_i);
  int Computing_NumOfBranches(int CurrSpID_i, cStack<int> *NextLocs_stack_ret,
                              float MinDegrees_f, int DoSaveNumVoxels_i,
                              cStack<int> *Pts_GT_MinDegrees_stack_ret,
                              int &HitTheHeart_ret);
  int Computing_NumOfBranches_NearHeart(int CurrSpID_i,
                                        cStack<int> *NextLocs_stack_ret);

  void BoundaryRedefinition();
  int ComputeDirectionOutside(int *CurrCenter3, int *NextPt3,
                              unsigned char *DistVolume_uc,
                              Vector3f &Dir_vec_ret, int *EndPt3_ret,
                              cStack<int> &Line_Stack_ret);
  int ComputeDirectionInside(int *CurrCenter3, int *NextPt3,
                             unsigned char *DistVolume_uc,
                             Vector3f &Dir_vec_ret, int *EndPt3_ret,
                             cStack<int> &Line_Stack_ret);
  int FindBranchType(Vector3f &Dir_vec);
  int FindArteryRightBranches(Vector3f &Dir_vec);
  int AddSpheresToTheEnd(int CurrSpID, Vector3f &MinDeg_vec,
                         unsigned char *Dist_uc, int Sphere_Type_i,
                         cStack<int> &RightAdj_stack_ret);
  int SearchSkeletonLines(int CurrLoc, Vector3f &Line_vec,
                          unsigned char *Skeleton_uc, int MaxNumVoxels,
                          int MinTime_i, float MinDegrees_f,
                          unsigned char *Wave_uc,
                          cStack<int> &Line_stack_ret);
  int SearchNearestVein(int CurrLoc, Vector3f &Line_vec,
                        unsigned char *Skeleton_uc, float MinDegrees_i,
                        unsigned char *Wave_uc, int &RightVeinSpID_ret,
                        int *NextCenter3_ret, cStack<int> &Line_stack_ret);
  int AddSpheresToTheVeinEnd(int CurrSpID, Vector3f &MinDeg_vec,
                             unsigned char *Dist_uc, int Sphere_Type_i,
                             cStack<int> &Adj_stack_ret);
  int AddSpheresToTheArteryEnd(int CurrSpID, Vector3f &MinDeg_vec,
                               unsigned char *Dist_uc, int Sphere_Type_i,
                               cStack<int> &Adj_stack_ret);
  int DoesExistABranchToTheHeart(int CurrSpID);

  // Tracking Vessels
public:
  void Tracking_Vessels(cStack<int> &BiggestDist_s);
  void MovingBalls(cStack<int> &BiggestDist_s);

  void SaveTheSlice(unsigned char *SliceImage_uc, int NumRepeat,
                    int CurrTime_i, int CenterZi);
  void SaveWaveSlice(unsigned char *SliceImage_uc, int CurrTime_i, int CurrZi,
                     int MinValue, unsigned char *AdditionalVolume);
  void SaveWaveSlice2(unsigned char *SliceImage_uc, int CurrZi, int MinX,
                      int MaxX, int MinY, int MaxY, char *Postfix);

  void Compute_FlowVectors(float *CurrCenter6x3, cStack<int> *CurrWave_s);
  void ComputeFlowVectors(float *CurrCenter6x3, float *FlowVector3_ret);
  void ChangePrevToCurrFrontWave();
  void AddPrevFrontWave(int CurrTime, cStack<int> *WaveLocs_s,
                        float *Center_Mx3, int *WaveSize_Mx2,
                        int ClearWindowSize);
  int IsIncreasingVessels(int *WaveSize_Mx2);
  void ExpandingBox(int *StartCenterPt3, int CurrTime_i, int *XYZLevels6);
  void RegionGrown(int SX, int SY, int SZ, int CurrTime);
  void Compute_Histogram_Ave(cFrontPlane *CurrPlane);
  double ComputeGaussianProb(float Ave, float Std, float Value);
  void ComputeAveStd(int I1, int I2, float &Mean_ret, float &Std_ret);
  int ComputeMeanStd_Cube(int Xi, int Yi, int Zi, int HalfCubeSize,
                          float &Mean_ret, float &Std_ret, float &Min,
                          float &Max);
  int ComputeMeanStd_Sphere(int Xi, int Yi, int Zi, int Radius,
                            float &Mean_ret, float &Std_ret, float &Min_ret,
                            float &Max_ret);
  int *getSphereIndex(int SphereRadius, int &NumVoxels_ret);
  void RemovingThreeSphereLoop(int CurrSpID, int NextSpID);
  void RemovingThreeSphereLoop2(int CurrSpID, int NextSpID);

  // Seed Pts-Based Segmentation
public:
  void InitSeedPtsFromDistance(int Lower_Bound);
  void AddingMoreSeedPtsAtHeartBoundary();
  void SeedPtsClassification();
  unsigned char *SeedPtsBased_BloodVesselExtraction();
  void ComputingNeighbors(map<long int, int> &CurrBoundaryLoc_m,
                          long int TimeScale, int CurrTime);
  void ComputingMaxSphereAndFindingHeart();
  void ComputingSeedPtsInfoSphere3();
  int FindBiggestSphereLocation_MainBranches(cStack<int> &Voxel_stack,
                                             int CurrSpID_i,
                                             Vector3f &Dir_vec,
                                             int *Center3_ret,
                                             int &MaxSize_ret);
  int FindBiggestSphereLocation_ForJCT(cStack<int> &Voxel_stack,
                                       int *CurrCenter3_i, float *Dir3_f,
                                       int *Center3_ret);
  int FindBiggestSphereLocation_ForJCT2(cStack<int> &Voxel_stack,
                                        int *CurrCenter3_i, float *Dir3_f,
                                        int *Center3_ret);
  int FindBiggestSphereLocation_ForHeart(int *Start3, int *End3,
                                         int *Center3_ret, int &MaxSize_ret);
  int FindBiggestSphereLocation_ForSmallBranches(cStack<int> &Voxel_stack,
                                                 int *Center3_ret,
                                                 int &MaxSize_ret, int SpID);
  int FindBiggestSphereLocation_ForHeart(cStack<int> &Voxel_stack,
                                         int *Center3_ret, int &MaxSize_ret,
                                         int SpID_debug);
  int FindBiggestSphereLocation_ForBloodVessels(cStack<int> &Voxel_stack,
                                                int *Center3_ret,
                                                int &MaxSize_ret, int SpID);
  int FindBiggestSphereLocation(cStack<int> &Voxel_stack, int *Center3_ret,
                                int &MaxSize_ret);
  int FindBiggestSphereLocation_ForNewJCT(cStack<int> &Voxel_stack,
                                          int *Center3_ret, int &MaxSize_ret,
                                          int SpID_Debug);
  int FindBiggestSphereLocation_ForNewJCT2(cStack<int> &Voxel_stack,
                                           int *Center3_ret, int &MaxSize_ret,
                                           int SpID_Debug);
  int FindBiggestSphereLocation_TowardArtery(cStack<int> &Voxel_stack,
                                             int PrevSpID, int CurrSpID,
                                             int *Center3_ret,
                                             int &MaxSize_ret);
  int FindBiggestSphereLocation_TowardArtery2(cStack<int> &Voxel_stack,
                                              int PrevSpID, int CurrSpID,
                                              int *Center3_ret,
                                              int &MaxSize_ret);
  int FindBiggestSphereLocation_TowardArtery(cStack<int> &Voxel_stack,
                                             cStack<int> &PrevSpIDs_stack,
                                             int CurrSpID, int *Center3_ret,
                                             int &MaxSize_ret);
  int FindBiggestSphereLocation_TowardArtery2(cStack<int> &Voxel_stack,
                                              cStack<int> &PrevSpIDs_stack,
                                              int CurrSpID, int *Center3_ret,
                                              int &MaxSize_ret);
  int FindBiggestSphereLocation_AwayFromHeart(cStack<int> &Voxel_stack,
                                              int CurrSpID, Vector3f &Dir_vec,
                                              int *Center3_ret,
                                              int &MaxSize_ret);
  int FindBiggestSphereLocation_AwayFromHeart2(cStack<int> &Voxel_stack,
                                               int CurrSpID,
                                               Vector3f &Dir_vec,
                                               int *Center3_ret,
                                               int &MaxSize_ret);
  int FindBiggestSphereLocation_ForMissingBranches(cStack<int> &Voxel_stack,
                                                   int CurrSpID_i,
                                                   int *Center3_ret,
                                                   int &MaxSize_ret);
  int FindBiggestSphereLocation_ForMissingBranches2(cStack<int> &Voxel_stack,
                                                    int CurrSpID_i,
                                                    int *Center3_ret,
                                                    int &MaxSize_ret);

  void Compute3DLine(int x1, int y1, int z1, int x2, int y2, int z2,
                     cStack<int> &Voxels_ret);
  void DrawLind_3D(int x1, int y1, int z1, int x2, int y2, int z2, int color);
  void DrawLind_3D_GivenVolume(int x1, int y1, int z1, int x2, int y2, int z2,
                               int color, unsigned char *Volume);
  void ComputingCCSeedPts(int SeedPtsIdx, int CCID_i);
  void ComputeMovingDirection(cStack<int> &Contact_stack, int *SphereIndex_i,
                              double *MovingDir3_ret);
  void QuickSortCC(cCCInfo *data, int p, int r);
  void Swap(cCCInfo &x, cCCInfo &y);
  int Partition(cCCInfo *data, int low, int high);
  void Voxel_Classification();
  void Finding_Boundaries_Heart_BloodVessels();
  int AddASphere(int SphereR, int *Center3, float *Dir, int Type);
  void DeleteASphere(int SphereID);
  void DeleteASphereAndLinks(int SphereID);
  void ComputingTheBiggestSphere_For_SmallBranches_At(int Xi, int Yi, int Zi,
                                                      int &SphereR_ret);
  void ComputingTheBiggestSphereAt(int Xi, int Yi, int Zi,
                                   cStack<int> &ContactVoxels_stack_ret,
                                   int &SphereR_ret);
  void ComputingNextCenterCandidates(int *CurrCenter3, int *NextCenter3,
                                     int CurrSphereR, int NextSphereR,
                                     cStack<int> &Boundary_stack);
  void ComputingNextCenterCandidates_MultiBranches(
      int CurrSpID_i, cStack<int> &Boundary_stack_ret);
  void ComputingNextCenterCandidates_For_Heart(int *CurrCenter3,
                                               int *NextCenter3, int SphereR,
                                               cStack<int> &Boundary_stack);
  void ComputingNextCenterCandidates_For_SmallBranches(
      int *CurrCenter3, int *NextCenter3, int SphereR, int MaxDist,
      cStack<int> &Boundary_stack);
  void ComputingNextCenterCandidatesToTheHeart(int *CurrCenter3,
                                               int *NextCenter3,
                                               int CurrSphereR,
                                               int NextSphereR,
                                               cStack<int> &Boundary_stack);
  void ComputingNextCenterCandidates_ForNewJCT(int *CurrCenter3,
                                               int *NextCenter3,
                                               int CurrSphereR,
                                               int NextSphereR,
                                               cStack<int> &Boundary_stack);
  void ComputingNextCenterCandidates_ForMissingBranches(
      int CurrSpID_i, int *NextCenter3, cStack<int> &Boundary_stack);

  float ComputDotProduct(int PrevSpID_i, int CurrSpID_i, int NextSpID_i);
  float ComputDotProduct(cStack<int> &PrevSpID_stack, int CurrSpID_i,
                         int NextSpID_i);
  void HeartSegmentation(int MaxRadius);
  void ComputingALoopID(int CurrSpID, int NextSpID);
  int ComputingMainBranches();
  void Extending_End_Spheres_For_Heart();
  int Extending_End_Spheres_AwayFromTheHeart();
  int Extending_End_Spheres_Toward_Artery();
  void FindingHeartConnections();
  void FindingHeartConnections2();
  void Adding_A_Neighbor_To_Isolated_Spheres_for_Heart();
  void Adding_A_Neighbor_To_Isolated_Spheres_For_SmallBranches();
  void Finding_Artery();
  void AddingSeedPts_for_Blood_Vessels();
  void ComputingMaxSpheres_for_Blood_Vessels();
  void FindingLineStructures_AddingSeedPts();
  void ComputingMaxSpheres_for_Small_Branches();

  void MarkingLinesBetweenSpheres(int *Volume);
  void MarkingSpID(int SpID, int *Volume);
  void MarkingSpID_ForSmallBranches(int SpID, int *Volume);
  void UnMarkingSpID(int SpID, int *Volume);

  int Finding_ExistSphere_Near_SameDir(int CurrSpID, int *Loc3, float *Dir3,
                                       int Radius, int &ExstSpID_ret);
  int FindingNextNewSphere_AwayFromHeart(int PrevSpID_i, int CurrSpID_i,
                                         int &NextSpID_ret,
                                         cStack<int> &Deselection_stack);
  void ComputingColorTable1();
  int FindingANearestSphereFromNext(int CurrSpID, int *NextLoc3,
                                    float *CurrDir3, int Radius,
                                    int &ExstSpID_ret);
  int IsCenterLineInside(int CurrSpID, int NextSpID);
  int IsCenterLineInside(int *Center13, int *Center23);
  int IsCenterLineInside(int *CurrCenter3, int *NextCenter3, int SpR1_i,
                         int SpR2_i);
  int IsCenterLineInsideFully(int *Center13, int *Center23);
  int IsCenterLineHitTheHeart(int *Center13, int *Center23);
  int IsCenterLineInsideBVIntensity(int *CurrCenter3, int *NextCenter3);
  int IsThickCenterLineOutsideBVIntensity(int *Center1_i, int *Center2_i,
                                          int SpR1_i, int SpR2_i);
  int FindingNextNewSphere_MainBranches(int PrevSpID_i, int CurrSpID_i,
                                        int &NextSpID_ret,
                                        cStack<int> &Deselection_stack);

  int IsCenterLineInterceptOtherSpheres(int *Center13, int *Center23,
                                        cStack<int> &SpIDs_stack);
  int IsOutsideLungs(int *SphereCenter3, int Radius);
  void RefinementBranches();
  int FindingNextNewSphere_Toward_Artery(cStack<int> &PrevSpIDs_stack,
                                         int CurrSpID_i, int CurrLRType,
                                         cStack<int> &Deselection_stack,
                                         int &NextSpID_ret);

  void ComputeAccumulatedVector(int PrevSpID, int CurrSpID, int NumSpheres,
                                Vector3f &AccVec_ret);
  int Is_Increasing_SphereR_Direction(int PrevSpID, int CurrSpID,
                                      int NumSpheres);
  int Is_Connected_NumSpheres_BiggerThan(int PrevSpID, int CurrSpID,
                                         int NumSpheres);
  int FindingNextNewSphereFromTheJunction(int JCTSpID_i, int &NextSpID_ret,
                                          cStack<int> &Neighbors_stack_ret,
                                          double &Dot_ret);
  int Self_ConnectedBranches(int CurrSpID, int GivenSpID);

  int FindingNextNewSphere_ForMissingBranches(int CurrSpID_i, int *MinDegLoc3,
                                              int CurrLRType,
                                              cStack<int> &Deselection_stack,
                                              int &NextSpID_ret);

  unsigned char *Refinement2ndDerivative();
  unsigned char *ThickBVVolume();
  unsigned char *ThickBVVolume_PEDetection();

  void Finding_MainBranchesOfHeart();
  int ComputingNextSphereLocations_MultiBranches2(int CurrSpID_i);
  double ComputingNextSphereLocationAndDotP(int PrevSpID_i, int CurrSpID_i,
                                            Vector3f &PrevToCurrDir_ret);
  double ComputingNextSphereLocationAndDotP(int CurrSpID_i,
                                            Vector3f &PrevToCurrDir_ret);
  int HitTheHeart(int CurrSpID_i);
  int ConnectToTheHeart(int PrevSpID, int CurrSpID);
  int ConnectToTheHeart(int CurrSpID, int *NextCenter3);
  int IsThickCenterLinePenetrateLungs(int PrevSpID, int CurrSpID);
  int IsThickCenterLinePenetrateLungs(int *Center1_i, int *Center2_i,
                                      int SpR1_i, int SpR2_i);

  void RemovingThreeSphereLoop(int CurrSpID, int NextSpID,
                               cStack<int> &Neighbors_stack);
  void AddToMap(double DotP, int CurrSpID_i, map<double, int> &Dot_map);
  void RecomputingBranches();
  //		int DoDisconnect(int JCT1PrevSpID_i, int JCT1SpID_i, int
  //JCT1NextSpID_i, 									int JCT2PrevSpID_i, int JCT2SpID_i, int JCT2NextSpID_i);
  int ConnectionEvaluation(int CurrSpID_i, int NextSpID_i);
  int IsThickCenterLineInside(int CurrSpID_i, int ExstSpID_i,
                              int GoesToHeartSpID_i);

  int IsRealJunction(int PrevJCT1SpID, int CurrJCT1SpID, int NextSpID_i,
                     int PrevJCT2SpID, int CurrJCT2SpID,
                     cStack<int> &Branch1_stack_ret,
                     cStack<int> &Branch2_stack_ret);
  void MakeABranch(int PrevSpID, int CurrSpID, int NeighborSpID_i,
                   float *DirVec3, cStack<int> &Branch_stack,
                   cStack<int> &NewSpID_stack_ret,
                   cStack<int> &NhbrSpID_stack_ret);
  void MakeTwoBranches(int PrevSpID1, int CurrSpID1, int NexSpID_i,
                       cStack<int> &Branch1_stack, int PrevSpID2,
                       int CurrSpID2, cStack<int> &Branch2_stack,
                       int CurrLRType_i, int &NextNew1SpID_ret,
                       int &NextNew2SpID_ret);
  int IsContinuous(int CurrSpID, int ExstSpID);
  int IsAContinuousConnection(int PrevSpID, int CurrSpID, int ExstSpID);
  int IsRealConnection(int PrevSpID, int CurrSpID, int ExstSpID,
                       int NumSteps);
  int JCTEvaluation(int SpID1_i, int SpID2_i, int SpID3_i);
  int JCTEvaluation(int CurrSpID_i, int NextSpID_i, float &CurrDeg_ret,
                    float &NextDeg_ret);
  int JCTEvaluation2(int SpID1_i, int SpID2_i);
  void EvaluatingBranches();
  void PE_Detection(unsigned char *BVVolume);
  void Naming_EachBranch();
  void RecomputingMaxR_InBloodVessels(unsigned char *BVVolume);
  void RecomputingMaxR_InBV_Muscles(unsigned char *BVVolume);
  int IsOutsideLungSpheres(int SpID);
  int ComputingNodulePossiblity(int SpID);
  int ComputingNumBVVoxels(int SpID);
  int ComputingNumMuscleVoxelsWithinR(int SpID, int RMinus);
  int ComputingNumMuscleVoxels(int CurrSpID, int NextSpID);
  int ComputingNumMuscleVoxels2(int CurrSpID, int NextSpID);
  void ComputingSensitivity(int NumPESpheres_i, int *PESpIDs,
                            cStack<int> &DeadEndsRefined_stack,
                            float Ratio_MuscleTh_f,
                            cStack<int> &PE_Spheres_stack,
                            unsigned char *PE_Volume_uc);
  void ComputingGradientFromCenter(int SpID);

public:
  void SeedPtsBased_Skeletonization(double Threshold_d);
  void ComputingTheBiggestSphereAt_Data(int Xi, int Yi, int Zi,
                                        double Threshold_d, int &SphereR_ret);
  int ComputingTheBiggestSphereAt_DataRange(cStack<int> &VoxelLocs,
                                            _DataType Lower_Th,
                                            _DataType Upper_Th,
                                            int *Center3_ret,
                                            int &SphereR_ret);
  void MarkingPEVolume(int CurrSpID_i, int NeighborSpID_i,
                       unsigned char *PE_Volume_uc);

public:
  void SwapValues(int &Value1, int &Value2);
  void SwapValues(float &Value1, float &Value2);
  void Display_ASphere(int SpID);
  void Display_ASphere(int SpID, int PrintExtraInfo_i);
  void PrintFileOutput(char *FileName, int FlipX);
  void SaveSphereVolume(int FileNum);
  void SaveSphereVolume_AwayFromHeart(int FileNum);
  void DrawLineStack(float *Dir3, int *Center3, int Length, int LineStackNum,
                     int KeepFirstElem_i, int KeepLastElem_i);
  void DrawLineStack(int *CurrCenter3, int *NextCenter3, int LineStackNum,
                     int KeepFirstElem_i, int KeepLastElem_i);
  void SaveSphereVolume_NewCenterBV(int FileNum);

public:
  void SaveGradientMagnitudes(char *OutFileName, int NumLeaping);
  void SaveSecondDerivative(char *OutFileName, int NumLeaping);
  void SaveGeometry_RAW(char *filename);
  void Destroy();

public:
  void Output_for_Samrat();
};

extern unsigned char *RawivHeader_guc;
// extern char				*TargetName_gc;
extern char TargetName_gc[512];
extern double GaussianKernel_3x3x3_d[3][3][3];
extern double GaussianKernel_5x5x5_d[5][5][5];
extern double GaussianKernel_7x7x7_d[7][7][7];
extern double GaussianKernel_9x9x9_d[9][9][9];
extern double GaussianKernel_11x11x11_d[11][11][11];

template <class _DataType> void QuickSort(_DataType *data, int p, int r);

template <class _DataType> void Swap(_DataType &x, _DataType &y);

template <class _DataType> int Partition(_DataType *data, int low, int high);

template <class T>
T *Bilateral_Filter(T *data, int WindowSize, double &Min, double &Max,
                    double S1, double S2);

template <class T>
void SaveVolume(T *data, float Minf, float Maxf, char *Name);

void SaveImage(int Width, int Height, unsigned char *Image, char *PPMFile);

template <class T>
void SaveVolumeRawivFormat(T *data, float Minf, float Maxf, char *Name,
                           int ResX, int ResY, int ResZ, float SpanX,
                           float SpanY, float SpanZ);

template <class T>
void SaveVolumeRawivFormat(T *data, float Minf, float Maxf, char *Name,
                           int ResX, int ResY, int ResZ, float SpanX,
                           float SpanY, float SpanZ, int FlipX);

void SaveImage(int Width, int Height, unsigned char *Image, char *PPMFile);

extern int SubConf2DTable[27][2];

extern float *DataOrg_gf;

// STree.cpp
void LoadTreeStructure(char *SkeletonFileName_c);
void AVSeparation_Evaluate(unsigned char *BV, int W, int H, int D);
void Evaluation(int CurrLoc);

#endif

/*

        for (k=0; k<Depth_mi; k++) {
                for (j=0; j<Height_mi; j++) {
                        for (i=0; i<Width_mi; i++) {

                        }
                }
        }

        // Computing X, Y and Z locations
        Zi = Loc/WtimesH_mi;
        Yi = (Loc - Zi*WtimesH_mi)/Width_mi;
        Xi = Loc % Width_mi;

        for (n=Zi-1; n<=Zi+1; n++) {
                for (m=Yi-1; m<=Yi+1; m++) {
                        for (l=Xi-1; l<=Xi+1; l++) {


                        }
                }
        }

*/

/*
        if (TempSpID_i==2200 && CurrSpID_i==1368) {
                DrawLineStack(&PrevToCurrDir_vec[0],
   &SeedPtsInfo_ms[CurrSpID_i].MovedCenterXYZ_i[0],
                                                SeedPtsInfo_ms[CurrSpID_i].MaxSize_i*5,
   1);
        }


        if (CurrSpID_i==2444 && ExstSpID_i==68) {
                DrawLineStack(&Center_i[0], &Center2_i[0], i/4+1);
        }

*/

//	printf ("(%5d-->%5d-->%5d) = %7.4f ", PrevSpID_i, CurrSpID_i,
//NextSpID_i, Dot_f);
