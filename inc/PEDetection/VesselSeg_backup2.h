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

#ifndef FILE_VESSEL_SEG_H
#define FILE_VESSEL_SEG_H


#include <map.h>
#include <PEDetection/Timer.h>
#include <PEDetection/TFGeneration.h>
#include <PEDetection/Stack.h>
#include <PEDetection/Geometric.h>
#include <PEDetection/MarchingCubes.h>


// Marking LocalMinMaxVoxels_muc[]

// Debugging Options
//#define	DEBUG_TRACKING
//#define DEBUG_BLOCKING_PLANE
#define DEBUG_EXPANDINGBOX




#define	UNKNOWN				0
#define VESSEL_WALL_TRUE	1
#define VESSEL_INSIDE		2

// The minimum value of gradient magnitude
// to compute the second derivative and for wave propagation
#define	MIN_GM				5

struct sLineS {
	int		NumHits;
	float	AveDist_f;
	float	AveGM_f;
	float	Direction_f[3];
	float	CenterPt_f[3];

};


struct VoxelInfo {
					// Triangles on X, Y, and Z Positive direction
	unsigned char	TonXPos, TonYPos, TonZPos;
	unsigned char	IsVesselWall;	// 0=Unknown, 1=false, 2=true;
	cStack<int>		*Triangles_s;
};


struct sWaveTimeSize {
	int			WaveTime_i;
	int 		Size;
	float		CenterPt_f[3];
	float		Direction_f[3];
	cStack<int>	*VoxelLocs_s;
};

struct sTracking {
	float		CenterPt_f[3];
	float		Direction_f[3];
};

#define		NUM_WAVE_SIZE			10
#define		NUM_CENTER_PTS			10

struct sFrontWave {
	int			WaveTime_i;
	int 		Size;
	int 		WaveSize[NUM_WAVE_SIZE*2];	// 0=Wave Size, 1=Clear Window Size
	float		CenterPt_f[NUM_CENTER_PTS*3];
	cStack<int>	*VoxelLocs_s;
};


// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
struct sFrontPlanes {
	int			WaveTime_i;
	int 		TotalSize;
	float		CenterPt_f[6*3];
	cStack<int>	*VoxelLocs_s[6];
};


template <class _DataType> 
class cVesselSeg {

	protected:
		int 		NumMaterial_mi;	// the number of clusters
		int			*Histogram_mi;
		float		HistogramFactorI_mf;
		float		HistogramFactorG_mf;

		int			Width_mi, Height_mi, Depth_mi;
		int			WtimesH_mi, WHD_mi;
		
		_DataType	*Data_mT, *ClassifiedData_mT;
		float		MinData_mf, MaxData_mf;
		float		*MaterialProb_mf;
		float		*GradientMag_mf, MinGrad_mf, MaxGrad_mf;
		float		*GradientVec_mf;
		float		*SecondDerivative_mf, MinSecond_mf, MaxSecond_mf;
		char		*OutFileName_mc;
		
		double		GaussianKernel3_md[3][3][3];
		double		GaussianKernel5_md[5][5][5];
		double		GaussianKernel7_md[7][7][7];
		double		GaussianKernel9_md[9][9][9];
		double		GaussianKernel11_md[11][11][11];
		
		float		SpanX_mf, SpanY_mf, SpanZ_mf;
		
		int			*Distance_mi;
		unsigned char	*SeedPtVolume_muc;
		unsigned char	*FlaggedVoxelVolume_muc;
		unsigned char	*LungSegmented_muc, *Heart_muc;
		float			*LungSecondD_mf;
		
		int				*Wave_mi;
		
		// Geometric Data
		int				NumVertices_mi, NumTriangles_mi;
		float			*Vertices_mf;
		int				*Triangles_mi;
		int				*TNeighbors_mi;	// Triangle Neightbors

		// Seed Point Locations(X, Y, Z) and the number of seed points
		int							*SeedPts_mi, NumSeedPts_m;
		map<int, unsigned char>		SeedPts_mm;

		// Marching time for the fast marching algorithm
		float						*MarchingTime_mf;
		
		// Skeleton volume
		unsigned char				*Skeletons_muc;
		map<int, unsigned char>		Skeletons_mm;

		// Line Structure Voxels
		map<int, struct sLineS *>	LineSVoxels_mm;
		
		struct VoxelInfo 			*VInfo_m;
		
		
		int						NumSegmentedVoxels_mi, NumLineStructureVoxels_mi;
		int						NumCenterVoxels_mi;
		
		struct sFrontWave		*CurrFrontWave_m;
		int						MaxNumBranches_mi, CurrNumBranches_mi;
		
		struct sFrontWave		*PrevFrontWave_m;
		int						MaxNumPrevBranches_mi, CurrNumPrevBranches_mi;
		
		int						NumRanges_mi, OutlierRanges_mi[8][2];
		unsigned char			*ZeroCrossingVoxels_muc;
		float					*FlowVectors_mf;
		
		
	public:
		cVesselSeg();
		~cVesselSeg();		

		void setData(_DataType *Data, float Minf, float Maxf);
		void setGradient(float *Grad, float Minf, float Maxf);
		void setGradientVectors(float *GradVec);
		void setWHD(int W, int H, int D);
		void setProbabilityHistogram(float *Prob, int NumMaterial, int *Histo, float HistoF);
		void setSecondDerivative(float *SecondD, float Min, float Max);
		void setXYZSpans(float SpanX, float SpanY, float SpanZ);
		
	public:
		void CopyFlaggedVoxelVolume(unsigned char *FlaggedVoxelVolume);
		void CopyDistanceVolume(int *Distance);

	public:
		void VesselExtraction(char *OutFileName, _DataType LungMatMin, _DataType LungMatMax,
								_DataType MSoftMatMin, _DataType MSoftMatMax,
								_DataType VesselMatMin, _DataType VesselMatMax);
	protected:
		void BiggestDistanceVoxels(cStack<int> &BDistVoxels_s);
		cStack<int>	*BlockingOnThePlane(int Xi, int Yi, int Zi);
		void BlockingOnThePlane2(int Xi, int Yi, int Zi);
		void Build_TriangleToVoxelMap();
		void RemovingNonVesselWalls();
		void RemovingNonVesselWalls2(cStack<int> &BDistVoxels_s);
		
		void Lung_Extending();
		int ExtractLungOnly(int LocX, int LocY, int LocZ);
		void LungBinarySegment(_DataType LungMatMin, _DataType LungMatMax);
		void MedianFilterForLungSegmented(int NumRepeat);
		void Vessel_BinarySegment(_DataType VesselMatMin, _DataType VesselMatMax);
		void Removing_Outside_Lung();
		
		void Complementary_Union_Lung_BloodVessels();
		int ExtractHeartOnly(unsigned char *Data, int LocX, int LocY, int LocZ);
		
		void SaveSecondD_Lung();
		void SaveSecondD_Lung2();
		void ComputeIntersection(Vector3f Pt1, Vector3f Pt2, 
						Vector3f Pt3, Vector3f Pt4, c3DPlane Plane, Vector3f &Pt_Ret);
		void ArbitraryRotate(Vector3f &Pt1, Vector3f Axis, float Theta, Vector3f &Pt_Ret);
		void ComputeSaddlePoint(float *T8, float &DistCT, float &GT);
		void RemoveSecondDFragments(int MinNumNegVoxels, float Sign_PN);
		void Cleaning_Phantom_Edges();
		void Marking_Inside_BloodVessels(_DataType MSoftMatMin, _DataType MSoftMatMax);
		void Remove_Phantom_Triangles(_DataType LungMatMin, _DataType LungMatMax,
									_DataType VesselMatMin, _DataType VesselMatMax);
		void Rearrange_Triangle_Indexes();
		void Remove_Triangle_Fragments();
		void Finding_Triangle_Neighbors();
		
	protected:
		void ComputeDistance();
		void SmoothingClassifiedData(int WindowSize);
		void GaussianSmoothing3D(_DataType *data, int WindowSize);
		void ComputeGaussianKernel();
		
	protected:
		int IsLineStructure(int Xi, int Yi, int Zi, float *DirVec, int WindowSize);
		int IsLineStructureDist(int Xi, int Yi, int Zi, float *DirVec, int WindowSize);
		void EigenDecomposition(float *Mat, float *Eigenvalues, float *Eigenvectors);
		void ComputeSkeletons(_DataType MatMin, _DataType MatMax);
		int FindNearestSeedPoint(int X, int Y, int Z);
		int SkeletonTracking_Marking(int EndLoc);
		int FastMarching_NearestLoc(map<int, unsigned char> SeedPts_m);
		void ConnectingOneVoxelLengthSkeletons(map<int, unsigned char> &SeedPts_m);
		void ConnectingOneVoxelLengthSkeletons2(map<int, unsigned char> &SeedPts_m);
		
	protected:
		void ComputeDistanceVolume(double GradThreshold);
		void ComputeGVF();
		void FlagNonUniformGradient();
		void FlagNonUniformGradient2();
		void ConnectingFlaggedVoxels();
		void ConnectingFlaggedVoxels2();
		void ConnectedComponents(char *OutFileName, int &MaxCCLoc_Ret);
		int MarkingCC(int CCLoc, unsigned char MarkingNum, unsigned char *CCVolume_uc);
		void FindingRootAndEndLoc(int MaxCCLoc, int &RootLoc_Ret, int &EndLoc_Ret);
		void ComputeSkeletons(int RootLoc, int EndLoc, int Threshold_Dist);
		int ComputeDistanceFromCurrSkeletons(map<int, unsigned char> &InitialBoundary_m);
		
		
	private:
		double ComputeAveRadius(double *StartPt_d, double *Direction_d, double *NewStartPt_Ret);
		void ComputePerpendicular8Rays(double *StartPt, double *Direction, double *Rays);
		void ComputeRadius(double *StartPt, double *Rays8, double *HitLocs8, double *Radius8);
		int getANearestZeroCrossing(double *CurrLoc, double *DirVec, double *ZeroCrossingLoc_Ret, 
							double& FirstDAtTheLoc_Ret, double& DataPosFromZeroCrossingLoc_Ret);

		float Normalize(float *Vec);
		double Normalize(double	*Vec);
		double DataInterpolation(double* LocXYZ);
		double DataInterpolation(double LocX, double LocY, double LocZ);
		double GradientInterpolation(double* LocXYZ);
		double GradientInterpolation(double LocX, double LocY, double LocZ);
		double SecondDInterpolation(double* LocXYZ);
		double SecondDInterpolation(double LocX, double LocY, double LocZ);
		void MakeCube8Indexes(int Xi, int Yi, int Zi, int *Locs8);
		void SaveVoxels_SecondD(int LocX, int LocY, int LocZ,	float *CenterPt3,
									int VoxelResolution, int TotalResolution);
		void SaveVoxels_Volume(unsigned char *Data, int LocX, int LocY, int LocZ,
							char *Postfix, int VoxelResolution, int TotalResolution);
		int GradVecInterpolation(double LocX, double LocY, double LocZ, double* GradVec_Ret);
		double TrilinearInterpolation(unsigned char *Data, double LocX, double LocY, double LocZ);
		
	private:
		int Index(int X, int Y, int Z, int ith, int NumElements);
		int Index(int X, int Y, int Z);
		void SaveSeedPtImages_Line(int *SeedPts_i, int NumSeedPts);
		void SaveInitVolume(_DataType Min, _DataType Max);
		void SaveDistanceVolume();
		void DisplayFlag(int Flag);
		void Display_Distance(int ZPlane);
		void Display_Distance2(int ZPlane);
		void DisplayKernel(int WindowSize);
		
		// Tracking Vessels
	public:
		void TrackingVessels(cStack<int> &BiggestDist_s);
		void Tracking_Vessels(cStack<int> &BiggestDist_s);
		void MovingBalls(cStack<int> &BiggestDist_s);
		void InitFrontWave();
		void AddFrontWave(int CurrTime, cStack<int> *WaveLocs_s, float *Center_Mx3, 
										int *WaveSize_Mx2, int ClearWindowSize);
		
		cStack<int> *FindBiggestSizeFrontWave(float *Center_Mx3_ret, int *WaveSize_Mx2_ret, int WindowSize);
		void DisplayBiggestFrontWave(int CurrTime_i, cStack<int> *FrontWave_s);
		int IsOutlier(int DataLoc);
		void ComputeZeroCrossingVoxels();
		int IsClearWindow(int DataLoc, int WindowSize, int CurrTime);
		void SaveTheSlice(unsigned char *SliceImage_uc, int NumRepeat, int CurrTime_i, int CenterZi);
		void Compute_FlowVectors(float *CurrCenter6x3, cStack<int> *CurrWave_s);
		void ComputeFlowVectors(float *CurrCenter6x3, float *FlowVector3_ret);
		void ChangePrevToCurrFrontWave();
		void AddPrevFrontWave(int CurrTime, cStack<int> *WaveLocs_s, float *Center_Mx3,
										int *WaveSize_Mx2, int ClearWindowSize);
		void InitPrevFrontWave();
		int IsIncreasingVessels(int *WaveSize_Mx2);
		void MakingBiggestBox(int *CenterLoc3, int CurrTime, int *XYZLevels6);
		void MakingBiggestBox2(int *CenterLoc3, int CurrTime, int *XYZLevels6);
		void ExpandingBox(int *StartCenterPt3, int CurrTime_i, int *XYZLevels6);
		void ComputeCC(cStack<int> *FrontWave_s, int CurrTime);
				
	public:
		void SaveSecondDerivative(char *OutFileName, int NumLeaping);
		void SaveGeometry_RAW(char *filename);
		void Destroy();


};


extern unsigned char	*RawivHeader_guc;
extern char				*TargetName_gc;
extern double GaussianKernel_3x3x3_d[3][3][3];
extern double GaussianKernel_5x5x5_d[5][5][5];
extern double GaussianKernel_7x7x7_d[7][7][7];
extern double GaussianKernel_9x9x9_d[9][9][9];
extern double GaussianKernel_11x11x11_d[11][11][11];


template<class _DataType>
void QuickSort(_DataType* data, int p, int r);

template<class _DataType>
void Swap(_DataType& x, _DataType& y);

template<class _DataType>
int  Partition(_DataType* data, int low, int high );

template<class T>
T *Bilateral_Filter(T *data, int WindowSize, double& Min, double& Max, double S1, double S2);

template<class T>
void SaveVolume(T *data, float Minf, float Maxf, char *Name);

void SaveImage(int Width, int Height, unsigned char *Image, char *PPMFile);

template<class T>
void SaveVolumeRawivFormat(T *data, float Minf, float Maxf, char *Name, 
			int ResX, int ResY, int ResZ, float SpanX, float SpanY, float SpanZ);

void SaveImage(int Width, int Height, unsigned char *Image, char *PPMFile);

extern int	SubConf2DTable[27][2];


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

