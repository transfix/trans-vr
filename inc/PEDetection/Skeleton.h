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

#ifndef FILE_SKELETON_H
#define FILE_SKELETON_H


#include <map.h>
#include <PEDetection/TFGeneration.h>
#include <PEDetection/Stack.h>
#include <PEDetection/Geometric.h>

// Marking LocalMinMaxVoxels_muc[]
#define FLAG_CONNECTED		250
#define FLAG_NONUNIFORM		200

#define FLAG_LOCAL_MAX		100
#define FLAG_LOCAL_MIN		 90

#define FLAG_SEGMENTED		 50
#define FLAG_EMPTY			  0

//#define DEBUG_FLAG_NONUNIFORM 
//#define DEBUG_CONNECTING
//#define	DEBUG_DIST_TF

#define DEBUG_CONNECTING2
#define		DEBUG_COMP_AVE_R



template <class _DataType> 
class cSkeleton {
	protected:
		int 		NumMaterial_mi;	// the number of clusters
		int			*Histogram_mi;
		float		HistogramFactorI_mf;
		float		HistogramFactorG_mf;

		int			Width_mi, Height_mi, Depth_mi;
		int			WtimesH_mi, WHD_mi;
		
		_DataType	*Data_mT;
		float		MinData_mf, MaxData_mf;
		float		*MaterialProb_mf;
		float		*GradientMag_mf, MinGrad_mf, MaxGrad_mf;
		float		*GradientVec_mf;
		float		*SecondDerivative_mf, MinSecond_mf, MaxSecond_mf;


		// Binary segmented volume by clipping of min & max values
		unsigned char	*InitVolume_muc;
		int				*Distance_mi;
		
		float			*GVFDistance_mf;
		unsigned char	*VoxelFlags_muc;
		
		int			NumSegmentedVoxels_mi, NumFlaggedVoxels_mi;
		
		unsigned char	*CCVolume_muc;
		
		int				*DistanceFromSkeletons_mi;
		unsigned char	*Skeletons_muc;

		
		
	public:
		cSkeleton();
		~cSkeleton();		

		void setData(_DataType *Data, float Minf, float Maxf);
		void setGradient(float *Grad, float Minf, float Maxf);
		void setGradientVectors(float *GradVec);
		void setWHD(int W, int H, int D);
		void setProbabilityHistogram(float *Prob, int NumMaterial, int *Histo, float HistoF);
		void setSecondDerivative(float *SecondD, float Min, float Max);
		
	public:
		unsigned char *getFlaggedVoxelVolume();
		int *getDistanceVolume();
		
	public:
		void Skeletonize(char *OutFileName, _DataType MatMin, _DataType MatMax);

	protected:
		void BinarySegment(_DataType MatMin, _DataType MatMax);
		void BinarySegment2(_DataType MatMin, _DataType MatMax);
		void BinarySegment3(_DataType MatMin, _DataType MatMax);
		void ComputeDistance();
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
		int SkeletonTracking_Marking(int EndLoc);
		
		
	private:
		double ComputeAveRadius(double *StartPt_d, double *Direction_d, double *NewStartPt_Ret);
		void ComputePerpendicular8Rays(double *StartPt, double *Direction, double *Rays);
		void ComputeRadius(double *StartPt, double *Rays8, double *HitLocs8, double *Radius8);
		int getANearestZeroCrossing(double *CurrLoc, double *DirVec, double *ZeroCrossingLoc_Ret, 
							double& FirstDAtTheLoc_Ret, double& DataPosFromZeroCrossingLoc_Ret);

		double Normalize(double	*Vec);
		double GradientInterpolation(double* LocXYZ);
		double GradientInterpolation(double LocX, double LocY, double LocZ);
		double SecondDInterpolation(double* LocXYZ);
		double SecondDInterpolation(double LocX, double LocY, double LocZ);
		
		
	private:
		int Index(int X, int Y, int Z, int ith, int NumElements);
		int Index(int X, int Y, int Z);
		void SaveInitVolume(_DataType Min, _DataType Max);
		void SaveDistanceVolume();
		void DisplayFlag(int Flag);
		void Display_Distance(int ZPlane);
		void Display_Distance2(int ZPlane);
		
	public:
		void Destroy();


};


extern unsigned char	*RawivHeader_guc;



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


#endif


