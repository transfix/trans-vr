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

#ifndef FILE_TFGENERATION_H
#define FILE_TFGENERATION_H

#include <map.h>

#include <PEDetection/Stack.h>


#define MAX_NUM_MATERIALS 30
#define MAXNUM_HISTOGRAM_ELEMENTS	5000

// Marking MatNumVolume_muc[]
#define MAT_ZERO_CROSSING		0x80 // 1000 0000
#define MAT_BOUNDARY_HAS_ALPHA	0x40 // 0100 0000
#define MAT_BOUNDARY			0x20 // 0010 0000
#define MAT_INSIDE_BOUNDARY		0x10 // 0001 0000 Read Material Body

#define MAT_OTHER_MATERIALS		0x08 // 0000 1000
#define MAT_CHECKED_VOXEL		0x01 // 0000 0001

#define MAX_NUM_CC_STACKS		999999




template <class _DataType> 
class cTFGeneration {
	protected:
		int 		NumMaterial_mi;	// the number of Clusters
		int			*Histogram_mi;
		float		HistogramFactorI_mf;
		float		HistogramFactorG_mf;

		int			Width_mi, Height_mi, Depth_mi;
		int			WtimesH_mi, WHD_mi;
		
		_DataType	*Data_mT;
		float		MinData_mf, MaxData_mf;

		float		*GradientVec_mf;
		float		*GradientMag_mf;
		float		MinGrad_mf, MaxGrad_mf;
		
		float		*SecondDerivative_mf;
		float		MinSecMag_mf, MaxSecMag_mf;
		
		float		*MaterialProb_mf;
		float		*H_vg_mf, *P_vg_mf;
		unsigned char *VoxelStatus_muc;
		float		*AlphaVolume_mf;
		
		// Location & SecondDerivatives
		map<int, float> MaterialBoundary_mm; // map<int location, float sigma_square>

		int			*TF_mi[MAX_NUM_MATERIALS];
		
		int				StartLoc_gi[3];
		unsigned char	*ZeroCrossingVoxels_muc; // Each Voxel has 10*10*10 resolution
		double			ZCVolumeGridSize_md;
		unsigned char	*VoxelVolume_muc; // which represent each voxel with 10*10*10
		
		unsigned char	*BoundaryVolume_muc;
		double			BV_GridSize_md;
		
		float			*ZeroCrossingCells_mf;
		int				*ConnectedComponentV_mi;

		cStack<int>		ConnectedVoxel_Stack[MAX_NUM_CC_STACKS];

		// For the Connected Surface Volume Computation
		int				*CCSurfaceVolume_mi;
		int				*CCIndexTable_mi, MaxSize_CCIndexTable_mi;
		int				*CCSurfaceIndex_mi, MaxSize_CCSurfaceIndexTable_mi;

		// Kindlmann's Sigma Computation
		double			*H_vg_md;
		double			*G_v_md;
		int				*HistogramVolume_mi;

		// Saving the distance from a voxel to a hit location
		float			*DistanceToZeroCrossingLoc_mf;
		

	public:
		cTFGeneration();
		~cTFGeneration();		

		void setData(_DataType *Data, float Minf, float Maxf);
		void setGradient(float *Grad, float Minf, float Maxf);
		void setGradientVectors(float *GradVec);
		void setWHD(int W, int H, int D);
		void setProbabilityHistogram(float *Prob, int NumMaterial, int *Histo, float HistoF);
		void InitTF(int NumClusters);
		
		_DataType getDataAt(int loc) { return Data_mT[loc]; };
		float getMinSecondD() { return MinSecMag_mf; };
		float getMaxSecondD() { return MaxSecMag_mf; };

	public:
		void ComputeSecondDerivative(char *TargetName); // Directional 2nd Derivative
		void DoNotComputeSecondDerivative(char *TargetName);
		float* getSecondDerivative() { return SecondDerivative_mf; };
	public:
		double GradientInterpolation(double* LocXYZ);
		double GradientInterpolation(double LocX, double LocY, double LocZ);
		double GradientInterpolation2(double LocX, double LocY, double LocZ);
		double SecondDInterpolation(double* LocXYZ);
		double SecondDInterpolation(double LocX, double LocY, double LocZ);
		double DataInterpolation(double* LocXYZ);
		double DataInterpolation(double LocX, double LocY, double LocZ);
		int GradVecInterpolation(double* LocXYZ, double* GradVec_Ret);
		int GradVecInterpolation(double LocX, double LocY, double LocZ, double* GradVec_Ret);
		int GradVecInterpolation2(double LocX, double LocY, double LocZ, double* GradVec_Ret);
		float* ComputeSecondD();
		double getDataValue(double *Loc_d, double *NormalizedGradVec, double Position_d);
		
		
	public:
		void ComputeIGGraph(char *OutFileName, int MaterialNum, int NumBins);
		void ComputeIGGraphAllMaterials(char *OutFileName, int NumElements);
		void ComputeIGGraphBoundaries(char *OutFileName, int NumElements);
		void ComputeIGGraphGeneral(char *OutFileName, int NumElements);
		void ComputeIGGraphGeneral(char *OutFileName, int NumElements, float MinGM, float MaxGM);
		void ComputeHistogramVolume(int NumElements, double& Sigma_Ret);
		
	public:
		void ComputeIHGraphAllMaterials(char *OutFileName, int NumElements);
		

	public:
		void ComputeTF(char *OutFileName, _DataType MatMin, _DataType MatMax, int *TransFunc);
		void ComputeTF2(char *OutFileName,int MaterialNum, int *TransFunc);
	protected:
		int IsMaterialBoundary(int DataLoc, int MaterialNum);
		int IsMaterialBoundary(int DataLoc, int MaterialNum, int Neighbor);
		int FindNeighbors8Or26(int CenterLoc, int *Neighbors);
		int FindNeighbors8Or26(int *CenterLoc, int *Neighbors);
		void FindNeighbors4Or6(int *CenterLoc, int *Neighbors);
		void InitializeVoxelStatus();
		void InitializeAlphaVolume();
		void SaveBoundaryVolume(char *FileName, _DataType MatMin, _DataType MatMax);
		void SaveAlphaVolume(char *OutFileName, _DataType MatMin, _DataType MatMax);
	public:
		void MarkingZeroCrossingLoc(_DataType MatMin, _DataType MatMax);
		void MarkingGradDirectedVoxels(_DataType MatMin, _DataType MatMax);
		void ComputeSigmaAtBoundary(_DataType MatMin, _DataType MatMax);
		void ComputeConnectedSurfaceVolume(double GradThreshold, char *TargetFileName);
		void ComputeConnectedSurfaceVolume2(double GradThreshold, char *TargetFileName);
		void MarkingLocalMaxVoxels(double GradThreshold);
		void MarkingLocalMaxVoxels2(double GradThreshold);
		int IsConnectedVoxel_To_ZeroCrossing(double *ZeroCrossingLoc3_d, int *TempLoc3_i);
		int FindMinIndex_18Adjacency(int *CCVolume_i, int *TempLoc3_i);
		void Initialize_CCIndexTable();
		void IncreaseSize_CCIndexTable(int MaxCCIndex_i);
		unsigned char *Rearrange_CCVolume(int *CCVolume_i);
		unsigned char *ConnectedPositiveVoxels();
		void Initialize_CCSurfaceIndexTable();
		void IncreaseSize_CCSurfaceIndexTable(int CurrMaxCCIndex_i);
		void put_CCSurfaceIndex(int Index, int NumVoxels);
		void ComputeAndSaveDistanceFloat(double GradThreshold, 
									float &MinData_Ret, float &MaxData_Ret, 
									float &MinGM_Ret, 	float &MaxGM_Ret, 
									float &MinSD_Ret, float &MaxSD_Ret, char *TargetFileName);
		void MinMaxComputationAtHitLocations(double GradThreshold, 
											float &MinData_Ret, float &MaxData_Ret, 
											float &MinGM_Ret, 	float &MaxGM_Ret, 
											float &MinSD_Ret, float &MaxSD_Ret,
											float &MinDist_Ret, float &MaxDist_Ret);
		
		
		
		
		int IsMaterialBoundaryUsingMinMax(int DataLoc, _DataType MatMin, _DataType MatMax);
		void ComputeAlpha(_DataType MatMin, _DataType MatMax);
		void ComputeMinMax(_DataType MatMin, _DataType MatMax);


		// Finding Zero-Crossing Locations from each voxel
		int FindZeroCrossingLocation(double *CurrLoc, double *GradVec, double *ZeroCrossingLoc_Ret, 
								double& FirstDAtTheLoc_Ret, double& DataPosFromZeroCrossingLoc_Ret);
		int FindZeroCrossingLocation(double *CurrLoc, double *GradVec, double *ZeroCrossingLoc_Ret, 
				double& FirstDAtTheLoc_Ret, double& DataPosFromZeroCrossingLoc_Ret, double Interval);
		int FindZeroCrossingLocation2(double *CurrLoc, double *GradVec, double *ZeroCrossingLoc_Ret, 
				double& FirstDAtTheLoc_Ret, double& DataPosFromZeroCrossingLoc_Ret, double Interval);
		int SearchAgain_WithVectorAdjustment(double *CurrLoc, double *GradVec, double *ZeroLoc_Ret);



		int FindZeroConnectedNeighbors(int CenterLoc, int *Neighbors);
		int FindZeroConnectedNeighbors2(int CenterLoc, int *Neighbors);
		int ComputeSigma(double *CurrLoc, double *NormalizedGradVec, double& SigmaPosDir_Ret,
							double& SigmaNegDir_Ret, double& SecondDPosDir_Ret, double& SecondDNegDir_Ret);
		int ComputeSigma(double *CurrLoc, double *NormalizedGradVec, 
							double& Sigma_Ret, double& SecondD_Ret, double Increase_d);
		void InitZeroCrossingVolume(int X, int Y, int Z);
		void SaveZeroCrossingVolume(char *Prefix);
		
	public:
		void ComputeH_vg(char *OutFileName, int NumElements);
		
	protected:
		void SaveIGGraph(char *OutFileName, int *IGHisto, int NumBins, int MinFreq, int MaxFreq);				
		void Save_H_vg(char *FileName, float *Hvg, int NumBins, float MinSD, float MaxSD);
		void Save_P_vg_Color(char *FileName, float *Pvg, int NumBins, float Minf, float Maxf);
		

	public:
		void AdjustingBoundaries(char *OutFileName);
	protected:
		void MakeMaterialNumVolume(int MatNum);
		int IsMaterialBoundaryInMaterialVolume(int DataLoc, int MaterialNum);
		
	public:
		void FindZeroSecondD(char *OutFileName);
		
		// map related functions
		void AddBoundary(int BoundaryLoc, float Sigma);

		int getNumBoundary();

	public:
		void InitBoundaryVolume();
		int FindBoundaryConnectedNeighbors_SaveACube(int CenterLoc, int *Neighbors);
		int FindBoundaryConnectedNeighbors_EdgeIntersections(int CenterLoc, int *Neighbors, 
		float *Intersections_Ret, int *Edges_Ret, int& NumIntersections_Ret, int& ConfigurationNum_Ret);
		int FindBoundaryConnectedNeighbors(int CenterLoc, int *Neighbors_Ret);


	public:
		void ComputeDistanceVolume(double GradThreshold, char *TargetFileName);
		float* ComputeDistanceVolume2(double GradThreshold, float& Minf_ret, float& Maxf_ret, 
									 char *TargetFileName, int NumBins);
		void Compute_Dist_GM(double GradThreshold, float& Minf_ret, float& Maxf_ret,
				unsigned char *DistVolume, unsigned char *GMVolume, char *TargetFileName);

		void ComputeDistanceHistoAt(double Intensity, double GradMag,
									float *DistanceVolume_f, float MinDist, float MaxDist);
		void HistoVolumeSigma_Evaluation(double GradThreshold, int NumElements,
										int DoSigmaEvaluation, char *TargetName_gc);
		void HistoVolumeSigma_Evaluation2(double GradThreshold, int NumElements, char *TargetName,
										int DoSigmaEvaluation, char *TargetName_gc);

	// Connected Component Extraction
	public:
		void Connected_Positive_SecondD_Voxels_RunSave(int DoNotReadFile);
	protected:
		void ConnectedPositiveSecondDVoxels();
		void ConnectedVoxels(int VertexI1, int VertexI2);
		int ConnectedC4(float *Cube8);
		
		
	protected:
		int	Index(int X, int Y, int Z, int ith, int NumElements);
		int	Index(int X, int Y, int Z);
		void NormalizeVector(double *GradVector3);

	// QuckSort Algorithm for 3 Elements
	protected:
		void QuickSort3Elements(int *Locs, int NumLocs, char Axis1, char Axis2, char Axis3);
		void QuickSort3Elements(int *Locs, int NumLocs, char Axis);
		void QuickSortLocs(int *Locs, int p, int r);
		void SwapLocs(int& x, int& y);
		int  PartitionLocs(int *Locs, int low, int high);
		
	public:
		double BoundaryFitting(int MaterialNum);
		void SaveSecondDerivative(char *OutFileName, int NumLeaping);
		void RemoveSecondD();
		void DisplayIGS(int X1, int X2, int Y1, int Y2, int Z1, int Z2, float *DistanceV);
		void Destroy();

};

template<class _DataType>
void QuickSort(_DataType* data, int p, int r);

template<class _DataType>
void Swap(_DataType& x, _DataType& y);

template<class _DataType>
int  Partition(_DataType* data, int low, int high );

template<class T>
T *Bilateral_Filter(T *data, int WindowSize, double& Min, double& Max, double S1, double S2);

template<class T>
void Anisotropic_Diffusion_3DScalar(T *ScalarValues, int NumIterations);

extern int  CubeEdgeDir[8][8][3];

#endif


