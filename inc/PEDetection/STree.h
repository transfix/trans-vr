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

#ifndef FILE_STREE_H
#define FILE_STREE_H


#include <map.h>
#include <deque.h>

#include <PEDetection/FrontPlane.h>


#define	MAX_HISTO_ELEMENTS		50


template <class _DataType> 
class cSTree_Eval {

	protected:
		int			Width_mi, Height_mi, Depth_mi;
		int			WtimesH_mi, WHD_mi;
		
		_DataType	*Data_mT;
		float		MinData_mf, MaxData_mf;


		unsigned char	*BVVolume_muc;
		char			CurrAVType_mc;
		int				TotalNumBranches_mi, NumWrongClasss_mi, NumMissing_mi;
		int				NumWrongArteries_mi, NumWrongVeins_mi;
		int				NumRepeat_Eval_mi;
		int				NumWrongBranch_Generations_mi[20];

		int				Range_BloodVessels_mi[2];
//		int				Range_Soft_mi[2];
		int				Range_Muscles_mi[2];
//		int				Range_Lungs_mi[2];

		float			VoxelSize_gf;

		int				HistoWrongA_mi[MAX_HISTO_ELEMENTS];	// Artery
		int				HistoWrongV_mi[MAX_HISTO_ELEMENTS];	// Vein
		int				HistoWrongM_mi[MAX_HISTO_ELEMENTS];	// Missing
		int				HistoWrongT_mi[MAX_HISTO_ELEMENTS];	// Total

		
		
		map<int, cSkeletonVoxel *>			StartLoc_mmap;
		map<float, cSkeletonVoxel *>		SkeletonDeadEnds_mmap;
		map<int, cSkeletonVoxel *>			Skeleton_mmap;




	public:
		cSTree_Eval();
		~cSTree_Eval();		


	public:
		void setWHD(int W, int H, int D);
		void setData(_DataType *Data, float Minf, float Maxf);
		void AVSeparation_Evaluate(unsigned char *BV);
		void LoadTreeStructure(char *SkeletonFileName_c);
		void setMuscleRange(int M1, int M2);
		void setBVRange(int BV1, int BV2);

	protected:
		void Evaluation(int CurrLoc);
		int RecomputingMaxR_InBV_Muscles(int Xi, int Yi, int Zi);
		int RecomputingMaxR_InBloodVessels(int Xi, int Yi, int Zi);
		int *getSphereIndex(int SphereRadius, int &NumVoxels_ret);
		void ComputingPixelSpacing();
		int ComputingTheBiggestSphereAt_DataRange(cStack<int> &VoxelLocs, 
								_DataType Lower_Th, _DataType Upper_Th, 
								int *Center3_ret, int &SphereR_ret);
		
	private:
		int Index(int X, int Y, int Z);
		int Index(int *Loc3);
		void IndexInverse(int Loc, int &X, int &Y, int &Z);
		void IndexInverse(int Loc, int *Center3_ret);
	

};



extern char TargetName_gc[512];


extern int SphereR00_gi[];
extern int SphereR01_gi[];
extern int SphereR02_gi[];
extern int SphereR03_gi[];
extern int SphereR04_gi[];
extern int SphereR05_gi[];
extern int SphereR06_gi[];
extern int SphereR07_gi[];
extern int SphereR08_gi[];
extern int SphereR09_gi[];
extern int SphereR10_gi[];
extern int SphereR11_gi[];
extern int SphereR12_gi[];
extern int SphereR13_gi[];
extern int SphereR14_gi[];
extern int SphereR15_gi[];
extern int SphereR16_gi[];
extern int SphereR17_gi[];
extern int SphereR18_gi[];
extern int SphereR19_gi[];
extern int SphereR20_gi[];
extern int SphereR21_gi[];
extern int SphereR22_gi[];
extern int SphereR23_gi[];
extern int SphereR24_gi[];
extern int SphereR25_gi[];
extern int SphereR26_gi[];

extern int NumSphereR00_gi;
extern int NumSphereR01_gi;
extern int NumSphereR02_gi;
extern int NumSphereR03_gi;
extern int NumSphereR04_gi;
extern int NumSphereR05_gi;
extern int NumSphereR06_gi;
extern int NumSphereR07_gi;
extern int NumSphereR08_gi;
extern int NumSphereR09_gi;
extern int NumSphereR10_gi;
extern int NumSphereR11_gi;
extern int NumSphereR12_gi;
extern int NumSphereR13_gi;
extern int NumSphereR14_gi;
extern int NumSphereR15_gi;
extern int NumSphereR16_gi;
extern int NumSphereR17_gi;
extern int NumSphereR18_gi;
extern int NumSphereR19_gi;
extern int NumSphereR20_gi;
extern int NumSphereR21_gi;
extern int NumSphereR22_gi;
extern int NumSphereR23_gi;
extern int NumSphereR24_gi;
extern int NumSphereR25_gi;
extern int NumSphereR26_gi;





#endif

