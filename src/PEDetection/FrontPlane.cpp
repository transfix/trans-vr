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

#include <math.h>
#include <float.h>
#include <PEDetection/FrontPlane.h>


//--------------------------------------------------------------
// Skeleton Voxels
//--------------------------------------------------------------

cSkeletonVoxel::cSkeletonVoxel()
{
	Xi = Yi = Zi = 0;
	Generation_i = 0;
	AV_c = ' ';
	LR_c = ' ';
	End_c = ' ';
	PrevVoxel_i = -1;
	for (int i=0; i<MAXNUM_NEXT; i++) NextVoxels_i[i] = -1;
}

cSkeletonVoxel::~cSkeletonVoxel()
{

}

void cSkeletonVoxel::set(int X, int Y, int Z)
{
	Xi = X;
	Yi = Y;
	Zi = Z;
}

void cSkeletonVoxel::set(int X, int Y, int Z, int Loc)
{
	Xi = X;
	Yi = Y;
	Zi = Z;
	AddNextVoxel(Loc);
}

void cSkeletonVoxel::set(int X, int Y, int Z, char End)
{
	this->Xi = X;
	this->Yi = Y;
	this->Zi = Z;
	this->End_c = End;
}

void cSkeletonVoxel::getXYZ(int &X, int &Y, int &Z)
{
	X = this->Xi;
	Y = this->Yi;
	Z = this->Zi;
}

void cSkeletonVoxel::getXYZ(float &Xf, float &Yf, float &Zf)
{
	Xf = (float)this->Xi;
	Yf = (float)this->Yi;
	Zf = (float)this->Zi;
}

int cSkeletonVoxel::DoesExistNext(int Loc)
{
	int		i, Exist_i;


	Exist_i = 0;
	for (i=0; i<MAXNUM_NEXT; i++) {
		if (NextVoxels_i[i]==Loc) {
			Exist_i = 1;
			break;
		}
	}
	return Exist_i;
}

int cSkeletonVoxel::DoesMatchType(cSkeletonVoxel *source)
{
	if (this->AV_c!=source->AV_c) return 0;
	if (this->LR_c!=source->LR_c) return 0;
	return 1;
}
void cSkeletonVoxel::AddNextVoxel(int Loc)
{
	int		i;
	

	if (DoesExistNext(Loc)==0) {
		for (i=0; i<MAXNUM_NEXT; i++) {
			if (NextVoxels_i[i]==-1) {
				NextVoxels_i[i] = Loc;
				break;
			}
		}
	}
}

int	cSkeletonVoxel::getNumNext()
{
	int		i, NumNext_i = 0;
	
	for (i=0; i<MAXNUM_NEXT; i++) {
		if (NextVoxels_i[i]>0) NumNext_i++;
		else break;
	}

	return NumNext_i;
}

void cSkeletonVoxel::RemoveNextID(int Loc)
{
	int		i, j, SavedNextIDs_i[MAXNUM_NEXT];
	
	
	for (i=0; i<MAXNUM_NEXT; i++) {
		SavedNextIDs_i[i] = this->NextVoxels_i[i];
		this->NextVoxels_i[i] = -1;
	}
	
	for (j=0, i=0; i<MAXNUM_NEXT; i++) {
		if (SavedNextIDs_i[i]==Loc) { }
		else {
			this->NextVoxels_i[j] = SavedNextIDs_i[i];
			j++;
		}
	}
}

void cSkeletonVoxel::Copy(cSkeletonVoxel *Source)
{
	this->Xi = Source->Xi;
	this->Yi = Source->Yi;
	this->Zi = Source->Zi;
	this->Generation_i = Source->Generation_i;
	this->AV_c = Source->AV_c;
	this->LR_c = Source->LR_c;
	this->End_c = Source->End_c;
	this->PrevVoxel_i = Source->PrevVoxel_i;
	for (int i=0; i<MAXNUM_NEXT; i++) {
		this->NextVoxels_i[i] = Source->NextVoxels_i[i];
	}
}

void cSkeletonVoxel::Display()
{

	printf ("%3d %3d %3d, ", this->Xi, this->Yi, this->Zi);
	printf ("Generation = %2d ", this->Generation_i);

	if (this->LR_c=='L') printf ("Left ");
	if (this->LR_c=='R') printf ("Right ");

	if (this->AV_c=='A') printf ("Artery ");
	if (this->AV_c=='V') printf ("Vein ");
	printf ("\n"); fflush (stdout);
}


//--------------------------------------------------------------
// Seed Pts Info
//--------------------------------------------------------------
cSeedPtsInfo::cSeedPtsInfo()
{
	Ave_f = Std_f = -1;
	Min_f = FLT_MAX;
	Max_f = -FLT_MAX;
	Median_f = FLT_MAX;
	Type_i = CLASS_UNKNOWN;
	MaxSize_i = -1;
	CCID_i = -1;
	NumOpenVoxels_i = -1;
	Traversed_i = 0;
	LungSegValue_uc = 0;
	LoopID_i = -1;
	NumBloodVessels_i = 0;
	TowardHeart_SpID_i = -1;
	ConnectedNeighbors_s.setDataPointer(0);
	for (int i=0; i<3; i++) {
		BVCenter_LocXYZ_i[i] = -1;
		MovedCenterXYZ_i[i] = -1;
		Direction_f[i] = (float)0.0;
	}
}

cSeedPtsInfo::~cSeedPtsInfo()
{
//	ConnectedNeighbors_s.Destroy();
}

void cSeedPtsInfo::Init()
{
	Ave_f = Std_f = -1;
	Min_f = FLT_MAX;
	Max_f = -FLT_MAX;
	Median_f = FLT_MAX;
	Type_i = CLASS_UNKNOWN;
	MaxSize_i = -1;
	CCID_i = -1;
	NumOpenVoxels_i = -1;
	Traversed_i = 0;
	LungSegValue_uc = 0;
	TowardHeart_SpID_i = -1;
	ConnectedNeighbors_s.setDataPointer(0);
	for (int i=0; i<3; i++) {
		BVCenter_LocXYZ_i[i] = -1;
		MovedCenterXYZ_i[i] = -1;
		Direction_f[i] = (float)0.0;
	}
}

void cSeedPtsInfo::Init(int Type)
{
	Ave_f = Std_f = -1;
	Min_f = FLT_MAX;
	Max_f = -FLT_MAX;
	Median_f = FLT_MAX;
	Type_i = Type;
	MaxSize_i = -1;
	CCID_i = -1;
	NumOpenVoxels_i = -1;
	Traversed_i = 0;
	LungSegValue_uc = 0;
	TowardHeart_SpID_i = -1;
	ConnectedNeighbors_s.setDataPointer(0);
	for (int i=0; i<3; i++) {
		BVCenter_LocXYZ_i[i] = -1;
		MovedCenterXYZ_i[i] = -1;
		Direction_f[i] = (float)0.0;
	}
}


int cSeedPtsInfo::getNextNeighbor(int PrevSphereID)
{
	int		i, SphereID1;
	
	
	for (i=0; i<this->ConnectedNeighbors_s.Size(); i++) {
		this->ConnectedNeighbors_s.IthValue(i, SphereID1);
		if (SphereID1!=PrevSphereID) return SphereID1;
	}
	
	return -1;
}

int cSeedPtsInfo::getNextNeighbors(int PrevSphereID, cStack<int> &Neighbors_ret)
{
	int		i, NextSpID_i;
	
	
	Neighbors_ret.setDataPointer(0);
	for (i=0; i<this->ConnectedNeighbors_s.Size(); i++) {
		this->ConnectedNeighbors_s.IthValue(i, NextSpID_i);
		if (PrevSphereID!=NextSpID_i) Neighbors_ret.Push(NextSpID_i);
	}
	
	return Neighbors_ret.Size();
}

int cSeedPtsInfo::getNextNeighbors(int PrevSpID1, int PrevSpID2, cStack<int> &Neighbors_ret)
{
	int		i, NextSpID_i;
	
	
	Neighbors_ret.setDataPointer(0);
	for (i=0; i<this->ConnectedNeighbors_s.Size(); i++) {
		this->ConnectedNeighbors_s.IthValue(i, NextSpID_i);
		if (PrevSpID1!=NextSpID_i && PrevSpID2!=NextSpID_i) Neighbors_ret.Push(NextSpID_i);
	}
	
	return Neighbors_ret.Size();
}

int cSeedPtsInfo::getNextNeighbors(int PrevSpID1, int PrevSpID2)
{
	int		i, NextSpID_i;
	
	for (i=0; i<this->ConnectedNeighbors_s.Size(); i++) {
		this->ConnectedNeighbors_s.IthValue(i, NextSpID_i);
		if (PrevSpID1!=NextSpID_i && PrevSpID2!=NextSpID_i) return NextSpID_i;
	}
	return -1;
}


int cSeedPtsInfo::ComputeDistance(cSeedPtsInfo& src)
{
	int Distance_i, X1, Y1, Z1, X2, Y2, Z2;
	
	X1 = this->MovedCenterXYZ_i[0];
	Y1 = this->MovedCenterXYZ_i[1];
	Z1 = this->MovedCenterXYZ_i[2];
	X2 = src.MovedCenterXYZ_i[0];
	Y2 = src.MovedCenterXYZ_i[1];
	Z2 = src.MovedCenterXYZ_i[2];
	
	Distance_i = (X2-X1)*(X2-X1) + (Y2-Y1)*(Y2-Y1) + (Z2-Z1)*(Z2-Z1);
	return Distance_i;
}

void cSeedPtsInfo::Copy(cSeedPtsInfo& src)
{
	int		i;
	for (i=0; i<3; i++) this->BVCenter_LocXYZ_i[i] = src.BVCenter_LocXYZ_i[i];
	for (i=0; i<3; i++) this->MovedCenterXYZ_i[i] = src.MovedCenterXYZ_i[i];
	for (i=0; i<3; i++) this->Direction_f[i] = src.Direction_f[i];
	this->Ave_f = src.Ave_f;
	this->Std_f = src.Std_f;
	this->Median_f = src.Median_f;
	this->Min_f = src.Min_f;
	this->Max_f = src.Max_f;
	this->MaxSize_i = src.MaxSize_i;
	this->Type_i = src.Type_i;
	this->ConnectedNeighbors_s.setDataPointer(0);
	this->ConnectedNeighbors_s.Copy(src.ConnectedNeighbors_s);
	this->TowardHeart_SpID_i = src.TowardHeart_SpID_i;
	this->NumOpenVoxels_i = src.NumOpenVoxels_i;
	this->CCID_i = src.CCID_i;
	this->LoopID_i = src.LoopID_i;
	this->LungSegValue_uc = src.LungSegValue_uc;
	this->Traversed_i = src.Traversed_i;
	this->NumBloodVessels_i = src.NumBloodVessels_i;
	this->General_i = src.General_i;
}

void cSeedPtsInfo::Copy(cSeedPtsInfo *src)
{
	int		i;
	for (i=0; i<3; i++) this->BVCenter_LocXYZ_i[i] = src->BVCenter_LocXYZ_i[i];
	for (i=0; i<3; i++) this->MovedCenterXYZ_i[i] = src->MovedCenterXYZ_i[i];
	for (i=0; i<3; i++) this->Direction_f[i] = src->Direction_f[i];
	this->Ave_f = src->Ave_f;
	this->Std_f = src->Std_f;
	this->Median_f = src->Median_f;
	this->Min_f = src->Min_f;
	this->Max_f = src->Max_f;
	this->MaxSize_i = src->MaxSize_i;
	this->Type_i = src->Type_i;
	this->ConnectedNeighbors_s.setDataPointer(0);
	this->ConnectedNeighbors_s.Copy(src->ConnectedNeighbors_s);
	this->TowardHeart_SpID_i = src->TowardHeart_SpID_i;
	this->NumOpenVoxels_i = src->NumOpenVoxels_i;
	this->CCID_i = src->CCID_i;
	this->LoopID_i = src->LoopID_i;
	this->LungSegValue_uc = src->LungSegValue_uc;
	this->Traversed_i = src->Traversed_i;
	this->NumBloodVessels_i = src->NumBloodVessels_i;
	this->General_i = src->General_i;
}


int cSeedPtsInfo::getType(char Type_ret[])
{
	int		i=0;
	char	Temp_c[]="  ";
	char	T_c[] = "HAOVNDBIT";
	char	E_c[] = "         ";
	
	if ((Type_i & CLASS_HEART  )==CLASS_HEART 			) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_ARTERY )==CLASS_ARTERY 			) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_ARTERY_ROOT )==CLASS_ARTERY_ROOT) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_VEIN   )==CLASS_VEIN   			) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_REMOVED)==CLASS_REMOVED			) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_DEADEND)==CLASS_DEADEND			) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_HEARTBRANCH)==CLASS_HEARTBRANCH	) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_LIVEEND)==CLASS_LIVEEND 		) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	if ((Type_i & CLASS_HITHEART )==CLASS_HITHEART 		) Type_ret[i]=T_c[i++]; else Type_ret[i]=E_c[i++];
	Type_ret[i++] = Temp_c[2];
	Type_ret[i++] = Temp_c[3];
	return this->Type_i;
}

void cSeedPtsInfo::DisplayType()
{
	if ((Type_i & CLASS_HEART  )==CLASS_HEART 			) printf ("H"); else printf (" ");
	if ((Type_i & CLASS_ARTERY )==CLASS_ARTERY 			) printf ("A"); else printf (" ");
	if ((Type_i & CLASS_ARTERY_ROOT )==CLASS_ARTERY_ROOT) printf ("O"); else printf (" ");
	if ((Type_i & CLASS_VEIN   )==CLASS_VEIN   			) printf ("V"); else printf (" ");
	if ((Type_i & CLASS_REMOVED)==CLASS_NOT_CONNECTED	) printf ("N"); else printf (" ");
	if ((Type_i & CLASS_DEADEND)==CLASS_DEADEND			) printf ("D"); else printf (" ");
	if ((Type_i & CLASS_HEARTBRANCH)==CLASS_HEARTBRANCH	) printf ("B"); else printf (" ");
	if ((Type_i & CLASS_LIVEEND)==CLASS_LIVEEND 		) printf ("I"); else printf (" ");
	if ((Type_i & CLASS_HITHEART )==CLASS_HITHEART 		) printf ("T"); else printf (" ");
	printf (" ");
	printf ("Type# = %4d ", Type_i);
	fflush(stdout);
}

void cSeedPtsInfo::Display()
{
	printf ("%3d %3d %3d ", MovedCenterXYZ_i[0], MovedCenterXYZ_i[1], MovedCenterXYZ_i[2]);
	printf ("MaxR = %2d ", MaxSize_i);
	printf ("#N = %2d ", ConnectedNeighbors_s.Size());
	printf ("\n"); fflush (stdout);
}


//--------------------------------------------------------------
// CC Info
//--------------------------------------------------------------

cCCInfo::cCCInfo()
{
	CCID_i = -1;
	MinR_i = 100;
	MaxR_i = 0;
	ConnectedPts_s.setDataPointer(0);
	MinN_i = 100;
	MaxN_i = 0;
	MinLungSeg_i = 256;
	MaxLungSeg_i = 0;
	NumLines_i = -1;
}

cCCInfo::~cCCInfo()
{

}

void cCCInfo::Copy(cCCInfo *Src)
{
	this->CCID_i = Src->CCID_i;
	this->MinR_i = Src->MinR_i;
	this->MaxR_i = Src->MaxR_i;
	this->MinN_i = Src->MinN_i;
	this->MaxN_i = Src->MaxN_i;
	this->MinLungSeg_i = Src->MinLungSeg_i;
	this->MaxLungSeg_i = Src->MaxLungSeg_i;
	this->NumLines_i = Src->NumLines_i;
	this->ConnectedPts_s.setDataPointer(0);
	this->ConnectedPts_s.Copy(Src->ConnectedPts_s);

/*
	int		i, Idx;
	for (i=0; i<Src->ConnectedPts_s.Size(); i++) {
		Src->ConnectedPts_s.IthValue(i, Idx);
		this->ConnectedPts_s.Push(Idx);
	}
*/
}

void cCCInfo::Copy(cCCInfo &Src)
{
	int		i, Idx;
	this->CCID_i = Src.CCID_i;
	this->MinR_i = Src.MinR_i;
	this->MaxR_i = Src.MaxR_i;
	this->MinN_i = Src.MinN_i;
	this->MaxN_i = Src.MaxN_i;
	this->MinLungSeg_i = Src.MinLungSeg_i;
	this->MaxLungSeg_i = Src.MaxLungSeg_i;
	this->NumLines_i = Src.NumLines_i;
	this->ConnectedPts_s.setDataPointer(0);
	for (i=0; i<Src.ConnectedPts_s.Size(); i++) {
		Src.ConnectedPts_s.IthValue(i, Idx);
		this->ConnectedPts_s.Push(Idx);
	}
}


//--------------------------------------------------------------
// Front Planes
//--------------------------------------------------------------

cFrontPlane::cFrontPlane()
{
	int		i;
	
	WaveTime_i = -1;
	TotalSize = -1;
	for (i=0; i<6; i++) {
		CenterPt_f[i*3+0] = -1.0;
		CenterPt_f[i*3+1] = -1.0;
		CenterPt_f[i*3+2] = -1.0;
	}
	for (i=0; i<6; i++) {
		VoxelLocs_s[i].setDataPointer(0);
	}
	for (i=0; i<256; i++) Histogram_i[i] = 0;
	AveIntensity_f = 0.0;
	Std_f = 0.0;
}


cFrontPlane::~cFrontPlane()
{
	for (int i=0; i<6; i++) {
		VoxelLocs_s[i].Destroy();
	}
}


void cFrontPlane::Init()
{
	int		i;
	
	WaveTime_i = -1;
	TotalSize = -1;
	for (i=0; i<6; i++) {
		CenterPt_f[i*3+0] = -1.0;
		CenterPt_f[i*3+1] = -1.0;
		CenterPt_f[i*3+2] = -1.0;
	}
	for (i=0; i<6; i++) VoxelLocs_s[i].setDataPointer(0);
	for (i=0; i<256; i++) Histogram_i[i] = 0;
	AveIntensity_f = 0.0;
	Std_f = 0.0;
}


void cFrontPlane::getAveStd(float &Ave_ret, float &Std_ret)
{
	Ave_ret = AveIntensity_f;
	Std_ret = Std_f;
}


void cFrontPlane::ComputeAveStd()
{
	int		i, NumVoxels= 0;
	double	Tempd = 0.0;
	
	
	for (i=0; i<256; i++) {
		Tempd += (double)Histogram_i[i]*i;
		NumVoxels += Histogram_i[i];
	}
	AveIntensity_f = (float)(Tempd / NumVoxels);
	
	Tempd = 0.0;
	for (i=0; i<256; i++) {
		if (Histogram_i[i]==0) continue;
		Tempd += ((double)AveIntensity_f - i)*((double)AveIntensity_f - i)*Histogram_i[i];
	}
	Tempd /= (double)NumVoxels;
	Std_f = (float)sqrt(Tempd);
}

void cFrontPlane::AddToHistogram(int Value)
{
	if (Value>=0 && Value<=255) Histogram_i[Value]++;
	else { 
		printf ("cFrontPlane::AddToHistogram: Out of range\n"); fflush (stdout);
	}
}

// Deep Copy
void cFrontPlane::Copy(cFrontPlane *Source)
{
	int		i, j, Locs;
	
	this->WaveTime_i = Source->WaveTime_i;
	this->TotalSize = Source->TotalSize;
	for (i=0; i<6; i++) {
		this->CenterPt_f[i*3+0] = Source->CenterPt_f[i*3+0];
		this->CenterPt_f[i*3+1] = Source->CenterPt_f[i*3+1];
		this->CenterPt_f[i*3+2] = Source->CenterPt_f[i*3+2];
	}
	for (i=0; i<6; i++) {
		this->VoxelLocs_s[i].setDataPointer(0);
		for (j=0; j<Source->VoxelLocs_s[i].Size(); j++) {
			Source->VoxelLocs_s[i].IthValue(j, Locs);
			this->VoxelLocs_s[i].Push(Locs);
		}
	}
	for (i=0; i<256; i++) this->Histogram_i[i] = Source->Histogram_i[i];
	this->AveIntensity_f = Source->AveIntensity_f;
	this->Std_f = Source->Std_f;
}


void cFrontPlane::CopyIthStack(int ith, cStack<int> &Source_Stack)
{
	int		j, Locs;
	this->VoxelLocs_s[ith].setDataPointer(0);
	for (j=0; j<Source_Stack.Size(); j++) {
		Source_Stack.IthValue(j, Locs);
		this->VoxelLocs_s[ith].Push(Locs);
	}
}


void cFrontPlane::DeleteStack()
{
	for (int i=0; i<6; i++) {
		VoxelLocs_s[i].Destroy();
	}
}


void cFrontPlane::Display()
{
	int		i;
	printf ("WaveTime_i = %2d, ", WaveTime_i);
	printf ("TotalSize = %4d, ", TotalSize);
	printf ("Each Plane Size = ");
	for (i=0; i<6; i++) {
		printf ("%d=%d, ", i, VoxelLocs_s[i].Size());
	}
	printf ("\n"); fflush (stdout);
	printf ("Histogram\n");
	for (i=0; i<256; i++) {
		if (Histogram_i[i]>0) printf (" %3d ", i);
	}
	printf ("\n"); fflush (stdout);
	for (i=0; i<256; i++) {
		if (Histogram_i[i]>0) printf ("%4d ", Histogram_i[i]);
	}
	printf ("\n"); fflush (stdout);
	printf ("Ave = %f, ", AveIntensity_f);
	printf ("Std = %f", Std_f);
	printf ("\n"); fflush (stdout);
}


