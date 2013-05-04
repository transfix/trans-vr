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

#include <PEDetection/MembraneSeg.h>

template<class _DataType> 
cMembraneSeg<_DataType> ::cMembraneSeg()
{	

}

template<class _DataType>
cMembraneSeg<_DataType>::~cMembraneSeg()
{

}

template<class _DataType>
void cMembraneSeg<_DataType>::setData(_DataType *Data, float Min, float Max)
{
	Data_mT = Data; 
	MinData_mf = Min, 
	MaxData_mf = Max; 
}

template<class _DataType>
void cMembraneSeg<_DataType>::setWHD(int W, int H, int D)
{
	Width_mi = W;
	Height_mi = H;
	Depth_mi = D;
	
	WtimesH_mi = W*H;
	WHD_mi = W*H*D;
}

template<class _DataType>
void cMembraneSeg<_DataType>::setZeroSurface(double ZeroSurface)
{
	ZeroSurfaceValue_md = ZeroSurface;
}


template<class _DataType>
void cMembraneSeg<_DataType>::setFileName(char *FileName)
{
	TargetName_mc = FileName;
}


template<class _DataType>
_DataType* cMembraneSeg<_DataType>::Class(int ClassNum)
{
	int		binfile_fd1;
	int		i, Classes[256];

//	char	ClassFileName[] = "/work/smpark/data-NMJ/membrane0513_Classes_3.rawiv";
//	char	ClassFileName[] = "/work/smpark/data-NMJ/membrane0513_Classes_5.rawiv";
	char	ClassFileName[] = "/work/smpark/data-NMJ/membrane_1_Classes.rawiv";

//	char	ClassFileName[] = "/work/smpark/data-NMJ/membrane1024_Classes_5.rawiv";
//	char	ClassFileName[] = "/work/smpark/data-NMJ/membrane1024_Classes_3.rawiv";
//	char	ClassFileName[] = "/work/smpark/data-NMJ/membrane0256_Classes.rawiv";


	printf ("Class() ...\n"); fflush (stdout);

	if ((binfile_fd1 = open (ClassFileName, O_RDONLY)) < 0) {
		printf ("could not open %s\n", ClassFileName); fflush (stdout);
		exit(1);
	}
	
	unsigned char *RawivHeader_guc = new unsigned char [68];
	read(binfile_fd1, RawivHeader_guc, 68); // Remove the rawiv file header
	
	ClassData_muc = new unsigned char [WHD_mi];
	if (read(binfile_fd1, ClassData_muc, sizeof(unsigned char)*WHD_mi) != sizeof(unsigned char)*WHD_mi) {
		printf ("The file could not be read: %s\n", ClassFileName); fflush (stdout);
		close (binfile_fd1);
		exit(1);
	}

/*
	int		loc[3], j, k, l, m, n;
	unsigned char *CopyClassData_muc = new unsigned char [WHD_mi];
	for (i=0; i<WHD_mi; i++) CopyClassData_muc[i] = ClassData_muc[i];
	
	for (k=1; k<Depth_mi-1; k++) {
		for (j=1; j<Height_mi-1; j++) {
			for (i=1; i<Width_mi-1; i++) {
				loc[0] = k*WtimesH_mi + j*Width_mi + i;
				
				if (CopyClassData_muc[loc[0]]==ClassNum) {
					for (n=k-1; n<=k+1; n++) {
						for (m=j; m<=j; m++) {
							for (l=i-1; l<=i+1; l++) {
								loc[1] = n*WtimesH_mi + m*Width_mi + l;
								ClassData_muc[loc[1]] = ClassNum;
							}
						}
					}
				}
				
			}
		}
	}
	delete [] CopyClassData_muc;
*/

	for (i=0; i<256; i++) Classes[i] = 0;

	_DataType	*NewData_T = new _DataType [WHD_mi];
	int		NumVoxels = 0;
	for (i=0; i<WHD_mi; i++) {
		if (ClassData_muc[i]!=ClassNum) NewData_T[i] = 100.0;
		else NumVoxels++;
		Classes[(int)ClassData_muc[i]]++;
	}
	printf ("# voxels of Class %d = %d\n", ClassNum, NumVoxels);
	for (i=0; i<256; i++) {
		if (Classes[i] > 0) {
			printf ("Class # %d --> # Voxels = %d\n", i, Classes[i]);
		}
	}
	fflush (stdout);
	
	return 	NewData_T;
}


template<class _DataType>
_DataType* cMembraneSeg<_DataType>::TopSurfaceSeg()
{
	int			i, j, k, loc[3], FoundSurface_i;
	_DataType	*TopSurfaceData_T = new _DataType [WHD_mi];
	_DataType	Temp_T;
	

	for (i=0; i<WHD_mi; i++) TopSurfaceData_T[i] = Data_mT[i];
	
	for (j=0; j<Height_mi; j++) {
		for (i=0; i<Width_mi; i++) {
			FoundSurface_i = false;
			for (k=Depth_mi-1; k>=Depth_mi*3/4; k--) {
				loc[0] = Index (i, j, k);
				loc[1] = Index (i, j, k-1);
				if (TopSurfaceData_T[loc[0]] > ZeroSurfaceValue_md && 
					TopSurfaceData_T[loc[1]] > ZeroSurfaceValue_md) TopSurfaceData_T[loc[0]] = 255;
				else {
					FoundSurface_i = true;
					Temp_T = TopSurfaceData_T[loc[1]];
					break;
				}
			}
//			if (FoundSurface_i==false) continue;
			k-=3;
			for (; k>=0; k--) {
				loc[0] = Index (i, j, k);
				TopSurfaceData_T[loc[0]] = 255;
			}
		}
	}

	return 	TopSurfaceData_T;
	
}

template<class _DataType>
_DataType* cMembraneSeg<_DataType>::DownSurfaceSeg()
{
	int			i, j, k, m, loc[3], FoundSurface_i;
	_DataType	*TopSurfaceData_T = new _DataType [WHD_mi];
	_DataType	Temp_T;
	

	for (i=0; i<WHD_mi; i++) TopSurfaceData_T[i] = Data_mT[i];
	
	for (j=0; j<Height_mi; j++) {
		for (i=0; i<Width_mi; i++) {
			FoundSurface_i = false;
			for (k=Depth_mi-1; k>=Depth_mi*3/4; k--) {
				loc[0] = Index (i, j, k);
				loc[1] = Index (i, j, k-1);
				if (TopSurfaceData_T[loc[0]] > ZeroSurfaceValue_md && 
					TopSurfaceData_T[loc[1]] > ZeroSurfaceValue_md) continue;
				else {
					FoundSurface_i = true;
					Temp_T = TopSurfaceData_T[loc[0]];
					break;
				}
			}
			if (FoundSurface_i==true) {
				for (m=Depth_mi-1; m>=k-3; m--) {
					loc[0] = Index (i, j, m);
					TopSurfaceData_T[loc[0]] = Temp_T;
				}
			}
		}
	}

	return 	TopSurfaceData_T;
	
}

template<class _DataType>
int	cMembraneSeg<_DataType>::Index(int X, int Y, int Z)
{
	if (X<0 || Y<0 || Z<0 || X>=Width_mi || Y>=Height_mi || Z>=Depth_mi) return 0;
	else return (Z*WtimesH_mi + Y*Width_mi + X);
}




cMembraneSeg<unsigned char>		__cMem0;
//cMembraneSeg<unsigned short>	__cMem1;
//cMembraneSeg<float>				__cMem2;

