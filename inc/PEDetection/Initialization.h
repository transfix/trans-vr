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

#ifndef FILE_INITIALIZATION_H
#define FILE_INITIALIZATION_H

#include <PEDetection/GVF.h>

//template <class _DataType> 
//class cGVF;


template <class _DataType> 
class cInitialValue
{
	public:
	  typedef cGVF<_DataType>	ClassGVF;

	protected:
		_DataType	*Data_mT;
		float		MinData_mf, MaxData_mf;
		float		*Gradient_mf;
		int			*Histogram_mi;
//		int			*Accumulated_Histogram_mi;

		int			NumInitialValues_mi;
		int			MaxNumInitialValues_mi;
		_DataType	*InitialValues_mT;
		_DataType	*LowerBoundInitialValues_mT;
		_DataType	*UpperBoundInitialValues_mT;
		int			*InitialValueLocations_mi;

		int			Width_mi, Height_mi, Depth_mi;
		int			WtimesH_mi, WHD_mi;

		ClassGVF	*gvf_m;
		

	public:
		cInitialValue();
		~cInitialValue();

		void setGradient(float *Grad) { Gradient_mf = Grad; }
		void setData(_DataType *Data, float Min, float Max) { Data_mT = Data; MinData_mf=Min, MaxData_mf=Max; }
		void setHistogram(int *Histogram);
		void setWHD(int W, int H, int D);
		void setGVF(ClassGVF *gvf) { gvf_m = gvf; };

		int getNumInitialValues() { return NumInitialValues_mi; }

		void getInitalValues(_DataType *InitValues);
		void getInitialValueLocations(int *InitLoc);


	public:
		// Using GVF Seed Points as initial values
		int  FindInitialValues(int NumSeedPts, int *SeedPtsLocations);
		int  AgglomerativeMerging(int NumMaterials);
	private:
		int IsLocalMax(int xi, int yj, int zk, _DataType& Min, _DataType& Max);
		int IsLocalMin(int xi, int yj, int zk, _DataType& Min, _DataType& Max);
		void AddInitialValue(int xi, int yj, int zk, _DataType LocalMin, _DataType LocalMax);
		void UpdateMinMaxInitialValue(int xi, int yj, int zk, _DataType Min, _DataType Max);
		void UpdateMinMaxInitialValue(int xi, int yj, int zk, _DataType Min, _DataType Max, _DataType Ave);
		void RangeMergeInitialValues();								// Range Only
		int  RangeHistogramMergeInitialValues();	// Range + Histogram
		void DistanceMergeInitialValues(int NumMaterials);
		void DistanceHistogramMergeInitialValues(int NumMaterials);
		void MaxGradientMagnitudeMerge(int NumMaterials);
		int FindMinMaxNeighbor(int xi, int yj, int zk, _DataType& Min, _DataType& Max);	
	private: // Quick Sort Related Functions
		void QuickSortInitialValues();
		void QuickSort(_DataType* data, int p, int r);

		void Swap(unsigned char& x, unsigned char& y);
		void Swap(unsigned short& x, unsigned short& y);
		void Swap(float& x, float& y);
		void Swap(int& x, int& y);
		int  Partition(_DataType* data, int low, int high );


	private:
		void MergingOverlappedArea(int NumMaterials);
		int FindMaxOverlappedArea(int &Area1, int &Area2);
		void DisplayOverlappedAreaMatrix();
		
	public:
		void DisplayInitialValuesRange();
		void DisplayInitialValuesRange(int FirstN);
	
};


extern char TargetName_gc[512], InputFileName_gc[512];

void SaveImage(int Width, int Height, unsigned char *Image, char *PPMFile);

#endif

