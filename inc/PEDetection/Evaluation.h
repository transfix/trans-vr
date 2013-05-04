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

#ifndef FILE_EVALUATION_H
#define FILE_EVALUATION_H

#include <map.h>

#define MAX_MATRANGES	50

template <class _DataType> 
struct sMatRangeInfo {
	_DataType		DataMin, DataMax;
	double			AveFirstD, AveSecondD;
	int				NumBoundaries;
	unsigned char	ithMaterial, TotalNumMaterials;
};


template <class _DataType> 
class cEvaluation
{
	protected:
		_DataType	*Data_mT;
		float		MinData_mf, MaxData_mf;
		float		*GradientMag_mf;
		float		MinGradMag_mf, MaxGradMag_mf;
		int			*Histogram_mi;
		float		HistogramFactor_mf;
		int			Width_mi, Height_mi, Depth_mi;
		int			WtimesH_mi, WHD_mi;
		float		*SecondDerivative_mf;
		float		MinSecond_mf, MaxSecond_mf;
		
		int							NumMatRanges_mi;
		sMatRangeInfo<_DataType>	MatRanges_ms[MAX_MATRANGES];
		
		map<double, sMatRangeInfo<_DataType> *> MatRange_mm;


	public:
		cEvaluation();
		~cEvaluation();

		void setGradient(float *Grad, float Min, float Max);
		void setData(_DataType *Data, float Min, float Max);
		void setHistogram(int* Histo, float HistoF);
		void setWHD(int W, int H, int D);
		void setSecondDerivative(float *SecondD, float Min, float Max);
		void FindAndEvaluateRanges(float *Material_Prob, int NumClusters);

	private:
		int IsMaterialBoundary(int DataLoc, _DataType DataMin, _DataType DataMax);
		int AddAMatRange(sMatRangeInfo<_DataType>& MatRange, double SecondD);
		void BoundaryFitting(int NumRanges);

	public:
		void DisplayMatRangeInfo();
		void DisplayMatRangeInfoFormated();
		
};


#endif

