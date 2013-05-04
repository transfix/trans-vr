/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <VolumeGridRover/EM.h>

#define TRUE	1
#define FALSE	0

#define SVMM		// Spatially Variant Mixture Model

using namespace std;

const double pi = 3.141592;
double GaussianDensity(double x, double mean, double var)
{
	double y;

	y = ((double)1.0/sqrt((double)2.0*pi*var))*exp(-1.0*(x-mean)*(x-mean)/(2*var));
	return y;
}


//----------------------------------------------------------------------------
// cEMClustering Class Member Functions
//----------------------------------------------------------------------------

template <class _DataType>
cEMClustering<class _DataType>::cEMClustering()
{
	PI_md = NULL;
	Mu_md = NULL;
	Sigma_md = NULL;
	PrevMu_md = NULL;
	PrevSigma_md = NULL;
	TotalProb_Data_md = NULL;
	W_md = NULL;
}

/* with number of pixles and number of clusters */
template <class _DataType>
cEMClustering<_DataType>::cEMClustering(int NumData, int NumClusters)
{
	int		i;
	
	NumDataPoints_mi = NumData;
	NumClusters_mi = NumClusters;

	PI_md = NULL;
	Mu_md = NULL;
	Sigma_md = NULL;
	PrevMu_md = NULL;
	PrevSigma_md = NULL;
	TotalProb_Data_md = NULL;
	W_md = NULL;

	PI_md = new double [NumClusters];			// Num PI_md 	= Num Clusters
	Mu_md = new double [NumClusters];			// Num Mu_md 	= Num Clusters
	Sigma_md = new double [NumClusters];		// Num Sigma_md = Num Clusters
	PrevMu_md = new double [NumClusters];		// Num Mu_md 	= Num Clusters
	PrevSigma_md = new double [NumClusters];	// Num Sigma_md = Num Clusters
	TotalProb_Data_md = new double [NumData];	// Num TotalProb_Data_md = Num Data 
	W_md = new double [NumData*NumClusters];	// Num W_md = Num Clusters * Num Data;


	for (i=0; i<NumClusters; i++) {
		PI_md[i] = (double)1.0/NumClusters;
	}
	for (i=0; i<NumData*NumClusters; i++) {
		W_md[i] = (double)1.0/NumClusters;
	}
	Histogram_mi = NULL;
}

/* destructor */
template <class _DataType>
cEMClustering<_DataType>::~cEMClustering()
{
	delete [] PI_md;
	delete [] Mu_md;
	delete [] Sigma_md;
	delete [] PrevMu_md;
	delete [] PrevSigma_md;
	delete [] TotalProb_Data_md;
	delete [] W_md;
}


template <class _DataType>
void cEMClustering<_DataType>::InitializeEM(int NumClusters)
{
	int		i, j;

	NumClusters_mi = NumClusters;
	for (i=0; i<NumClusters; i++) {
		PI_md[i] = (double)1.0/NumClusters;
	}
	for (i=0; i<NumDataPoints_mi; i++) {
		if (Histogram_mi[i]==0) {
			for (j=0; j<NumClusters; j++) {
				W_md[i*NumClusters + j] = (double)1.0/NumClusters;
			}
		}
		else {
			for (j=0; j<NumClusters; j++) {
				W_md[i*NumClusters + j] = (double)1.0/NumClusters;
			}
		}
	}
}


template <class _DataType>
void cEMClustering<_DataType>::setData(_DataType *Data, float Minf, float Maxf)
{
	Min_mf = Minf;
	Max_mf = Maxf;
	Data_m = Data;
}

template <class _DataType>
void cEMClustering<_DataType>::setHistogram(int* Histo, float HistoF)
{
	int 		i;
	
	Histogram_mi = Histo;
	HistogramFactor_mf = HistoF;
	if (Histogram_mi==NULL) {
		SumFrequencies_mi = NumDataPoints_mi;
	}
	else {
		SumFrequencies_mi=0;
		for (i=0; i<NumDataPoints_mi; i++) {
			SumFrequencies_mi += Histo[i];
		}
	}
	cout << "Sum Frequencies = " << SumFrequencies_mi << endl;
}


template <class _DataType>
void cEMClustering<_DataType>::setMeanVariance(int ith, double Mean,double variance)
{
	PrevMu_md[ith]		= Mu_md[ith]	= Mean;
	PrevSigma_md[ith]	= Sigma_md[ith] = variance;
//	cout << "Mean & Variance = " << Mu_md[ith] << " " << Sigma_md[ith] << endl;
}

template <class _DataType>
float *cEMClustering<_DataType>::getProbability()
{
	int		i, j, loc[2];
	float	*Probability;

	if (Histogram_mi==NULL) {

		// Format of the Probability 
		// Mat1		x*y*z
		// Mat2     x*y*z
		// Mat3     x*y*z
		//  ...      ...

		Probability = new float[NumDataPoints_mi*NumClusters_mi];

		for (j=0; j<NumClusters_mi; j++) {
			for (i=0; i<NumDataPoints_mi; i++) {

				loc[0]= i*NumClusters_mi + j;
				loc[1]= j*NumDataPoints_mi + i;
				Probability[loc[1]] = (float)W_md[loc[0]];
			}
		}
	}
	else {
		// Format of the Probability 
		// Data #    Mat1  Mat2  Mat3
		//   0       0.1   0.9   0.0
		//   1       0.0   0.1   0.9
		//  ...      ...

		Probability = new float[NumDataPoints_mi*NumClusters_mi];

		for (i=0; i<NumDataPoints_mi; i++) {
			for (j=0; j<NumClusters_mi; j++) {
				loc[0]= i*NumClusters_mi + j;
				Probability[loc[0]] = (float)W_md[loc[0]];
				if (Histogram_mi[i]==0) Probability[loc[0]] = 0;
			}
		}
	}
	return Probability;
}



template <class _DataType>
void cEMClustering<_DataType>::ComputeWij()
{
	int		i, j, loc[2];
	double	DataValue;

	for (i=0; i<NumDataPoints_mi; i++) {

		if (Histogram_mi==NULL) DataValue = (double)Data_m[i];
		else {
			if (Histogram_mi[i]==0) continue;
			DataValue = (double)i/HistogramFactor_mf + Min_mf;
		}

		TotalProb_Data_md[i]=0.0;
		for (j=0; j<NumClusters_mi; j++) {
#ifdef	SVMM
			loc[0]= i*NumClusters_mi + j;
			TotalProb_Data_md[i] += W_md[loc[0]]*GaussianDensity(DataValue, Mu_md[j], Sigma_md[j]);
#else
			TotalProb_Data_md[i] += PI_md[j]*GaussianDensity(DataValue, Mu_md[j], Sigma_md[j]);
#endif
		}
		if (TotalProb_Data_md[i]<1e-100) TotalProb_Data_md[i]=1e-100; 
		for (j=0; j<NumClusters_mi; j++) {
			loc[0]= i*NumClusters_mi + j;
#ifdef	SVMM
			W_md[loc[0]] = W_md[loc[0]]*GaussianDensity(DataValue, Mu_md[j], Sigma_md[j]);
#else
			W_md[loc[0]] = PI_md[j]*GaussianDensity(DataValue, Mu_md[j], Sigma_md[j]);
#endif
			W_md[loc[0]] /= TotalProb_Data_md[i];
			if (W_md[loc[0]]>1.0) W_md[loc[0]]=1.0;
		
/*
			cout << "Data, Mean & Variance = " << DataValue << "  ";
			cout << Mu_md[j] << "  " << Sigma_md[j] << endl;
			cout << "Gaussian = " << GaussianDensity(DataValue, Mu_md[j], Sigma_md[j]) << endl;
			cout << "Total Prob Data = " << TotalProb_Data_md[i] << endl;
			cout << "Wij = " << W_md[loc[0]] << endl;
*/
		}
	}
}

template <class _DataType>
void cEMClustering<_DataType>::ComputePIj()
{
	int		i, j, loc[2];
	double	Tempd=0.0;
	
	if (Histogram_mi==NULL) {
		for (j=0; j<NumClusters_mi; j++) {
			PI_md[j] = 0.0;
			for (i=0; i<NumDataPoints_mi; i++) {
				loc[0]= i*NumClusters_mi + j;
				PI_md[j] += W_md[loc[0]];
			}
			PI_md[j] /= SumFrequencies_mi;
	//		cout << "PI_md[j] = " << PI_md[j] << endl;
			Tempd += PI_md[j];
		}
	}
	else {
		for (j=0; j<NumClusters_mi; j++) {
			PI_md[j] = 0.0;
			for (i=0; i<NumDataPoints_mi; i++) {
				loc[0]= i*NumClusters_mi + j;
				PI_md[j] += W_md[loc[0]]*Histogram_mi[i];
			}
			PI_md[j] /= SumFrequencies_mi;
	//		cout << "PI_md[j] = " << PI_md[j] << endl;
			Tempd += PI_md[j];
		}
	}
//	cout << "Sum of PI_md = " << Tempd << endl;
}

template <class _DataType>
void cEMClustering<_DataType>::SaveCurrMeansVariances()
{
	int		j;
	for (j=0; j<NumClusters_mi; j++) {
		PrevMu_md[j] = Mu_md[j];
		PrevSigma_md[j] = Sigma_md[j];
	}
}


template <class _DataType>
void cEMClustering<_DataType>::ComputeNextMeans()
{
	int		i, j, loc[2];
	double	Tempd, DataValue;


	if (Histogram_mi==NULL) {
		for (j=0; j<NumClusters_mi; j++) {
			Tempd = 0.0;
			Mu_md[j] = 0.0;
			for (i=0; i<NumDataPoints_mi; i++) {
				loc[0]= i*NumClusters_mi + j;
				DataValue = (double)Data_m[i];
				Tempd += W_md[loc[0]];
				Mu_md[j] += W_md[loc[0]]*DataValue;
			}
			Mu_md[j] /= Tempd;
	//		cout << "Mean = " << Mu_md[j] << endl;
		}
	}
	else {
		for (j=0; j<NumClusters_mi; j++) {
			Tempd = 0.0;
			Mu_md[j] = 0.0;
			for (i=0; i<NumDataPoints_mi; i++) {
				loc[0]= i*NumClusters_mi + j;
				DataValue = (double)i/HistogramFactor_mf + Min_mf;
				Tempd += W_md[loc[0]]*Histogram_mi[i];
				Mu_md[j] += W_md[loc[0]]*DataValue*Histogram_mi[i];
			}
			Mu_md[j] /= Tempd;
	//		cout << "Mean = " << Mu_md[j] << endl;
		}
	}
}

template <class _DataType>
void cEMClustering<_DataType>::ComputeNextVariances()
{
	int		i, j, loc[2];
	double	Tempd, DataValue;


	if (Histogram_mi==NULL) {
		for (j=0; j<NumClusters_mi; j++) {
			Tempd = 0.0;
			Sigma_md[j] = 0.0;
			for (i=0; i<NumDataPoints_mi; i++) {
				loc[0]= i*NumClusters_mi + j;
				DataValue = (double)Data_m[i];
				Tempd += W_md[loc[0]];
				Sigma_md[j] += W_md[loc[0]]*(DataValue-Mu_md[j])*(DataValue-Mu_md[j]);
			}
			Sigma_md[j] /= Tempd;
			if (fabs(Sigma_md[j])<1e-6) Sigma_md[j]=1e-6;
	//		cout << "Variance = " << Sigma_md[j] << endl;
		}
	}
	else {
		for (j=0; j<NumClusters_mi; j++) {
			Tempd = 0.0;
			Sigma_md[j] = 0.0;
			for (i=0; i<NumDataPoints_mi; i++) {
				loc[0]= i*NumClusters_mi + j;
				DataValue = (double)i/HistogramFactor_mf + Min_mf;
				Tempd += W_md[loc[0]]*Histogram_mi[i];
				Sigma_md[j] += W_md[loc[0]]*(DataValue-Mu_md[j])*(DataValue-Mu_md[j])*Histogram_mi[i];
			}
			Sigma_md[j] /= Tempd;
			if (fabs(Sigma_md[j])<1e-6) Sigma_md[j]=1e-6;
	//		cout << "Variance = " << Sigma_md[j] << endl;
		}
	}
}


template <class _DataType>
bool cEMClustering<_DataType>::CheckConvergence(double errorrate)
{
	int		j;
	
	for (j=0; j<NumClusters_mi; j++) {
		if (fabs(PrevMu_md[j] - Mu_md[j])/Mu_md[j] < errorrate &&
			fabs(PrevSigma_md[j] - Sigma_md[j])/Sigma_md[j] < errorrate) return true;
		else return false;
	}
	return false;
}


template <class _DataType>
void cEMClustering<_DataType>::iterate()
{
	int		i;
	int		numiteration = 0;


	cout << "the number of points is: " << NumDataPoints_mi << endl;
	cout << "the number of clusters is: " << NumClusters_mi << endl;
	for (i=0; i<NumClusters_mi; i++) {
		cout << "Initial Mean = " << Mu_md[i] << " ";
		cout << "Variance = " << Sigma_md[i];
		cout << ", Std. = " << sqrt(Sigma_md[i]) << endl;
	}

	do {
		numiteration++;
		ComputeWij();
		ComputePIj();
		SaveCurrMeansVariances();
		ComputeNextMeans();
		ComputeNextVariances();
//		cout << "After find NextVariances = " << numiteration << endl;
//		if (numiteration%10==0) cout << "Num Iteration = " << numiteration << endl;

	} while (!CheckConvergence(1e-15) && numiteration < NUMITERATION_EM);
//	cout << "Error Rate" << endl;
//	PrintErrorRate();
	
	cout << "Num Iterations = " << numiteration << endl;

	for (i=0; i<NumClusters_mi; i++) {
		cout << "Final Mean = " << Mu_md[i] << " ";
		cout << "Variance = " << Sigma_md[i];
		cout << ", Std. = " << sqrt(Sigma_md[i]) << endl;
	}
}


template <class _DataType>
void cEMClustering<_DataType>::Destroy()
{
	delete [] PI_md;
	delete [] Mu_md;
	delete [] Sigma_md;
	delete [] PrevMu_md;
	delete [] PrevSigma_md;
	delete [] TotalProb_Data_md;
	delete [] W_md;
	
	PI_md = NULL;
	Mu_md = NULL;
	Sigma_md = NULL;
	PrevMu_md = NULL;
	PrevSigma_md = NULL;
	TotalProb_Data_md = NULL;
	W_md = NULL;
}


template class cEMClustering<unsigned char>;
template class cEMClustering<unsigned short>;
template class cEMClustering<int>;
template class cEMClustering<float>;

