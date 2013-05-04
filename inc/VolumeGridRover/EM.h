/*
  Copyright 2006-2008 The University of Texas at Austin

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

#ifndef __EM_H__
#define __EM_H__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define NUMITERATION_EM 100
#define TRUE	1
#define FALSE	0

#define SVMM		// Spatially Variant Mixture Model

namespace EM
{
  static const double pi = 3.141592;
  static inline double GaussianDensity(double x, double mean, double var)
  {
    //double y;
    //y = ((double)1.0/sqrt((double)2.0*pi*var))*exp(-1.0*(x-mean)*(x-mean)/(2*var));
    //return y;
    return ((double)1.0/sqrt((double)2.0*pi*var))*exp(-1.0*(x-mean)*(x-mean)/(2*var));
  }

  template <class _DataType> 
    class cEMClustering
    {
    private:
      int 		NumDataPoints_mi;	// the number of data points(the number of pixels)
      int 		NumClusters_mi;	// the number of Cluster objects
      float		Min_mf, Max_mf;
      int			*Histogram_mi;
      float		HistogramFactor_mf;
      int			SumFrequencies_mi;
      _DataType	*Data_m;

      double	*PI_md;					// Num PI_md 	= Num Clusters
      double	*Mu_md;					// Num Mu_md 	= Num Clusters
      double	*Sigma_md;				// Num Sigma_md = Num Clusters
      double	*PrevMu_md;				// Num Mu_md 	= Num Clusters
      double	*PrevSigma_md;			// Num Sigma_md = Num Clusters
      double	*TotalProb_Data_md;		// Num TotalProb_Data_md = Num Data 
      double	*W_md;					// Num W_md = Num Clusters * Num Data;
      
    public:
      /* constructors */
      cEMClustering()  /* in case of one cluster */
      {
	PI_md = NULL;
	Mu_md = NULL;
	Sigma_md = NULL;
	PrevMu_md = NULL;
	PrevSigma_md = NULL;
	TotalProb_Data_md = NULL;
	W_md = NULL;
      }
      cEMClustering(int NumData, int NumClusters) /* in case of number of pixels and multiple clusters */
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
      ~cEMClustering()
      {
	delete [] PI_md;
	delete [] Mu_md;
	delete [] Sigma_md;
	delete [] PrevMu_md;
	delete [] PrevSigma_md;
	delete [] TotalProb_Data_md;
	delete [] W_md;
      }	

      void InitializeEM(int NumClusters)
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

      void setData(_DataType *Data, float Minf, float Maxf)
      {
	Min_mf = Minf;
	Max_mf = Maxf;
	Data_m = Data;
      }

      void setHistogram(int* Histo, float HistoF)
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
	//std::cout << "Sum Frequencies = " << SumFrequencies_mi << std::endl;
      }

      void setMeanVariance(int ith, double Mean,double variance)
      {
	PrevMu_md[ith]		= Mu_md[ith]	= Mean;
	PrevSigma_md[ith]	= Sigma_md[ith] = variance;
	//std::cout << "Mean & Variance = " << Mu_md[ith] << " " << Sigma_md[ith] << std::endl;
      }
		
      float *getProbability()
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
		
      void ComputeWij()
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
	      std::cout << "Data, Mean & Variance = " << DataValue << "  ";
	      std::cout << Mu_md[j] << "  " << Sigma_md[j] << std::endl;
	      std::cout << "Gaussian = " << GaussianDensity(DataValue, Mu_md[j], Sigma_md[j]) << std::endl;
	      std::cout << "Total Prob Data = " << TotalProb_Data_md[i] << std::endl;
	      std::cout << "Wij = " << W_md[loc[0]] << std::endl;
	    */
	  }
	}
      }

      void ComputePIj()
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
	    //		std::cout << "PI_md[j] = " << PI_md[j] << std::endl;
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
	    //		std::cout << "PI_md[j] = " << PI_md[j] << std::endl;
	    Tempd += PI_md[j];
	  }
	}
	//	std::cout << "Sum of PI_md = " << Tempd << std::endl;
      }

      void SaveCurrMeansVariances()
      {
	int		j;
	for (j=0; j<NumClusters_mi; j++) {
	  PrevMu_md[j] = Mu_md[j];
	  PrevSigma_md[j] = Sigma_md[j];
	}
      }

      void ComputeNextMeans()
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
	    //		std::cout << "Mean = " << Mu_md[j] << std::endl;
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
	    //		std::cout << "Mean = " << Mu_md[j] << std::endl;
	  }
	}
      }

      void ComputeNextVariances()
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
	    //		std::cout << "Variance = " << Sigma_md[j] << std::endl;
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
	    //		std::cout << "Variance = " << Sigma_md[j] << std::endl;
	  }
	}
      }

      bool CheckConvergence(double errorrate)
      {
	int		j;
	
	for (j=0; j<NumClusters_mi; j++) {
	  if (fabs(PrevMu_md[j] - Mu_md[j])/Mu_md[j] < errorrate &&
	      fabs(PrevSigma_md[j] - Sigma_md[j])/Sigma_md[j] < errorrate) return true;
	  else return false;
	}
	return false;
      }

      void iterate()
      {
	int		i;
	int		numiteration = 0;
	
	std::cout << "the number of points is: " << NumDataPoints_mi << std::endl;
	std::cout << "the number of clusters is: " << NumClusters_mi << std::endl;
	for (i=0; i<NumClusters_mi; i++) {
	  std::cout << "Initial Mean = " << Mu_md[i] << " ";
	  std::cout << "Variance = " << Sigma_md[i];
	  std::cout << ", Std. = " << sqrt(Sigma_md[i]) << std::endl;
	}
	
	do {
	  numiteration++;
	  ComputeWij();
	  ComputePIj();
	  SaveCurrMeansVariances();
	  ComputeNextMeans();
	  ComputeNextVariances();
	  //		std::cout << "After find NextVariances = " << numiteration << std::endl;
	  //		if (numiteration%10==0) std::cout << "Num Iteration = " << numiteration << std::endl;
	  
	} while (!CheckConvergence(1e-15) && numiteration < NUMITERATION_EM);
	//	std::cout << "Error Rate" << std::endl;
	//	PrintErrorRate();
	
	std::cout << "Num Iterations = " << numiteration << std::endl;
	
	for (i=0; i<NumClusters_mi; i++) {
	  std::cout << "Final Mean = " << Mu_md[i] << " ";
	  std::cout << "Variance = " << Sigma_md[i];
	  std::cout << ", Std. = " << sqrt(Sigma_md[i]) << std::endl;
	}
      }

      void Destroy()
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
    };
}

#endif
