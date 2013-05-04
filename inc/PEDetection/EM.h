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

#ifndef FILE_EM_H
#define FILE_EM_H

#define NUMITERATION_EM 100


template <class _DataType> 
class cEMClustering {
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
		cEMClustering();			/* in case of one cluster */
		cEMClustering(int, int);	/* in case of number of pixels and multiple clusters */
		/* destructor */
		~cEMClustering();		

		void InitializeEM(int NumClusters);
		void setData(_DataType *Data, float Minf, float Maxf);
		void setHistogram(int* Histo, float HistoF);
		void setMeanVariance(int ith, double Mean,double variance);
		
		float *getProbability();
		
		void ComputeWij();
		void ComputePIj();
		void SaveCurrMeansVariances();
		void ComputeNextMeans();
		void ComputeNextVariances();
		bool CheckConvergence(double errorrate);
		void iterate();
		void Destroy();

};

#endif

