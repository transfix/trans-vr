/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// BilateralFilter.cpp: implementation of the BilateralFilter class.
//
//////////////////////////////////////////////////////////////////////

#include <Filters/BilateralFilter.h>
#include <math.h>
#include <stdlib.h>
#include <q3progressdialog.h>
#include <qdatetime.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

BilateralFilter::BilateralFilter()
{
	setDefaults();
}

BilateralFilter::~BilateralFilter()
{
	destroyBuffers();
}

bool BilateralFilter::applyFilter(QWidget* parent, unsigned char* image, 
								  unsigned int width, unsigned int height,unsigned int depth)
{
	// allocate memory for buffers
	if (!allocateBuffers(width*height*depth))
		return false;

	QTime t;
	t.start();
	Q3ProgressDialog progressDialog("Performing bilateral filtering.", 
		"Cancel", depth, parent, "Bilateral Filter", true);
	progressDialog.setProgress(0);

	//int FilterDiameter3 = FilterDiameter*FilterDiameter*FilterDiameter;
	//int FilterDiameter2 = FilterDiameter*FilterDiameter;
	int offsetTable[FilterDiameter*FilterDiameter*FilterDiameter];
	buildOffsetTable(offsetTable, width, height);
	double spatialMask[FilterDiameter*FilterDiameter*FilterDiameter];
	computeSpatialMask(spatialMask);
	double radiometricTable[256];

	//double radiometricMask[FilterDiameter*FilterDiameter*FilterDiameter];

	unsigned int c, currentBuffer=0, sliceSize=width*height;
	// get the first FilterRadius+1 slices into the buffers
	for (c=0; c<FilterRadius+1; c++) {
		if (c<depth) {
			bringInSlice(m_Buffers[currentBuffer+FilterRadius+c], &(image[sliceSize*c]), sliceSize);
		}
	}
	for (c=0; c<256; c++) {
		radiometricTable[c] = radiometricFactor(c);
	}


	int k,j,i;
	int x,y,z;
	double sum, denom;
	int sample;
	double weight;//,weight2;
	int w=(int)width, h=(int)height, d=(int)depth;
	bool bool1, bool2;
	//int index1,index2;
	// for each slice
	for (k=0; k<d; k++) {
		progressDialog.setProgress(k);
		// for each row
		for (j=0; j<h; j++) {
			// for each voxel
			for (i=0; i<w; i++) {

				sample = m_Buffers[(currentBuffer+FilterRadius)%FilterDiameter][w*(j)+(i)];
				sum = 0; denom = 0;

				
				for (z=0; z<FilterDiameter; z++) {
					//if (k+z>=FilterRadius && k+z<d+FilterRadius) {
					bool1 = k+z>=FilterRadius && k+z<d+FilterRadius;
						for (y=0; y<FilterDiameter; y++) {
							//if (j+y>=FilterRadius && j+y<h+FilterRadius) {
							bool2 = bool1 && (j+y>=FilterRadius && j+y<h+FilterRadius);
								for (x=0; x<FilterDiameter; x++) {
									if (i+x>=FilterRadius && i+x<w+FilterRadius  &&
										bool2) {
										//j+y>=FilterRadius && k+y<h+FilterRadius &&
										//k+z>=FilterRadius && k+z<d+FilterRadius
										//) {
										//index1 = abs(sample-m_Buffers[(currentBuffer+z)%FilterDiameter][w*(j+y-FilterRadius)+(i+x-FilterRadius)]);
										//index2 = m_Buffers[(currentBuffer+z)%FilterDiameter][w*(j+y-FilterRadius)+(i+x-FilterRadius)];
										weight = radiometricTable[abs(sample-m_Buffers[(currentBuffer+z)%FilterDiameter][w*(j+y-FilterRadius)+(i+x-FilterRadius)])]*
											spatialMask[z*FilterDiameter*FilterDiameter+y*FilterDiameter+x];
										//weight = radiometricFactor(sample-m_Buffers[(currentBuffer+z)%FilterDiameter][w*(j+y-FilterRadius)+(i+x-FilterRadius)])*
										//	spatialMask[z*FilterDiameter*FilterDiameter+y*FilterDiameter+x];
										//if (weight!=weight2) {
										//	weight = weight2;
										//}
										denom+=weight;
										sum+=weight*(double)m_Buffers[(currentBuffer+z)%FilterDiameter][w*(j+y-FilterRadius)+(i+x-FilterRadius)];
									}
								}
							//}
						}
					//}
				}
				
				/*
				for (c=0; c<FilterDiameter3; c++) {
					weight = radiometricFactor(sample-m_Buffers[(currentBuffer+tableSize/FilterDiameter2)%FilterDiameter][w*(j+y-FilterRadius)+(i+x-FilterRadius)])*
						spatialMask[z*FilterDiameter*FilterDiameter+y*FilterDiameter+x];
					denom+=weight;
					sum+=weight*m_Buffers[(currentBuffer+z)%FilterDiameter][w*(j+y-FilterRadius)+(i+x-FilterRadius)];
					
				}
				*/



				image[sliceSize*k+w*j+i] = (unsigned char)(sum/denom);

			}
		}



		// pull the next slice into the buffer
		if (k+FilterRadius+1<d) {
			bringInSlice(m_Buffers[currentBuffer], 
				&(image[(k+FilterRadius+1)*sliceSize]),
				sliceSize);
		}
		currentBuffer = (currentBuffer+1)%FilterDiameter;
	}
	qDebug("Time to filter: %d", t.elapsed());

	return true;
}


void BilateralFilter::setDefaults()
{
	m_MemoryAllocated = 0;
	unsigned int c;
	for (c=0; c<FilterDiameter; c++) {
		m_Buffers[c] = 0;
	}

	//m_SpatialSigma = 1.5;
	//m_RadiometricSigma = 20;
	m_SpatialSigma = 1.5;
	m_RadiometricSigma = 200;
}

void BilateralFilter::computeSpatialMask(double* mask) const
{
	int i,j,k;
	unsigned int index = 0;
	for (k=-FilterRadius; k<=FilterRadius; k++) {
		for (j=-FilterRadius; j<=FilterRadius; j++) {
			for (i=-FilterRadius; i<=FilterRadius; i++) {
				mask[index++] = exp((double)(k*k+j*j+i*i)/(-m_SpatialSigma*m_SpatialSigma*2.0));
			}
		}
	}
}

void BilateralFilter::bringInSlice(int* dest, unsigned char* source, unsigned int amount)
{
	unsigned int c;
	for (c=0; c<amount; c++) {
		dest[c] = source[c];
	}
}

void BilateralFilter::buildOffsetTable(int* offsetTable, int width, int height) const
{
	int i,j,k;	
	int c = 0;

	for (k=0; k<FilterDiameter; k++) {
		for (j=0; j<FilterDiameter; j++) {
			for (i=0; i<FilterDiameter; i++) {
				offsetTable[c] = (k-FilterRadius)*width*height + (j-FilterRadius)*width + (i-FilterRadius);
			}
		}
	}
		
}

double BilateralFilter::radiometricFactor(double dist) const
{
	return exp((double)(dist*dist)/
		(-m_RadiometricSigma*m_RadiometricSigma*2.0));
}

bool BilateralFilter::allocateBuffers(unsigned int amount)
{
	if (amount>m_MemoryAllocated) {
		destroyBuffers();
		return forceAllocateBuffers(amount);
	}
	else {
		return true;
	}
}

bool BilateralFilter::forceAllocateBuffers(unsigned int amount)
{
	unsigned int c;
	for (c=0; c<FilterDiameter; c++) {
		m_Buffers[c] = new int[amount];
		if (!m_Buffers[c]) {
			destroyBuffers();
			return false;
		}
	}
	return true;
}

void BilateralFilter::destroyBuffers()
{
	unsigned int c;
	for (c=0; c<FilterDiameter; c++) {
		delete [] m_Buffers[c];
		m_Buffers[c] = 0;
	}
}

