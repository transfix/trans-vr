/*
  Copyright 2002-2005 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
           John Wiggins <prok@ices.utexas.edu>
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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// OOCBilateralFilter.cpp: implementation of the OOCBilateralFilter class.
//
//////////////////////////////////////////////////////////////////////

#include <Filters/OOCBilateralFilter.h>
#include <math.h>
#include <stdlib.h>
#include <qprogressdialog.h>
#include <qdatetime.h>
#include <stdio.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

OOCBilateralFilter::OOCBilateralFilter()
{
	setDefaults();
}

OOCBilateralFilter::~OOCBilateralFilter()
{
	destroyBuffers();
	delete [] m_SpatialMask;
	delete [] m_Buffers;
}

void OOCBilateralFilter::setDataType(OutOfCoreFilter::Type t)
{
	m_Type = t;

	if (m_MemoryAllocated > 0)
		destroyBuffers();
}
	
int OOCBilateralFilter::getNumCacheSlices() const
{
	return m_FilterRadius+1;
}

bool OOCBilateralFilter::initFilter(unsigned int width, unsigned int height, unsigned int depth, double functionMin, double functionMax)
{
	m_Width = width;
	m_Height = height;
	m_Depth = depth;

	m_FuncMin = functionMin;
	m_FuncMax = functionMax;
	
	m_CacheSlice=0;
	m_CurrentSlice=0;

	// compute the new radiometric sigma
	//m_RadiometricSigma = 0.8 * (m_FuncMax - m_FuncMin);
	
	// build the radiometric table
	for (int c=0; c<256; c++) {
		double factor = c * ((m_FuncMax - m_FuncMin) / 255.0);
		//m_RadiometricTable[c] = radiometricFactor((double)c);
		m_RadiometricTable[c] = radiometricFactor(factor);
	}

//	printf("Radiometric Table: {");
//	for (int c=0; c < 256; c++)
//		printf("%f, ", m_RadiometricTable[c]);
//	printf("}\n");
//	fflush(stdout);
	
	if (m_MemoryAllocated > 0)
		return true;
	else
		return allocateBuffers(m_Width*m_Height);
}

bool OOCBilateralFilter::initFilter(unsigned int width, unsigned int height, unsigned int depth, double functionMin, double functionMax, double RadiometricSigma, double SpatialSigma, int FilterRadius)
{
  m_RadiometricSigma = RadiometricSigma;
  m_SpatialSigma = SpatialSigma;
  m_FilterRadius = FilterRadius;
  m_FilterDiameter = m_FilterRadius*2+1;
  if(m_Buffers) delete [] m_Buffers;
  m_Buffers = new void*[m_FilterDiameter];
  memset(m_Buffers,0,sizeof(void*)*m_FilterDiameter);
  return initFilter(width,height,depth,functionMin,functionMax);
}

void OOCBilateralFilter::addSlice(void *slice)
{
	bringInSlice(m_Buffers[(m_CacheSlice+m_FilterRadius)%m_FilterDiameter], slice, m_Width*m_Height);

	m_CacheSlice++;
}

bool OOCBilateralFilter::filterSlice(void *slice)
{
	int k,j,i;
	int x,y,z;
	double sum, denom;
	int isample;
	double fsample;
	double weight, normalizedDiff;
	int w=(int)m_Width, h=(int)m_Height, d=(int)m_Depth;
	bool bool1, bool2;

	// for each slice
	k = m_CurrentSlice;

	if (m_Type == OutOfCoreFilter::U_CHAR
			|| m_Type == OutOfCoreFilter::U_SHORT
			|| m_Type == OutOfCoreFilter::U_INT) {
	for (j=0; j<h; j++) {
		// for each voxel
		for (i=0; i<w; i++) {

			isample = ((int*)m_Buffers[(m_CurrentSlice+m_FilterRadius)%m_FilterDiameter])[w*(j)+(i)];
			sum = 0; denom = 0;

			for (z=0; z<m_FilterDiameter; z++) {
				bool1 = k+z>=m_FilterRadius && k+z<d+m_FilterRadius;
				for (y=0; y<m_FilterDiameter; y++) {
					bool2 = bool1 && (j+y>=m_FilterRadius && j+y<h+m_FilterRadius);
					for (x=0; x<m_FilterDiameter; x++) {
						if (i+x>=m_FilterRadius && i+x<w+m_FilterRadius && bool2) {
							normalizedDiff = (double)abs(isample-((int*)m_Buffers[(m_CurrentSlice+z)%m_FilterDiameter])[w*(j+y-m_FilterRadius)+(i+x-m_FilterRadius)]);
							normalizedDiff /= (m_FuncMax-m_FuncMin);
							normalizedDiff *= 255.0;
							//weight = m_RadiometricTable[abs(isample-((int*)m_Buffers)[(m_CurrentSlice+z)%m_FilterDiameter][w*(j+y-m_FilterRadius)+(i+x-m_FilterRadius)])]*
							weight = m_RadiometricTable[(int)normalizedDiff]*
								m_SpatialMask[z*m_FilterDiameter*m_FilterDiameter+y*m_FilterDiameter+x];
							denom+=weight;
							sum+=weight*(double)((int*)m_Buffers[(m_CurrentSlice+z)%m_FilterDiameter])[w*(j+y-m_FilterRadius)+(i+x-m_FilterRadius)];
						}
					}
				}
			}

			//double val = sum/denom;
			//if (isnan(val) || isinf(val)) {
			//	printf("sum = %f , denom = %f\n", sum, denom); fflush(stdout);
			//}
				
			switch (m_Type) {
				case OutOfCoreFilter::U_CHAR:
					((unsigned char *)slice)[w*j+i] = (unsigned char)(sum/denom);
					break;
				case OutOfCoreFilter::U_SHORT:
					((unsigned short *)slice)[w*j+i] = (unsigned short)(sum/denom);
					break;
				case OutOfCoreFilter::U_INT:
					((unsigned int *)slice)[w*j+i] = (unsigned int)(sum/denom);
					break;
				default:
					break;
			}
		}
	}
	}
	else { // m_Type == FLOAT or DOUBLE
	for (j=0; j<h; j++) {
		// for each voxel
		for (i=0; i<w; i++) {

			fsample = ((double*)m_Buffers[(m_CurrentSlice+m_FilterRadius)%m_FilterDiameter])[w*(j)+(i)];
			sum = 0; denom = 0;

			for (z=0; z<m_FilterDiameter; z++) {
				bool1 = k+z>=m_FilterRadius && k+z<d+m_FilterRadius;
				for (y=0; y<m_FilterDiameter; y++) {
					bool2 = bool1 && (j+y>=m_FilterRadius && j+y<h+m_FilterRadius);
					for (x=0; x<m_FilterDiameter; x++) {
						if (i+x>=m_FilterRadius && i+x<w+m_FilterRadius && bool2) {
							normalizedDiff = fabs(fsample-((double*)m_Buffers[(m_CurrentSlice+z)%m_FilterDiameter])[w*(j+y-m_FilterRadius)+(i+x-m_FilterRadius)]);
							normalizedDiff /= (m_FuncMax-m_FuncMin);
							normalizedDiff *= 255.0;
							//weight = m_RadiometricTable[(int)fabs(fsample-((double*)m_Buffers)[(m_CurrentSlice+z)%m_FilterDiameter][w*(j+y-m_FilterRadius)+(i+x-m_FilterRadius)])]*
							weight = m_RadiometricTable[(int)normalizedDiff]*
								m_SpatialMask[z*m_FilterDiameter*m_FilterDiameter+y*m_FilterDiameter+x];
							denom+=weight;
							sum+=weight*((double*)m_Buffers[(m_CurrentSlice+z)%m_FilterDiameter])[w*(j+y-m_FilterRadius)+(i+x-m_FilterRadius)];
						}
					}
				}
			}


			switch (m_Type) {
				case OutOfCoreFilter::FLOAT:
					((float *)slice)[w*j+i] = (float)(sum/denom);
					break;
				case OutOfCoreFilter::DOUBLE:
					((double *)slice)[w*j+i] = (double)(sum/denom);
					break;
				default:
					break;
			}
		}
	}
	}

	m_CurrentSlice++;

	return true;
}

void OOCBilateralFilter::setDefaults()
{
	m_MemoryAllocated = 0;
	//unsigned int c;
	//for (c=0; c<m_FilterDiameter; c++) {
	//m_Buffers[c] = 0;
	//}

	//m_SpatialSigma = 1.5;
	//m_RadiometricSigma = 20;
	m_SpatialSigma = 1.5;
	m_RadiometricSigma = 200;
	m_FilterRadius = 2;
	m_FilterDiameter = m_FilterRadius*2+1;

	m_Buffers = new void*[m_FilterDiameter];
    memset(m_Buffers,0,sizeof(void*)*m_FilterDiameter);
	/*for(c=0; c<m_FilterDiameter; c++){
	  m_Buffers[c] = 0;
	}*/

	m_FuncMin = 0.0;
	m_FuncMax = 255.0;

	m_CacheSlice = 0;
	m_CurrentSlice = 0;
	m_Width = 0;
	m_Height = 0;
	m_Depth = 0;

	m_Type = OutOfCoreFilter::U_CHAR;
	
	// build the spatial mask
	m_SpatialMask = new double [m_FilterDiameter*m_FilterDiameter*m_FilterDiameter];
	computeSpatialMask(m_SpatialMask);

}

void OOCBilateralFilter::computeSpatialMask(double* mask) const
{
	int i,j,k;
	unsigned int index = 0;
	for (k=-m_FilterRadius; k<=m_FilterRadius; k++) {
		for (j=-m_FilterRadius; j<=m_FilterRadius; j++) {
			for (i=-m_FilterRadius; i<=m_FilterRadius; i++) {
				mask[index++] = exp((double)(k*k+j*j+i*i)/(-m_SpatialSigma*m_SpatialSigma*2.0));
			}
		}
	}
}

void OOCBilateralFilter::bringInSlice(void* dest, void* source, unsigned int amount)
{
	unsigned int c;
	for (c=0; c<amount; c++) {
		switch (m_Type) {
			case OutOfCoreFilter::U_CHAR:
				((int *)dest)[c] = ((unsigned char *)source)[c];
				break;
			case OutOfCoreFilter::U_SHORT:
				((int *)dest)[c] = ((unsigned short *)source)[c];
				break;
			case OutOfCoreFilter::U_INT:
				((int *)dest)[c] = ((unsigned int *)source)[c];
				break;
			case OutOfCoreFilter::FLOAT:
				((double *)dest)[c] = ((float *)source)[c];
				break;
			case OutOfCoreFilter::DOUBLE:
				((double *)dest)[c] = ((double *)source)[c];
				break;
			default:
				break;
		}
	}
}

double OOCBilateralFilter::radiometricFactor(double dist) const
{
	return exp((double)(dist*dist)/
		(-m_RadiometricSigma*m_RadiometricSigma*2.0));
}

bool OOCBilateralFilter::allocateBuffers(unsigned int amount)
{
	if (amount>m_MemoryAllocated) {
		destroyBuffers();
		return forceAllocateBuffers(amount);
	}
	else {
		return true;
	}
}

bool OOCBilateralFilter::forceAllocateBuffers(unsigned int amount)
{
	unsigned int c;

	if (m_Type == U_CHAR || m_Type == U_SHORT || m_Type == U_INT) {
		for (c=0; c<m_FilterDiameter; c++) {
			m_Buffers[c] = (void *)new int[amount];
			if (!m_Buffers[c]) {
				destroyBuffers();
				return false;
			}
		}
	}
	else { // m_Type == FLOAT || m_Type == DOUBLE
		for (c=0; c<m_FilterDiameter; c++) {
			m_Buffers[c] = (void *)new double[amount];
			if (!m_Buffers[c]) {
				destroyBuffers();
				return false;
			}
		}
	}

	m_MemoryAllocated = amount;

	return true;
}

void OOCBilateralFilter::destroyBuffers()
{
	unsigned int c;
	for (c=0; c<m_FilterDiameter; c++) {
		if(m_Buffers[c]) delete [] m_Buffers[c];
		m_Buffers[c] = 0;
	}

	m_MemoryAllocated = 0;
}

