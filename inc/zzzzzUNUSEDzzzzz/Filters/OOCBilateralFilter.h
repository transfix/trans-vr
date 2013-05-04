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

// OOCBilateralFilter.h: interface for the OOCBilateralFilter class.
//
//////////////////////////////////////////////////////////////////////

#ifndef OOC_BILATERAL_FILTER_H
#define OOC_BILATERAL_FILTER_H

#include <Filters/OOCFilter.h>

//const int FilterRadius = 2;
//const int FilterDiameter = FilterRadius*2+1;

///\class OOCBilateralFilter OOCBilateralFilter.h
///\author John Wiggins
///\author Anthony Thane
///\brief OOCBilateralFilter is the same code as BilateralFilter, but rewritten to use the
/// OutOfCoreFilter interface.
class OOCBilateralFilter : public OutOfCoreFilter
{
public:
	OOCBilateralFilter();
	virtual ~OOCBilateralFilter();

	virtual void setDataType(OutOfCoreFilter::Type t);
	
	virtual int getNumCacheSlices() const;

	virtual bool initFilter(unsigned int width, unsigned int height, unsigned int depth, double functionMin, double functionMax);
	virtual bool initFilter(unsigned int width, unsigned int height, unsigned int depth, double functionMin, double functionMax, 
				double RadiometricSigma, double SpacialSigma, int FilterRadius);
	virtual void addSlice(void *slice);
	virtual bool filterSlice(void *slice);

protected:
	void setDefaults();

	void bringInSlice(void* dest, void* source, unsigned int amount);

	void computeSpatialMask(double* mask) const;
	double radiometricFactor(double dist) const;

	bool allocateBuffers(unsigned int amount);
	bool forceAllocateBuffers(unsigned int amount);
	void destroyBuffers();
	
	void **m_Buffers;
	unsigned int m_MemoryAllocated;
	unsigned int m_Width;
	unsigned int m_Height;
	unsigned int m_Depth;
	unsigned int m_CacheSlice;
	unsigned int m_CurrentSlice;
	OutOfCoreFilter::Type m_Type;

	double *m_SpatialMask;
	double m_RadiometricTable[256];

	double m_SpatialSigma;
	double m_RadiometricSigma;

	double m_FuncMin;
	double m_FuncMax;

	int m_FilterRadius;
	int m_FilterDiameter;
};

#endif

