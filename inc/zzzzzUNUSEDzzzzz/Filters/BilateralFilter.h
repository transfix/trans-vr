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

// BilateralFilter.h: interface for the BilateralFilter class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_BILATERALFILTER_H__A034E74C_77EC_470D_9CC4_D2D69C8C7701__INCLUDED_)
#define AFX_BILATERALFILTER_H__A034E74C_77EC_470D_9CC4_D2D69C8C7701__INCLUDED_

#include <Filters/Filter.h>

const int FilterRadius = 2;
const int FilterDiameter = FilterRadius*2+1;

///\class BilateralFilter BilateralFilter.h
///\author Anthony Thane
///\author John Wiggins
///\brief The BilateralFilter class is an implementation of the bilateral filtering algorithm using
/// the Filter interface.
class BilateralFilter : public Filter
{
public:
	BilateralFilter();
	virtual ~BilateralFilter();

	virtual bool applyFilter(QWidget* parent, unsigned char* image, unsigned int width, unsigned int height,
		unsigned int depth);

protected:
	void setDefaults();

	void computeSpatialMask(double* mask) const;
	void bringInSlice(int* dest, unsigned char* source, unsigned int amount);
	void buildOffsetTable(int* offsetTable, int width, int height) const;
	double radiometricFactor(double dist) const;

	bool allocateBuffers(unsigned int amount);
	bool forceAllocateBuffers(unsigned int amount);
	void destroyBuffers();
	int* m_Buffers[FilterDiameter];
	unsigned int m_MemoryAllocated;

	double m_SpatialSigma;
	double m_RadiometricSigma;
};

#endif // !defined(AFX_BILATERALFILTER_H__A034E74C_77EC_470D_9CC4_D2D69C8C7701__INCLUDED_)
