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

// RawIVSimpleSource.h: interface for the RawIVSimpleSource class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAWIVSIMPLESOURCE_H__65AEA466_9DCC_4980_A2C0_0033D3736E4F__INCLUDED_)
#define AFX_RAWIVSIMPLESOURCE_H__65AEA466_9DCC_4980_A2C0_0033D3736E4F__INCLUDED_

#include <VolumeFileTypes/VolumeSource.h>
#include <qstring.h>
#include <stdio.h>

///\class RawIVSimpleSource RawIVSimpleSource.h
///\author Anthony Thane
///\deprecated This class has been superceded by VolumeFileSource.
class RawIVSimpleSource : public VolumeSource  
{
public:
	RawIVSimpleSource();
	virtual ~RawIVSimpleSource();

	bool setFile(const QString& file);

	virtual void fillData(char* data, double xMin, double yMin, double zMin,
		double xMax, double yMax, double zMax,
		unsigned int xDim, unsigned int yDim, unsigned int zDim);

	virtual void fillThumbnail(char* data, 
		unsigned int xDim, unsigned int yDim, unsigned int zDim);

	virtual void fillData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim);

protected:
	virtual void setDefaults();
	bool readHeader(FILE* fp);
	bool allocateMemory(unsigned int num);
	void destroyMemory();

	unsigned char* m_Data;
};

#endif // !defined(AFX_RAWIVSIMPLESOURCE_H__65AEA466_9DCC_4980_A2C0_0033D3736E4F__INCLUDED_)
