/*
  Copyright 2002-2004 The University of Texas at Austin
  
	Authors: John Wiggins 2004 <prok@ices.utexas.edu>
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

//////////////////////////////////////////////////////////////////////
//
// VolumeSink.cpp: implementation of the VolumeSink class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/VolumeSink.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeSink::VolumeSink()
{
	VolumeSink::setDefaults();
}

VolumeSink::~VolumeSink()
{
}

void VolumeSink::setNumVariables(unsigned int num)
{
	m_NumVariables = num;
}

void VolumeSink::setNumTimeSteps(unsigned int num)
{
	m_NumTimeSteps = num;
}

void VolumeSink::setDimX(unsigned int dimX)
{
	m_DimX = dimX;
}

void VolumeSink::setDimY(unsigned int dimY)
{
	m_DimY = dimY;
}

void VolumeSink::setDimZ(unsigned int dimZ)
{
	m_DimZ = dimZ;
}

void VolumeSink::setMinX(double minX)
{
	m_MinX = minX;
}

void VolumeSink::setMinY(double minY)
{
	m_MinY = minY;
}

void VolumeSink::setMinZ(double minZ)
{
	m_MinZ = minZ;
}

void VolumeSink::setMinT(double minT)
{
	m_MinT = minT;
}

void VolumeSink::setMaxX(double maxX)
{
	m_MaxX = maxX;
}

void VolumeSink::setMaxY(double maxY)
{
	m_MaxY = maxY;
}

void VolumeSink::setMaxZ(double maxZ)
{
	m_MaxZ = maxZ;
}

void VolumeSink::setMaxT(double maxT)
{
	m_MaxT = maxT;
}

unsigned int VolumeSink::getDimX() const
{
	return m_DimX;
}

unsigned int VolumeSink::getDimY() const
{
	return m_DimY;
}

unsigned int VolumeSink::getDimZ() const
{
	return m_DimZ;
}

double VolumeSink::getMinX() const
{
	return m_MinX;
}

double VolumeSink::getMinY() const
{
	return m_MinY;
}

double VolumeSink::getMinZ() const
{
	return m_MinZ;
}

double VolumeSink::getMinT() const
{
	return m_MinT;
}

double VolumeSink::getMaxX() const
{
	return m_MaxX;
}

double VolumeSink::getMaxY() const
{
	return m_MaxY;
}

double VolumeSink::getMaxZ() const
{
	return m_MaxZ;
}

double VolumeSink::getMaxT() const
{
	return m_MaxT;
}

bool VolumeSink::error() const
{
	return m_Error;
}

bool VolumeSink::errorMustRestart() const
{
	return m_ErrorMustRestart;
}

const QString& VolumeSink::errorReason() const
{
	return m_ErrorReason;
}

void VolumeSink::resetError()
{
	if (!m_ErrorMustRestart) {
		m_Error = false;
		m_ErrorMustRestart = false;
		m_ErrorReason = QString("There is no error");
	}
	// else error reset not allowed for fatal error
}

void VolumeSink::setError(const QString& reason, bool mustRestart)
{
	m_Error = true;
	m_ErrorMustRestart = mustRestart;
	m_ErrorReason = reason;
}

void VolumeSink::setDefaults()
{
	m_MinX = 0.0; m_MinY = 0.0; m_MinZ = 0.0;
	m_MaxX = 1.0; m_MaxY = 1.0; m_MaxZ = 1.0;
	m_DimX = 0; m_DimY = 0; m_DimZ = 0;
	m_Error = false;
	m_ErrorMustRestart = false;
	m_ErrorReason = QString("There is no error");
}

