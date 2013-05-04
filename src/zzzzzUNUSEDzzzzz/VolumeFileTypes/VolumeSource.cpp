/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane 2002-2003 <thanea@ices.utexas.edu>
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

//////////////////////////////////////////////////////////////////////
//
// VolumeSource.cpp: implementation of the VolumeSource class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/VolumeSource.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeSource::VolumeSource()
{
	VolumeSource::setDefaults();
}

VolumeSource::~VolumeSource()
{

}

static inline double toDouble(uint val, double min, double max, double dim)
{
	return min + (double)val*(max-min)/(double)(dim-1);
}

bool VolumeSource::fillGradientData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep)
{
	// may not be implemented by the derived class
	return false;
}

bool VolumeSource::readRawData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, uint variable, uint timeStep)
{
	// if this base class version is called, then it's obviously not implemented
	// by the derived class.
	return false;
}

QString VolumeSource::getVariableName(unsigned int variable) const
{
	return "No Name";
}

QString VolumeSource::getContourSpectrumFileName(unsigned int variable, unsigned int timeStep)
{
	return QString();
}

QString VolumeSource::getContourTreeFileName(unsigned int variable, unsigned int timeStep)
{
	return QString();
}

void VolumeSource::computeContourSpectrum(QObject *obj, unsigned int variable, unsigned int timeStep)
{
	// do nothing
	qDebug("VolumeSource::computeContourSpectrum(...) not implemented");
}

void VolumeSource::computeContourTree(QObject *obj, unsigned int variable, unsigned int timeStep)
{
	// do nothing
	qDebug("VolumeSource::computeContourTree(...) not implemented");
}

void VolumeSource::cleanUpWorkerThread(int thid)
{
	// nothing
}

unsigned int VolumeSource::getNumVars() const
{
	return m_NumVariables;
}

unsigned int VolumeSource::getNumTimeSteps() const
{
	return m_NumTimeSteps;
}

double VolumeSource::getFunctionMinimum(unsigned int variable, unsigned int timeStep) const
{
	return 0.0;
}

double VolumeSource::getFunctionMaximum(unsigned int variable, unsigned int timeStep) const
{
	return 1.0;
}

//unsigned int VolumeSource::getNumVerts() const
Q_ULLONG VolumeSource::getNumVerts() const
{
	return ((Q_ULLONG)m_DimX)*m_DimY*m_DimZ;
}

//unsigned int VolumeSource::getNumCells() const
Q_ULLONG VolumeSource::getNumCells() const
{
	return ((Q_ULLONG)(m_DimX-1))*(m_DimY-1)*(m_DimZ-1);
}

double VolumeSource::getMinX() const
{
	return m_MinX;
}

double VolumeSource::getMinY() const
{
	return m_MinY;
}

double VolumeSource::getMinZ() const
{
	return m_MinZ;
}

double VolumeSource::getMinT() const
{
	return m_MinT;
}

double VolumeSource::getMaxX() const
{
	return m_MaxX;
}

double VolumeSource::getMaxY() const
{
	return m_MaxY;
}

double VolumeSource::getMaxZ() const
{
	return m_MaxZ;
}

double VolumeSource::getMaxT() const
{
	return m_MaxT;
}

unsigned int VolumeSource::getDimX() const
{
	return m_DimX;
}

unsigned int VolumeSource::getDimY() const
{
	return m_DimY;
}

unsigned int VolumeSource::getDimZ() const
{
	return m_DimZ;
}

void VolumeSource::setMinX(double minX)
{
	m_MinX = minX;
}

void VolumeSource::setMinY(double minY)
{
	m_MinY = minY;
}

void VolumeSource::setMinZ(double minZ)
{
	m_MinZ = minZ;
}

void VolumeSource::setMinT(double minT)
{
	m_MinT = minT;
}

void VolumeSource::setMaxX(double maxX)
{
	m_MaxX = maxX;
}

void VolumeSource::setMaxY(double maxY)
{
	m_MaxY = maxY;
}

void VolumeSource::setMaxZ(double maxZ)
{
	m_MaxZ = maxZ;
}

void VolumeSource::setMaxT(double maxT)
{
	m_MaxT = maxT;
}

void VolumeSource::setDimX(unsigned int dimX)
{
	m_DimX = dimX;
}

void VolumeSource::setDimY(unsigned int dimY)
{
	m_DimY = dimY;
}

void VolumeSource::setDimZ(unsigned int dimZ)
{
	m_DimZ = dimZ;
}

bool VolumeSource::error() const
{
	return m_Error;
}

bool VolumeSource::errorMustRestart() const
{
	return m_ErrorMustRestart;
}

const QString& VolumeSource::errorReason() const
{
	return m_ErrorReason;
}

VolumeSource::DownLoadFrequency VolumeSource::interactiveUpdateHint() const
{
	return DLFDelayed;
}

void VolumeSource::resetError()
{
	if (!m_ErrorMustRestart) {
		m_Error = false;
		m_ErrorMustRestart = false;
		m_ErrorReason = QString("There is no error");
	}
	// else error reset not allowed for fatal error
}

void VolumeSource::setError(const QString& reason, bool mustRestart)
{
	m_Error = true;
	m_ErrorMustRestart = mustRestart;
	m_ErrorReason = reason;
}

void VolumeSource::setDefaults()
{
	m_MinX = 0.0; m_MinY = 0.0; m_MinZ = 0.0; m_MinT = 0.0;
	m_MaxX = 1.0; m_MaxY = 1.0; m_MaxZ = 1.0; m_MaxT = 1.0;
	m_DimX = 0; m_DimY = 0; m_DimZ = 0;
	m_Error = false;
	m_ErrorMustRestart = false;
	m_ErrorReason = QString("There is no error");
}

