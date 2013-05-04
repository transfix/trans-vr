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

// BasicVolumeFileImpl.cpp: implementation of the BasicVolumeFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/BasicVolumeFileImpl.h>
#include <qstring.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

BasicVolumeFileImpl::BasicVolumeFileImpl()
{
	initDefaults();
}

BasicVolumeFileImpl::~BasicVolumeFileImpl()
{
	destroyArrays();
}

//! Returns the number of variables stored in the file
unsigned int BasicVolumeFileImpl::getNumVariables() const
{
	return m_NumVariables;
}

//! Returns the type for the given variable v
BasicVolumeFileImpl::Type BasicVolumeFileImpl::getVariableType(unsigned int v) const
{
	if (!m_VariableTypes) {
		return Char;
	}
	else if (v<m_NumVariables) {
		return m_VariableTypes[v];
	}
	else {
		return m_VariableTypes[0];
	}
}

//! Returns the name of the given variable v
QString BasicVolumeFileImpl::getVariableName(unsigned int v) const
{
	if (!m_VariableNames) {
		return "";
	}
	else if (v<m_NumVariables) {
		return m_VariableNames[v];
	}
	else {
		return m_VariableNames[0];
	}
}

//! Returns the number of vertices in the X dimension.
unsigned int BasicVolumeFileImpl::getDimX() const
{
	return m_DimX;
}

//! Returns the number of vertices in the Y dimension.
unsigned int BasicVolumeFileImpl::getDimY() const
{
	return m_DimY;
}

//! Returns the number of vertices in the Z dimension.
unsigned int BasicVolumeFileImpl::getDimZ() const
{
	return m_DimZ;
}

//! Returns the number of time steps.
unsigned int BasicVolumeFileImpl::getNumTimeSteps() const
{
	return m_NumTimeSteps;
}

//! Returns the minimum coordinate along the X axis
float BasicVolumeFileImpl::getMinX() const
{
	return m_MinX;
}

//! Returns the minimum coordinate along the Y axis
float BasicVolumeFileImpl::getMinY() const
{
	return m_MinY;
}

//! Returns the minimum coordinate along the Z axis
float BasicVolumeFileImpl::getMinZ() const
{
	return m_MinZ;
}

//! Returns the minimum time coordinate
float BasicVolumeFileImpl::getMinT() const
{
	return m_MinT;
}

//! Returns the maximum coordinate along the X axis
float BasicVolumeFileImpl::getMaxX() const
{
	return m_MaxX;
}

//! Returns the maximum coordinate along the Y axis
float BasicVolumeFileImpl::getMaxY() const
{
	return m_MaxY;
}

//! Returns the maximum coordinate along the Z axis
float BasicVolumeFileImpl::getMaxZ() const
{
	return m_MaxZ;
}

//! Returns the maximum time coordinate
float BasicVolumeFileImpl::getMaxT() const
{
	return m_MaxT;
}


// set* functions
//! Sets the number of variables stored in the file
void BasicVolumeFileImpl::setNumVariables(unsigned int number)
{
	// this will clear the contents of the info arrays if they exist
	makeSpaceForVariablesInfo(number);
}

//! Sets the type for the given variable v
void BasicVolumeFileImpl::setVariableType(unsigned int variable, Type type)
{
	if (variable < m_NumVariables)
		m_VariableTypes[variable] = type;
}

//! Sets the name of the given variable v
void BasicVolumeFileImpl::setVariableName(unsigned int variable, const QString& name)
{
	if (variable < m_NumVariables)
		m_VariableNames[variable] = name;
}

//! Sets the number of vertices in the X dimension.
void BasicVolumeFileImpl::setDimX(unsigned int x)
{
	m_DimX = x;
}

//! Sets the number of vertices in the Y dimension.
void BasicVolumeFileImpl::setDimY(unsigned int y)
{
	m_DimY = y;
}

//! Sets the number of vertices in the Z dimension.
void BasicVolumeFileImpl::setDimZ(unsigned int z)
{
	m_DimZ = z;
}

//! Sets the number of time steps.
void BasicVolumeFileImpl::setNumTimeSteps(unsigned int timeSteps)
{
	m_NumTimeSteps = timeSteps;
}

//! Sets the minimum coordinate along the X axis
void BasicVolumeFileImpl::setMinX(float x)
{
	m_MinX = x;
}

//! Sets the minimum coordinate along the Y axis
void BasicVolumeFileImpl::setMinY(float y)
{
	m_MinY = y;
}

//! Sets the minimum coordinate along the Z axis
void BasicVolumeFileImpl::setMinZ(float z)
{
	m_MinZ = z;
}

//! Sets the minimum time coordinate
void BasicVolumeFileImpl::setMinT(float t)
{
	m_MinT = t;
}

//! Sets the maximum coordinate along the X axis
void BasicVolumeFileImpl::setMaxX(float x)
{
	m_MaxX = x;
}

//! Sets the maximum coordinate along the Y axis
void BasicVolumeFileImpl::setMaxY(float y)
{
	m_MaxY = y;
}

//! Sets the maximum coordinate along the Z axis
void BasicVolumeFileImpl::setMaxZ(float z)
{
	m_MaxZ = z;
}

//! Sets the maximum time coordinate
void BasicVolumeFileImpl::setMaxT(float t)
{
	m_MaxT = t;
}

//! Sets the position of the next read. Each variable has its own position.
bool BasicVolumeFileImpl::setPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset)
{
	if (timeStep>=m_NumTimeSteps || variable>=m_NumVariables) {
		return false;
	}

	// do we need to perform an actual set?
	if (m_CurrentlyActiveVariable!=variable || 
		m_PositionTimeSteps[variable]!=timeStep || 
		m_PositionOffsets[variable]!=offset) {

		m_CurrentlyActiveVariable = variable;
		m_PositionTimeSteps[variable] = timeStep;
		m_PositionOffsets[variable] = offset;

		return protectedSetPosition(variable, timeStep, offset);
	}
	else {
		return true;
	}
}

//! Reads char data from the file into the supplied buffer
bool BasicVolumeFileImpl::readCharData(unsigned char* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep)
{
	if (!check(variable, timeStep, Char) || !setPosition(variable, timeStep, startPos)) {
		//no good
		return false;
	}

	return ((VolumeFile*)this)->readCharData(buffer, numSamples, variable);
}

//! Reads short data from the file into the supplied buffer
bool BasicVolumeFileImpl::readShortData(unsigned short* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep)
{
	if (!check(variable, timeStep, Short) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->readShortData(buffer, numSamples, variable);
}

//! Reads long data from the file into the supplied buffer
bool BasicVolumeFileImpl::readLongData(unsigned int* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep)
{
	if (!check(variable, timeStep, Long) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->readLongData(buffer, numSamples, variable);
}

//! Reads float data from the file into the supplied buffer
bool BasicVolumeFileImpl::readFloatData(float* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep)
{
	if (!check(variable, timeStep, Float) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->readFloatData(buffer, numSamples, variable);
}

//! Reads double data from the file into the supplied buffer
bool BasicVolumeFileImpl::readDoubleData(double* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep)
{
	if (!check(variable, timeStep, Double) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->readDoubleData(buffer, numSamples, variable);
}

//! Writes char data from the supplied buffer into the file.
bool BasicVolumeFileImpl::writeCharData(unsigned char* buffer,
	Q_ULLONG startPos, unsigned int numSamples, unsigned int variable,
	unsigned int timeStep)
{
	if (!check(variable, timeStep, Char) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->writeCharData(buffer, numSamples, variable);
}

//! Writes short data from the supplied buffer into the file.
bool BasicVolumeFileImpl::writeShortData(unsigned short* buffer,
	Q_ULLONG startPos, unsigned int numSamples, unsigned int variable,
	unsigned int timeStep)
{
	if (!check(variable, timeStep, Short) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->writeShortData(buffer, numSamples, variable);
}

//! Writes long data from the supplied buffer into the file.
bool BasicVolumeFileImpl::writeLongData(unsigned int* buffer,
	Q_ULLONG startPos, unsigned int numSamples, unsigned int variable,
	unsigned int timeStep)
{
	if (!check(variable, timeStep, Long) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->writeLongData(buffer, numSamples, variable);
}

//! Writes float data from the supplied buffer into the file.
bool BasicVolumeFileImpl::writeFloatData(float* buffer,
	Q_ULLONG startPos, unsigned int numSamples, unsigned int variable,
	unsigned int timeStep)
{
	if (!check(variable, timeStep, Float) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->writeFloatData(buffer, numSamples, variable);
}

//! Writes double data from the supplied buffer into the file.
bool BasicVolumeFileImpl::writeDoubleData(double* buffer,
	Q_ULLONG startPos, unsigned int numSamples, unsigned int variable,
	unsigned int timeStep)
{
	if (!check(variable, timeStep, Double) || !setPosition(variable, timeStep, startPos)) {
		// no good
		return false;
	}

	return ((VolumeFile*)this)->writeDoubleData(buffer, numSamples, variable);
}

//! Sets up default values for each member variable
void BasicVolumeFileImpl::initDefaults()
{
	m_VariableNames = 0;
	m_VariableTypes = 0;
	m_PositionOffsets = 0;
	m_PositionTimeSteps = 0;
	m_CurrentlyActiveVariable = 0xFFFFFFFF;
	m_NumVariables = 0;
	m_NumTimeSteps = 0;
	m_DimX = 128;
	m_DimY = 128;
	m_DimZ = 128;
	m_MinX = 0.0f;
	m_MinY = 0.0f;
	m_MinZ = 0.0f;
	m_MaxX = 1.0f;
	m_MaxY = 1.0f;
	m_MaxZ = 1.0f;
	m_MinT = 0.0f;
	m_MaxT = 0.0f;
}

//! Makes space for variable names and types
void BasicVolumeFileImpl::makeSpaceForVariablesInfo(unsigned int numVariables)
{
	destroyArrays();
	m_NumVariables = numVariables;
	m_VariableNames = new QString[numVariables];
	m_VariableTypes = new Type[numVariables];
	m_PositionOffsets = new Q_ULLONG[numVariables];
	m_PositionTimeSteps = new unsigned int[numVariables];
	unsigned int c;
	for (c=0; c<m_NumVariables; c++) {
		m_PositionOffsets[c] = 0;
		m_PositionTimeSteps[c] = 0;
	}
}

//! clears the variable arrays
void BasicVolumeFileImpl::destroyArrays()
{
	delete [] m_VariableNames;
	m_VariableNames = 0;
	delete [] m_VariableTypes;
	m_VariableTypes = 0;
	delete [] m_PositionOffsets;
	m_PositionOffsets = 0;
	delete [] m_PositionTimeSteps;
	m_PositionTimeSteps = 0;
}

//! prepares to read from the given variable
void BasicVolumeFileImpl::prepareToRead(unsigned int variable)
{
	if (m_CurrentlyActiveVariable!=variable) { 
		m_CurrentlyActiveVariable = variable;
		protectedSetPosition(variable, m_PositionTimeSteps[variable], m_PositionOffsets[variable]);
	}
}

//! increments the position of the currenty selected variable
void BasicVolumeFileImpl::incrementPosition(unsigned int offset)
{
	m_PositionOffsets[m_CurrentlyActiveVariable]+=offset;
}

bool BasicVolumeFileImpl::check(unsigned int variable, unsigned int timeStep, Type type)
{
	return (variable<m_NumVariables || timeStep<m_NumTimeSteps ||
		//m_VariableTypes[variable]==Char);
		m_VariableTypes[variable]==type);
}

//! Returns the number of bytes a sample of the given variable occupies
unsigned int BasicVolumeFileImpl::getBytesPerPixel(unsigned int variable) const
{
	unsigned int sizes[] = {1, 2, 4, 4};
	if (variable<m_NumVariables)
		return sizes[m_VariableTypes[variable]];
	else 
		return 0;
}

