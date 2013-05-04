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

// BasicVolumeFileImpl.h: interface for the BasicVolumeFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_BASICVOLUMEFILEIMPL_H__E5493BD6_4F0E_4A31_B746_58FC163B4AC0__INCLUDED_)
#define AFX_BASICVOLUMEFILEIMPL_H__E5493BD6_4F0E_4A31_B746_58FC163B4AC0__INCLUDED_

#include <VolumeFileTypes/VolumeFile.h>

///\class BasicVolumeFileImpl BasicVolumeFileImpl.h
///\brief An abstract class that provides a default implementation
/// for some of the methods of VolumeFile.
///\author Anthony Thane
///\author John Wiggins
class BasicVolumeFileImpl : public VolumeFile  
{
public:
	BasicVolumeFileImpl();
	virtual ~BasicVolumeFileImpl();

	virtual unsigned int getNumVariables() const;

	virtual Type getVariableType(unsigned int v) const;
	virtual QString getVariableName(unsigned int v) const;

	virtual unsigned int getDimX() const;
	virtual unsigned int getDimY() const;
	virtual unsigned int getDimZ() const;
	virtual unsigned int getNumTimeSteps() const;

	virtual float getMinX() const;
	virtual float getMinY() const;
	virtual float getMinZ() const;
	virtual float getMinT() const;
	virtual float getMaxX() const;
	virtual float getMaxY() const;
	virtual float getMaxZ() const;
	virtual float getMaxT() const;

	// set* functions
	virtual void setNumVariables(unsigned int number);

	virtual void setVariableType(unsigned int variable, Type type);
	virtual void setVariableName(unsigned int variable, const QString& name);
	
	virtual void setDimX(unsigned int x);
	virtual void setDimY(unsigned int y);
	virtual void setDimZ(unsigned int z);
	virtual void setNumTimeSteps(unsigned int timeSteps);
	
	virtual void setMinX(float x);
	virtual void setMinY(float y);
	virtual void setMinZ(float z);
	virtual void setMinT(float t);
	virtual void setMaxX(float x);
	virtual void setMaxY(float y);
	virtual void setMaxZ(float z);
	virtual void setMaxT(float t);

	virtual bool setPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset);

	virtual bool readCharData(unsigned char* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool readShortData(unsigned short* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool readLongData(unsigned int* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool readFloatData(float* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool readDoubleData(double* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	
	virtual bool writeCharData(unsigned char* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool writeShortData(unsigned short* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool writeLongData(unsigned int* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool writeFloatData(float* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);
	virtual bool writeDoubleData(double* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep);

protected:
///\fn void initDefaults()
///\brief Sets up default values for each member variable
	void initDefaults();
///\fn void makeSpaceForVariablesInfo(unsigned int numVariables)
///\brief Makes space for variable names and types
///\param numVariables The number of variables to make space for
	void makeSpaceForVariablesInfo(unsigned int numVariables);
///\fn void destroyArrays()
///\brief clears the variable arrays
	void destroyArrays();
///\fn void prepareToRead(unsigned int variable)
///\brief prepares to read from the given variable
///\param variable A variable index
	void prepareToRead(unsigned int variable);
///\fn void incrementPosition(unsigned int offset)
///\brief increments the position of the currenty selected variable
///\param offset An offset (in samples)
	void incrementPosition(unsigned int offset);

///\fn virtual bool protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset) = 0
///\brief Sets the position of the next read.
///\param variable A variable index
///\param timeStep A time step
///\param offset An offset (in samples) from the beginning of the variable and time step
///\return A bool indicating success or failure
	virtual bool protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset) = 0;

///\fn bool check(unsigned int variable, unsigned int timeStep, Type type)
///\brief Checks that the variable, timeStep, and type are correct
///\param variable A variable index
///\param timeStep A time step
///\param type A Type
///\return A bool indicating success or failure
	bool check(unsigned int variable, unsigned int timeStep, Type type);

///\fn unsigned int getBytesPerPixel(unsigned int variable) const
///\brief Returns the number of bytes a sample of the given variable occupies
///\param variable A variable index
///\return The number of bytes per sample
	unsigned int getBytesPerPixel(unsigned int variable) const;

	unsigned int m_NumTimeSteps, m_NumVariables;
	unsigned int m_DimX, m_DimY, m_DimZ;
	float m_MinX, m_MinY, m_MinZ;
	float m_MaxX, m_MaxY, m_MaxZ;
	float m_MinT, m_MaxT;

	unsigned int m_CurrentlyActiveVariable;
	QString* m_VariableNames;
	Type* m_VariableTypes;
	Q_ULLONG* m_PositionOffsets;
	unsigned int* m_PositionTimeSteps;

};

#endif // !defined(AFX_BASICVOLUMEFILEIMPL_H__E5493BD6_4F0E_4A31_B746_58FC163B4AC0__INCLUDED_)
