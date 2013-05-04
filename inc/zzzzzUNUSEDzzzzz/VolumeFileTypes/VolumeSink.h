/*
  Copyright 2002-2004 The University of Texas at Austin
  
	Authors: John Wiggins <prok@ices.utexas.edu>
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

// VolumeSink.h: interface for the VolumeSink class.
//
//////////////////////////////////////////////////////////////////////

#ifndef VOLUME_SINK_H
#define VOLUME_SINK_H

#include <VolumeFileTypes/VolumeFile.h>
#include <qstring.h>

typedef unsigned int uint;

///\class VolumeSink VolumeSink.h
///\brief The VolumeSink class is analogous to the VolumeSource class. Instead
///	of being a source of volume data, it's a sink (ie- a destination). A sink
/// could be a file, a socket, a memory buffer, etc.
///\author John Wiggins
class VolumeSink
{
public:
	//enum Type { Char, Short, Float, Long, Double };

	VolumeSink();
	virtual ~VolumeSink();

///\fn virtual bool writeHeader() = 0
///\brief This function writes the header to the sink.
///\return A bool indicating success or failure
	virtual bool writeHeader() = 0;
///\fn virtual bool writeRawData(char* data, uint xMin, uint yMin, uint zMin, uint xMax, uint yMax, uint zMax, uint variable, uint timeStep) = 0
///\brief This function writes raw data to the sink. The data buffer is a
/// volume with X varying fastest, followed by Y, followed by Z.
///\param data A buffer containing a 3D volume
///\param xMin The starting coordinate on the X axis
///\param yMin The starting coordinate on the Y axis
///\param zMin The starting coordinate on the Z axis
///\param xMax The ending coordinate on the X axis
///\param yMax The ending coordinate on the Y axis
///\param zMax The ending coordinate on the Z axis
///\param variable The index of the variable to write
///\param timeStep The time step to write
///\return A bool indicating success or failure
	virtual bool writeRawData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, uint variable, uint timeStep) = 0;

///\fn virtual void setNumVariables(unsigned int num)
///\brief This function sets the number of variables for the sink.
///\param num The number of variables
	virtual void setNumVariables(unsigned int num);
///\fn virtual void setNumTimeSteps(unsigned int num)
///\brief This function sets the number of time steps for the sink.
///\param num The number of time steps
	virtual void setNumTimeSteps(unsigned int num);

///\fn virtual void setVariableName(unsigned int variable, QString name) = 0
///\brief This function sets a specified variable's name.
///\param variable The index of the variable
///\param name The name for the variable
	virtual void setVariableName(unsigned int variable, QString name) = 0;
///\fn virtual void setVariableType(unsigned int variable, VolumeFile::Type type) = 0
///\brief This function sets a specified variable's type.
///\param variable The index of the variable
///\param type The type for the variable
	virtual void setVariableType(unsigned int variable, VolumeFile::Type type) = 0;

///\fn void setDimX(unsigned int dimX)
///\brief This function sets the number of vertices in the X dimension.
///\param dimX The number of vertices
	void setDimX(unsigned int dimX);
///\fn void setDimY(unsigned int dimY)
///\brief This function sets the number of vertices in the Y dimension.
///\param dimY The number of vertices
	void setDimY(unsigned int dimY);
///\fn void setDimZ(unsigned int dimZ)
///\brief This function sets the number of vertices in the Z dimension.
///\param dimZ The number of vertices
	void setDimZ(unsigned int dimZ);
///\fn void setMinX(double minX)
///\brief This function sets the minimum coordinate along the X axis.
///\param minX The coordinate
	void setMinX(double minX);
///\fn void setMinY(double minY)
///\brief This function sets the minimum coordinate along the Y axis.
///\param minY The coordinate
	void setMinY(double minY);
///\fn void setMinZ(double minZ)
///\brief This function sets the minimum coordinate along the Z axis.
///\param minZ The coordinate
	void setMinZ(double minZ);
///\fn void setMinT(double minT)
///\brief This function sets the minimum time coordinate.
///\param minT The coordinate
	void setMinT(double minT);
///\fn void setMaxX(double maxX)
///\brief This function sets the maximum coordinate along the X axis.
///\param maxX The coordinate
	void setMaxX(double maxX);
///\fn void setMaxY(double maxY)
///\brief This function sets the maximum coordinate along the Y axis.
///\param maxY The coordinate
	void setMaxY(double maxY);
///\fn void setMaxZ(double maxZ)
///\brief This function sets the maximum coordinate along the Z axis.
///\param maxZ The coordinate
	void setMaxZ(double maxZ);
///\fn void setMaxT(double maxT)
///\brief This function sets the maximum time coordinate.
///\param maxT The coordinate
	void setMaxT(double maxT);
	
///\fn unsigned int getDimX() const
///\brief Returns the number of vertices in the X dimension
///\return The number of vertices in the X dimension
	unsigned int getDimX() const;
///\fn unsigned int getDimY() const
///\brief Returns the number of vertices in the Y dimension
///\return The number of vertices in the Y dimension
	unsigned int getDimY() const;
///\fn unsigned int getDimZ() const
///\brief Returns the number of vertices in the Z dimension
///\return The number of vertices in the Z dimension
	unsigned int getDimZ() const;
///\fn double getMinX() const
///\brief Returns the minimum coordinate along the X axis.
///\return The minimum X coordinate
	double getMinX() const;
///\fn double getMinY() const
///\brief Returns the minimum coordinate along the Y axis.
///\return The minimum Y coordinate
	double getMinY() const;
///\fn double getMinZ() const
///\brief Returns the minimum coordinate along the Z axis.
///\return The minimum Z coordinate
	double getMinZ() const;
///\fn double getMinT() const
///\brief Returns the minimum time coordinate.
///\return The minimum time coordinate
	double getMinT() const;
///\fn double getMaxX() const
///\brief Returns the maximum coordinate along the X axis.
///\return The maximum X coordinate
	double getMaxX() const;
///\fn double getMaxY() const
///\brief Returns the maximum coordinate along the Y axis.
///\return The maximum Y coordinate
	double getMaxY() const;
///\fn double getMaxZ() const
///\brief Returns the maximum coordinate along the Z axis.
///\return The maximum Z coordinate
	double getMaxZ() const;
///\fn double getMaxT() const
///\brief Returns the maximum time coordinate.
///\return The maximum time coordinate
	double getMaxT() const;

///\fn bool error() const
///\brief This function indicates whether or not there has been an error.
///\return A bool indicating whether or not an error has occured
	bool error() const;
///\fn bool errorMustRestart() const
///\brief This function tells whether or not an error is fatal or not.
///\return A bool indicating whether or not a fatal error has occured
	bool errorMustRestart() const;
///\fn const QString& errorReason() const
///\brief This function returns a QString describing an error.
///\return A QString describing an error
	const QString& errorReason() const;

///\fn virtual void resetError()
///\brief This function resets the error state, clearing any non-fatal errors.
	virtual void resetError();

protected:
	void setError(const QString& reason, bool mustRestart = false);
	virtual void setDefaults();
	double m_MinX, m_MinY, m_MinZ;
	double m_MaxX, m_MaxY, m_MaxZ;
	double m_MinT, m_MaxT;
	unsigned int m_DimX, m_DimY, m_DimZ;
	unsigned int m_NumVariables, m_NumTimeSteps;

	QString m_ErrorReason;
	bool m_Error;
	bool m_ErrorMustRestart;

};

#endif

