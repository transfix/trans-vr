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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// VolumeSource.h: interface for the VolumeSource class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMESOURCE_H__6129CDB3_A02D_4ECC_8168_5AE808FD4365__INCLUDED_)
#define AFX_VOLUMESOURCE_H__6129CDB3_A02D_4ECC_8168_5AE808FD4365__INCLUDED_

#include <qstring.h>
#include <qobject.h>

#if !WIN32

#ifndef Q_ULLONG
typedef unsigned long long Q_ULLONG;
#endif

#else

#ifndef Q_ULLONG
typedef unsigned __int64 Q_ULLONG;
#endif

#endif

typedef unsigned int uint;

///\class VolumeSource VolumeSource.h
///\brief The abstract base class VolumeSource is a generic source of volume
///	data.
///\author Anthony Thane
///\author John Wiggins

///\enum VolumeSource::DownLoadFrequency
///\brief Specifies when volume data is updated when interacting with the
///	rover widget.

///\var VolumeSource::DownLoadFrequency VolumeSource::DLFInteractive
/// Updates happen as the rover is moving

///\var VolumeSource::DownLoadFrequency VolumeSource::DLFDelayed
/// Updates happen after the mouse button is released

///\var VolumeSource::DownLoadFrequency VolumeSource::DLFManual
/// Updates happen when the user explicitly requests them

class VolumeSource  
{
public:
	enum DownLoadFrequency { DLFInteractive, DLFDelayed, DLFManual };

	VolumeSource();
	virtual ~VolumeSource();

///\fn virtual void fillThumbnail(char* data, unsigned int xDim, unsigned int yDim, unsigned int zDim, uint variable, uint timeStep) = 0
///\deprecated This function is not used/implemented.
	virtual void fillThumbnail(char* data, 
		unsigned int xDim, unsigned int yDim, unsigned int zDim, uint variable, uint timeStep) = 0;

///\fn virtual void fillData(char* data, uint xMin, uint yMin, uint zMin, uint xMax, uint yMax, uint zMax, uint xDim, uint yDim, uint zDim, uint variable, uint timeStep) = 0
///\brief This function fills a buffer with volume data from a requested
///	region within a variable & time step. All data read is one byte per
///	sample.
///\param data The buffer to be filled
///\param xMin The minimum X coordinate
///\param yMin The minimum Y coordinate
///\param zMin The minimum Z coordinate
///\param xMax The maximum X coordinate
///\param yMax The maximum Y coordinate
///\param zMax The maximum Z coordinate
///\param xDim The number of samples to read along the X axis
///\param yDim The number of samples to read along the Y axis
///\param zDim The number of samples to read along the Z axis
///\param variable The variable to read from
///\param timeStep The time step to read from
	virtual void fillData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep) = 0;

///\fn virtual bool fillGradientData(char* data, uint xMin, uint yMin, uint zMin, uint xMax, uint yMax, uint zMax,uint xDim, uint yDim, uint zDim, uint variable, uint timeStep)
///\brief This function fills a buffer with gradient data for the specified
/// region of the volume.
///\param data The buffer to be filled
///\param xMin The minimum X coordinate
///\param yMin The minimum Y coordinate
///\param zMin The minimum Z coordinate
///\param xMax The maximum X coordinate
///\param yMax The maximum Y coordinate
///\param zMax The maximum Z coordinate
///\param xDim The number of samples to read along the X axis
///\param yDim The number of samples to read along the Y axis
///\param zDim The number of samples to read along the Z axis
///\param variable The variable to read from
///\param timeStep The time step to read from
///\return A bool indicating success or failure
	virtual bool fillGradientData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep);

///\fn virtual bool readRawData(char* data, uint xMin, uint yMin, uint zMin, uint xMax, uint yMax, uint zMax, uint variable, uint timeStep)
///\brief This function reads data from some region of the volume into a
/// buffer. Samples may be larger than one byte.
///\param data The buffer to be filled
///\param xMin The minimum X coordinate
///\param yMin The minimum Y coordinate
///\param zMin The minimum Z coordinate
///\param xMax The maximum X coordinate
///\param yMax The maximum Y coordinate
///\param zMax The maximum Z coordinate
///\param variable The variable to read from
///\param timeStep The time step to read from
///\return A bool indicating success or failure
	virtual bool readRawData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, uint variable, uint timeStep);

///\fn virtual unsigned int getNumVars() const
///\brief This function returns the number of variables.
///\return The number of variables
	virtual unsigned int getNumVars() const;
///\fn virtual unsigned int getNumTimeSteps() const
///\brief This function returns the number of time steps.
///\return The number of time steps
	virtual unsigned int getNumTimeSteps() const;

///\fn virtual unsigned int getVariableType(unsigned int variable) const = 0
///\brief This function returns the data type of a specific variable.
///\param variable The variable
///\return An unsigned int that can be cast to VolumeFile::Type
	virtual unsigned int getVariableType(unsigned int variable) const = 0;
///\fn virtual double getFunctionMinimum(unsigned int variable, unsigned int timeStep) const
///\brief This function returns the minimum value for the variable and time
///	step specified.
///\param variable The variable
///\param timeStep The time step
///\return A double representing the minimum value of the function
	virtual double getFunctionMinimum(unsigned int variable, unsigned int timeStep) const;
///\fn virtual double getFunctionMaximum(unsigned int variable, unsigned int timeStep) const
///\brief This function returns the maximum value for the variable and time
///	step specified.
///\param variable The variable
///\param timeStep The time step
///\return A double representing the maximum value of the function
	virtual double getFunctionMaximum(unsigned int variable, unsigned int timeStep) const;

///\fn virtual QString getVariableName(unsigned int variable) const
///\brief This function returns the name of a specified variable.
///\param variable The variable
///\return A QString object containing the variable's name
	virtual QString getVariableName(unsigned int variable) const;
///\fn virtual QString getContourSpectrumFileName(unsigned int variable, unsigned int timeStep)
///\brief This function returns the local pathname of a file containing a
///	cached contour spectrum.
///\param variable The variable
///\param timeStep The time step
///\return A QString that is either QString::null or the pathname of a file
	virtual QString getContourSpectrumFileName(unsigned int variable, unsigned int timeStep);
///\fn virtual QString getContourTreeFileName(unsigned int variable, unsigned int timeStep)
///\brief This function returns the local pathname of a file containing a
///	cached contour tree.
///\param variable The variable
///\param timeStep The time step
///\return A QString that is either QString::null or the pathname of a file
	virtual QString getContourTreeFileName(unsigned int variable, unsigned int timeStep);

///\fn virtual void computeContourSpectrum(QObject *obj, unsigned int variable, unsigned int timeStep)
///\brief This function computes a contour spectrum for some variable and time
///	step.
///\param obj A QObject instance for the threaded version of this function. It
///	is a pointer to an object that should be notified when the process has
///	completed. (you can safely pass NULL here if VFS_USE_THREADING is not
///	#define'd)
///\param variable The variable
///\param timeStep The time step
	virtual void computeContourSpectrum(QObject *obj, unsigned int variable, unsigned int timeStep);
///\fn virtual void computeContourTree(QObject *obj, unsigned int variable, unsigned int timeStep)
///\brief This function computes a contour tree for some variable and time
///	step.
///\param obj A QObject instance for the threaded version of this function. It
///	is a pointer to an object that should be notified when the process has
///	completed. (you can safely pass NULL here if VFS_USE_THREADING is not
///	#define'd)
///\param variable The variable
///\param timeStep The time step
	virtual void computeContourTree(QObject *obj, unsigned int variable, unsigned int timeStep);

///\fn virtual void cleanUpWorkerThread(int thid)
///\brief If VFS_USE_THREADING is #defined, this function is used to clean up
///	thread objects.
///\param thid The thread id of the thread.
	virtual void cleanUpWorkerThread(int thid);
	
	//unsigned int getNumVerts() const;
	//unsigned int getNumCells() const;
///\fn Q_ULLONG getNumVerts() const;
///\brief This function returns the number of vertices in the volume.
///\return The number of vertices
	Q_ULLONG getNumVerts() const;
///\fn Q_ULLONG getNumCells() const;
///\brief This function returns the number of cells in the volume.
///\return The number of cells in the volume
	Q_ULLONG getNumCells() const;

///\fn double getMinX() const;
///\brief Returns the minimum X coordinate
///\return The minimum X coordinate
	double getMinX() const;
///\fn double getMinY() const;
///\brief Returns the minimum Y coordinate
///\return The minimum Y coordinate
	double getMinY() const;
///\fn double getMinZ() const;
///\brief Returns the minimum Z coordinate
///\return The minimum Z coordinate
	double getMinZ() const;
///\fn double getMinT() const;
///\brief Returns the minimum time coordinate
///\return The minimum time coordinate
	double getMinT() const;
///\fn double getMaxX() const;
///\brief Returns the maximum X coordinate
///\return The maximum X coordinate
	double getMaxX() const;
///\fn double getMaxY() const;
///\brief Returns the maximum Y coordinate
///\return The maximum Y coordinate
	double getMaxY() const;
///\fn double getMaxZ() const;
///\brief Returns the Maximum Z coordinate
///\return The maximum Z coordinate
	double getMaxZ() const;
///\fn double getMaxT() const;
///\brief Returns the maximum time coordinate
///\return The maximum time coordinate
	double getMaxT() const;
///\fn unsigned int getDimX() const;
///\brief Returns the number of samples along the X axis
///\return The number of samples along the X axis
	unsigned int getDimX() const;
///\fn unsigned int getDimY() const;
///\brief Returns the number of samples along the Y axis
///\return The number of samples along the Y axis
	unsigned int getDimY() const;
///\fn unsigned int getDimZ() const;
///\brief Returns the number of samples along the Z axis
///\return The number of samples along the Z axis
	unsigned int getDimZ() const;

///\fn void setMinX(double minX);
///\brief This function sets the minimum X coordinate.
///\param minX The new minimum X coordinate
	void setMinX(double minX);
///\fn void setMinY(double minY);
///\brief This function sets the minimum Y coordinate.
///\param minY The new minimum Y coordinate
	void setMinY(double minY);
///\fn void setMinZ(double minZ);
///\brief This function sets the minimum Z coordinate.
///\param minZ The new minimum Z coordinate
	void setMinZ(double minZ);
///\fn void setMinT(double minT);
///\brief This function sets the minimum time coordinate.
///\param minT The new minimum time coordinate
	void setMinT(double minT);
///\fn void setMaxX(double maxX);
///\brief This function sets the maximum X coordinate.
///\param maxX The new maximum X coordinate
	void setMaxX(double maxX);
///\fn void setMaxY(double maxY);
///\brief This function sets the maximum Y coordinate.
///\param maxY The new maximum Y coordinate
	void setMaxY(double maxY);
///\fn void setMaxZ(double maxZ);
///\brief This function sets the maximum Z coordinate.
///\param maxZ The new maximum Z coordinate
	void setMaxZ(double maxZ);
///\fn void setMaxT(double maxT);
///\brief This function sets the maximum time coordinate.
///\param maxT The new maximum time coordinate
	void setMaxT(double maxT);
///\fn void setDimX(unsigned int dimX);
///\brief This function sets the number of samples along the X axis.
///\param dimX The new number of samples along the X axis
	void setDimX(unsigned int dimX);
///\fn void setDimY(unsigned int dimY);
///\brief This function sets the number of samples along the Y axis.
///\param dimY The new number of samples along the Y axis
	void setDimY(unsigned int dimY);
///\fn void setDimZ(unsigned int dimZ);
///\brief This function sets the number of samples along the Z axis.
///\param dimZ The new number of samples along the Z axis
	void setDimZ(unsigned int dimZ);

///\fn bool error() const;
///\brief Returns the current error state.
///\return True if there is an error, false if there is not.
	bool error() const;
///\fn bool errorMustRestart() const;
///\brief Returns whether an error is fatal or not.
///\return True if the current error is fatal, false if it is not
	bool errorMustRestart() const;
///\fn const QString& errorReason() const;
///\brief Returns a string describing the current error.
///\return A QString containing the error message
	const QString& errorReason() const;

///\fn virtual DownLoadFrequency interactiveUpdateHint() const;
///\brief Returns the preferred frequency of download requests. This should be
///	implemented by derived classes. See the description of 
///	VolumeSource::DownLoadFrequency for more details.
///\return A DownLoadFreqency
	virtual DownLoadFrequency interactiveUpdateHint() const;

///\fn virtual void resetError();
///\brief Resets any non fatal errors
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

#endif // !defined(AFX_VOLUMESOURCE_H__6129CDB3_A02D_4ECC_8168_5AE808FD4365__INCLUDED_)


