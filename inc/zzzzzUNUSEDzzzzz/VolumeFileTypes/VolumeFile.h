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

//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMEFILE_H__66D7982D_D1CF_4453_9CF2_959C9719F054__INCLUDED_)
#define AFX_VOLUMEFILE_H__66D7982D_D1CF_4453_9CF2_959C9719F054__INCLUDED_

#include <qglobal.h> // for Q_ULLONG

#if !WIN32 || defined(USING_GCC)
// yes, this is lame. Earlier versions of Qt don't define it
#ifndef Q_ULLONG
typedef unsigned long long Q_ULLONG;
#endif

#else

#ifndef Q_ULLONG
typedef unsigned __int64 Q_ULLONG;
#endif

#endif

class QString;

	
///\enum VolumeFile::Type
///\brief The different types volume files can contain

///\var VolumeFile::Type VolumeFile::Char
/// unsigned char data

///\var VolumeFile::Type VolumeFile::Short
/// unsigned short data

///\var VolumeFile::Type VolumeFile::Float
/// float data

///\var VolumeFile::Type VolumeFile::Long
/// unsigned long data

///\var VolumeFile::Type VolumeFile::Double
/// double data

///\enum VolumeFile::Mode
///\brief The different open modes for a file

///\var VolumeFile::Mode VolumeFile::Read
/// File is opened for reading

///\var VolumeFile::Mode VolumeFile::Write
/// File is opened for writing

///\class VolumeFile VolumeFile.h
///\brief The pure virtual base class for all volume file types.
///\author Anthony Thane
///\author John Wiggins
class VolumeFile  
{
public:
	enum Type { Char, Short, Float, Long, Double };

	enum Mode { Read, Write };

	VolumeFile();
	virtual ~VolumeFile();

///\fn virtual bool checkType(const QString& fileName) = 0
///\brief Returns true if the file given by fileName is of the correct type.
///\param fileName A path to a file
///\return A bool indicating success or failure
	virtual bool checkType(const QString& fileName) = 0;
///\fn virtual bool attachToFile(const QString& fileName, Mode mode = Read) = 0
///\brief Associates this reader with the file given by fileName.
///\param fileName A path to a file
///\param mode A Mode specifying either Read or Write (not both)
///\return A bool indicating success or failure
	virtual bool attachToFile(const QString& fileName, Mode mode = Read) = 0;

///\fn virtual QString getExtension() = 0
///\brief Returns the extension for this file type.
///\return A QString with the extension for this file type
	virtual QString getExtension() = 0;
///\fn virtual QString getFilter() = 0
///\brief Returns the file dialog filter for the file type.
///\return A QString containing a QFileDialog-friendly filter
	virtual QString getFilter() = 0;

	// get* functions
///\fn virtual unsigned int getNumVariables() const = 0
///\brief Returns the number of variables stored in the file
///\return The number of variables stored in the file
	virtual unsigned int getNumVariables() const = 0;

///\fn virtual Type getVariableType(unsigned int v) const = 0
///\brief Returns the type for the given variable v
///\param v A variable index
///\return The Type of variable v
	virtual Type getVariableType(unsigned int v) const = 0;
///\fn virtual QString getVariableName(unsigned int v) const = 0
///\brief Returns the name of the given variable v
///\param v A variable index
///\return A QString containing the name of the variable
	virtual QString getVariableName(unsigned int v) const = 0;

///\fn virtual unsigned int getDimX() const = 0
///\brief Returns the number of vertices in the X dimension.
///\return The number of vertices in the X dimension
	virtual unsigned int getDimX() const = 0;
///\fn virtual unsigned int getDimY() const = 0
///\brief Returns the number of vertices in the Y dimension.
///\return The number of vertices in the Y dimension
	virtual unsigned int getDimY() const = 0;
///\fn virtual unsigned int getDimZ() const = 0
///\brief Returns the number of vertices in the Z dimension.
///\return The number of vertices in the Z dimension
	virtual unsigned int getDimZ() const = 0;
///\fn virtual unsigned int getNumTimeSteps() const = 0
///\brief Returns the number of time steps.
///\return The number of time steps
	virtual unsigned int getNumTimeSteps() const = 0;

///\fn virtual float getMinX() const = 0
///\brief Returns the minimum coordinate along the X axis
///\return The minimum coordinate along the X axis
	virtual float getMinX() const = 0;
///\fn virtual float getMinY() const = 0
///\brief Returns the minimum coordinate along the Y axis
///\return The minimum coordinate along the Y axis
	virtual float getMinY() const = 0;
///\fn virtual float getMinZ() const = 0
///\brief Returns the minimum coordinate along the Z axis
///\return The minimum coordinate along the Z axis
	virtual float getMinZ() const = 0;
///\fn virtual float getMinT() const = 0
///\brief Returns the minimum time coordinate
///\return The minimum time coordinate
	virtual float getMinT() const = 0;
///\fn virtual float getMaxX() const = 0
///\brief Returns the maximum coordinate along the X axis
///\return The maximum coordinate along the X axis
	virtual float getMaxX() const = 0;
///\fn virtual float getMaxY() const = 0
///\brief Returns the maximum coordinate along the Y axis
///\return The maximum coordinate along the Y axis
	virtual float getMaxY() const = 0;
///\fn virtual float getMaxZ() const = 0
///\brief Returns the maximum coordinate along the Z axis
///\return The maximum coordinate along the Z axis
	virtual float getMaxZ() const = 0;
///\fn virtual float getMaxT() const = 0
///\brief Returns the maximum time coordinate
///\return The maximum time coordinate
	virtual float getMaxT() const = 0;
	
	// set* functions
///\fn virtual void setNumVariables(unsigned int number) = 0
///\brief Sets the number of variables stored in the file
///\param number The number of variables
	virtual void setNumVariables(unsigned int number) = 0;

///\fn virtual void setVariableType(unsigned int variable, Type type) = 0
///\brief Sets the type for the given variable.
///\param variable A variable index
///\param type The Type of the variable
	virtual void setVariableType(unsigned int variable, Type type) = 0;
///\fn virtual void setVariableName(unsigned int variable, const QString& name) = 0
///\brief Sets the name of the given variable v
///\param variable A variable index
///\param name A QString containing the variable name. (note that some filetypes may have a restriction on the length of the name)
	virtual void setVariableName(unsigned int variable, const QString& name) = 0;
	
///\fn virtual void setDimX(unsigned int x) = 0
///\brief Sets the number of vertices in the X dimension.
///\param x The number of vertices
	virtual void setDimX(unsigned int x) = 0;
///\fn virtual void setDimY(unsigned int y) = 0
///\brief sets the number of vertices in the y dimension.
///\param y The number of vertices
	virtual void setDimY(unsigned int y) = 0;
///\fn virtual void setDimZ(unsigned int z) = 0
///\brief Sets the number of vertices in the Z dimension.
///\param z The number of vertices
	virtual void setDimZ(unsigned int z) = 0;
///\fn virtual void setNumTimeSteps(unsigned int timeSteps) = 0
///\brief Sets the number of time steps.
///\param timeSteps The number of time steps
	virtual void setNumTimeSteps(unsigned int timeSteps) = 0;
	
///\fn virtual void setMinX(float x) = 0
///\brief Sets the minimum coordinate along the X axis
///\param x The coordinate
	virtual void setMinX(float x) = 0;
///\fn virtual void setMinY(float y) = 0
///\brief Sets the minimum coordinate along the Y axis
///\param y The coordinate
	virtual void setMinY(float y) = 0;
///\fn virtual void setMinZ(float z) = 0
///\brief Sets the minimum coordinate along the Z axis
///\param z The coordinate
	virtual void setMinZ(float z) = 0;
///\fn virtual void setMinT(float t) = 0
///\brief Sets the minimum time coordinate
///\param t The coordinate
	virtual void setMinT(float t) = 0;
///\fn virtual void setMaxX(float x) = 0
///\brief Sets the maximum coordinate along the X axis
///\param x The coordinate
	virtual void setMaxX(float x) = 0;
///\fn virtual void setMaxY(float y) = 0
///\brief Sets the maximum coordinate along the Y axis
///\param y The coordinate
	virtual void setMaxY(float y) = 0;
///\fn virtual void setMaxZ(float z) = 0
///\brief Sets the maximum coordinate along the Z axis
///\param z The coordinate
	virtual void setMaxZ(float z) = 0;
///\fn virtual void setMaxT(float t) = 0
///\brief Sets the maximum time coordinate
///\param t The coordinate
	virtual void setMaxT(float t) = 0;

///\fn virtual bool setPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset) = 0
///\brief Sets the position of the next read. Each variable has its own position.
///\param variable A variable index
///\param timeStep A time step index
///\param offset An offset (in samples) from the start of the variable and time step
///\return A bool indicating success or failure
	virtual bool setPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset) = 0;

	// read* functions
///\fn virtual bool readCharData(unsigned char* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Reads char data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param startPos A starting offset for the read (in samples)
///\param numSamples The number of samples to be read
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool readCharData(unsigned char* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool readShortData(unsigned short* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Reads short data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param startPos A starting offset for the read (in samples)
///\param numSamples The number of samples to be read
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool readShortData(unsigned short* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool readLongData(unsigned int* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Reads long data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param startPos A starting offset for the read (in samples)
///\param numSamples The number of samples to be read
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool readLongData(unsigned int* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool readFloatData(float* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Reads float data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param startPos A starting offset for the read (in samples)
///\param numSamples The number of samples to be read
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool readFloatData(float* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool readDoubleData(double* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Reads double data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param startPos A starting offset for the read (in samples)
///\param numSamples The number of samples to be read
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool readDoubleData(double* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;

///\fn virtual bool readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Reads char data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param numSamples The number of samples to be read
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Reads short data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param numSamples The number of samples to be read
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Reads long data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param numSamples The number of samples to be read
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool readFloatData(float* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Reads float data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param numSamples The number of samples to be read
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool readFloatData(float* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Reads double data from the file into the supplied buffer.
///\param buffer A memory buffer
///\param numSamples The number of samples to be read
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable) = 0;

	// write* functions
///\fn virtual bool writeHeader() = 0
///\brief Writes the header into the file
///\return A bool indicating success or failure
	virtual bool writeHeader() = 0;

///\fn virtual bool writeCharData(unsigned char* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Writes char data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param startPos A starting offset for the write (in samples)
///\param numSamples The number of samples to be written
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool writeCharData(unsigned char* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool writeShortData(unsigned short* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Writes short data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param startPos A starting offset for the write (in samples)
///\param numSamples The number of samples to be written
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool writeShortData(unsigned short* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool writeLongData(unsigned int* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Writes long data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param startPos A starting offset for the write (in samples)
///\param numSamples The number of samples to be written
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool writeLongData(unsigned int* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool writeFloatData(float* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Writes float data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param startPos A starting offset for the write (in samples)
///\param numSamples The number of samples to be written
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool writeFloatData(float* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
///\fn virtual bool writeDoubleData(double* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0
///\brief Writes double data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param startPos A starting offset for the write (in samples)
///\param numSamples The number of samples to be written
///\param variable A variable index
///\param timeStep A time step index
///\return A bool indicating success or failure
	virtual bool writeDoubleData(double* buffer, Q_ULLONG startPos, unsigned int numSamples, unsigned int variable, unsigned int timeStep) = 0;
	
///\fn virtual bool writeCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Writes char data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param numSamples The number of samples to be written
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool writeCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool writeShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Writes short data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param numSamples The number of samples to be written
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool writeShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool writeLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Writes long data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param numSamples The number of samples to be written
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool writeLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool writeFloatData(float* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Writes float data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param numSamples The number of samples to be written
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool writeFloatData(float* buffer, unsigned int numSamples, unsigned int variable) = 0;
///\fn virtual bool writeDoubleData(double* buffer, unsigned int numSamples, unsigned int variable) = 0
///\brief Writes double data from the supplied buffer into the file.
///\param buffer A memory buffer
///\param numSamples The number of samples to be written
///\param variable A variable index
///\return A bool indicating success or failure
	virtual bool writeDoubleData(double* buffer, unsigned int numSamples, unsigned int variable) = 0;
	
///\fn virtual VolumeFile* getNewVolumeFileLoader() const = 0
///\brief Returns a new loader which can be used to load a file
///\return A pointer to a new VolumeFile instance
	virtual VolumeFile* getNewVolumeFileLoader() const = 0;
	
};

#endif // !defined(AFX_VOLUMEFILE_H__66D7982D_D1CF_4453_9CF2_959C9719F054__INCLUDED_)
