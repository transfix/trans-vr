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

// RawVFileImpl.cpp: implementation of the RawVFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#include <qfileinfo.h>
#include <VolumeFileTypes/RawVFileImpl.h>
#include <ByteOrder/ByteSwapping.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RawVFileImpl::RawVFileImpl()
{
	m_Attached = false;
	m_VariableStartingLocations = 0;
}

RawVFileImpl::~RawVFileImpl()
{
	close();
}

void RawVFileImpl::setNumVariables(unsigned int number)
{
	// call the super-class version
	BasicVolumeFileImpl::setNumVariables(number);

	// allocate space
	if (m_VariableStartingLocations) delete [] m_VariableStartingLocations;
	m_VariableStartingLocations = new Q_ULLONG [number];

	// XXX: this array will have to be initialized in writeHeader()
	// (and no, there isn't much we can do about it. Maybe we could
	// implement setVariableType() and modify the array every time that
	// function is called. It's ugly either way.)
}

//! Returns true if the file given by fileName is of the correct type.
bool RawVFileImpl::checkType(const QString& fileName)
{
	// go through the file and make sure we understand it
	QFileInfo fileInfo(fileName);

	// first check to make sure it exists
	if (!fileInfo.exists()) {
		qDebug("File does not exist");
		return false;
	}

	QString absFilePath = fileInfo.absFilePath();

#ifdef LARGEFILE_KLUDGE
	PFile file(absFilePath);
#else
	QFile file(absFilePath);
#endif
	if (!file.open(IO_ReadOnly | IO_Raw)) {
		qDebug("Error opening file");
		return false;
	}
	
	float floatVals[4];
	unsigned int uintVals[3];
	unsigned int dimX, dimY, dimZ;
	unsigned int numVars, numTimeSteps;
	//float minX, minY, minZ, minT;
	//float maxX, maxY, maxZ, maxT;

	// read in the magic value
	file.readBlock((char*)uintVals, sizeof(unsigned int)*1);
	if (isLittleEndian()) swapByteOrder(uintVals, 1);
	if (uintVals[0] != 0xBAADBEEF) {
		return false;
	}

	// read in the dimensions
	file.readBlock((char*)uintVals, sizeof(unsigned int)*3);
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	dimX = uintVals[0];
	dimY = uintVals[1];
	dimZ = uintVals[2];

	// read in the number of vars and timesteps
	file.readBlock((char*)uintVals, sizeof(unsigned int)*2);
	if (isLittleEndian()) swapByteOrder(uintVals, 2);
	numTimeSteps = uintVals[0];
	numVars = uintVals[1];

	// read in the mins
	file.readBlock((char*)floatVals, sizeof(float)*4);
	// ignore them for now
	//minX = Flip(floatVals[0]);
	//minY = Flip(floatVals[1]);
	//minZ = Flip(floatVals[2]);
	//minT = Flip(floatVals[3]);

	// read in the maxs
	file.readBlock((char*)floatVals, sizeof(float)*4);
	// ignore them for now
	//maxX = Flip(floatVals[0]);
	//maxY = Flip(floatVals[1]);
	//maxZ = Flip(floatVals[2]);
	//maxT = Flip(floatVals[3]);

	// read in the var names and types
	unsigned int v;
	unsigned char vartype;
	char varname[64];
	unsigned int sizes[] = {0, 1, 2, 4, 4, 8};

	// the fixed part of the header is 56 bytes
	Q_ULLONG totalDataSize = 56;

	for (v=0; v<numVars; v++) {
		file.readBlock((char*)&vartype, sizeof(unsigned char)*1);
		if (vartype<1 || vartype>5) { // invalid type
			return false;
		}
		else {
			// the part of the header for each variable is 65 bytes
			totalDataSize += 65 + sizes[vartype]*dimX*dimY*dimZ*numTimeSteps;
		}
		file.readBlock((char*)varname, sizeof(char)*64);
	}

	// check the fileSize
	if (totalDataSize != (Q_ULLONG)file.size()) {
		// the size does not match the header information
		qDebug("size does not match size derived from header");
		return false;
	}

	// everything checks out
	return true;
}

//! Associates this reader with the file given by fileName.
bool RawVFileImpl::attachToFile(const QString& fileName, Mode mode)
{
	QFileInfo fileInfo(fileName);
	QString absFilePath = fileInfo.absFilePath();
//	off_t filesize;

	// remember the mode
	m_OpenMode = mode;

	if (mode == Read) {
		// first check to make sure it exists
		if (!fileInfo.exists()) {
			qDebug("File does not exist");
			return false;
		}
	
		QString absFilePath = fileInfo.absFilePath();
		
		m_File.setName(absFilePath);
		if (!m_File.open(IO_ReadOnly | IO_Raw)) {
			qDebug("Error opening file");
			return false;
		}

		if (readHeader(m_File.size())) {
			// success
			m_Attached = true;
			return true;
		}
		else {
			// failure
			close();
			return false;
		}
	}
	else { // mode == Write
		m_File.setName(absFilePath);
		if (!m_File.open(IO_WriteOnly | IO_Raw)) {
			qDebug("Error opening file");
			return false;
		}

		// success
		m_Attached = true;
	}

	return m_Attached;
}

//! Reads char data from the file into the supplied buffer
bool RawVFileImpl::readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
{
	// fail if trying to read from a write only instance
	if (m_OpenMode == Write)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Char) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)numSamples != m_File.readBlock((char*)buffer, numSamples)) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Reads short data from the file into the supplied buffer
bool RawVFileImpl::readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
	// fail if trying to read from a write only instance
	if (m_OpenMode == Write)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Short) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(short)) != m_File.readBlock((char*)buffer, numSamples*sizeof(short))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads long data from the file into the supplied buffer
bool RawVFileImpl::readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
	// fail if trying to read from a write only instance
	if (m_OpenMode == Write)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Long) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(unsigned int)) != m_File.readBlock((char*)buffer, numSamples*sizeof(unsigned int))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads float data from the file into the supplied buffer
bool RawVFileImpl::readFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	// fail if trying to read from a write only instance
	if (m_OpenMode == Write)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Float) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(float)) != m_File.readBlock((char*)buffer, numSamples*sizeof(float))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads double data from the file into the supplied buffer
bool RawVFileImpl::readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
	// fail if trying to read from a write only instance
	if (m_OpenMode == Write)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Double) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(double)) != m_File.readBlock((char*)buffer, numSamples*sizeof(double))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Writes the header to the file
bool RawVFileImpl::writeHeader()
{
	unsigned int uintVals[4], i;
	float floatVals[4];
	// XXX: Note that a variable critical to the function of all the
	// write*Data() functions is initialized by this functions.
	// (it's m_VariableStartingLocations in case you were wondering)
	Q_ULLONG totalDataSize = (Q_ULLONG)headerSize(m_NumVariables);
	unsigned int sizes[] = {0, 1, 2, 4, 4, 8};

	// write out the magic value
	uintVals[0] = 0xBAADBEEF;
	if (isLittleEndian()) swapByteOrder(uintVals, 1);
	m_File.writeBlock((char *)uintVals, sizeof(unsigned int));

	// now the dimensions
	uintVals[0] = m_DimX;
	uintVals[1] = m_DimY;
	uintVals[2] = m_DimZ;
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	m_File.writeBlock((char *)uintVals, 3*sizeof(unsigned int));

	// # timesteps, # variables
	uintVals[0] = m_NumTimeSteps;
	uintVals[1] = m_NumVariables;
	if (isLittleEndian()) swapByteOrder(uintVals, 2);
	m_File.writeBlock((char *)uintVals, 2*sizeof(unsigned int));

	// min extent
	floatVals[0] = m_MinX;
	floatVals[1] = m_MinY;
	floatVals[2] = m_MinZ;
	floatVals[3] = m_MinT;
	if (isLittleEndian()) swapByteOrder(floatVals, 4);
	m_File.writeBlock((char *)floatVals, 4*sizeof(float));

	// max extent
	floatVals[0] = m_MaxX;
	floatVals[1] = m_MaxY;
	floatVals[2] = m_MaxZ;
	floatVals[3] = m_MaxT;
	if (isLittleEndian()) swapByteOrder(floatVals, 4);
	m_File.writeBlock((char *)floatVals, 4*sizeof(float));

	// variable types and names
	for (i=0; i < m_NumVariables; i++) {
		unsigned char type;
		char name[64];
	 
		switch (m_VariableTypes[i]) {
			case Char:
				type = (unsigned char)1;
				break;
			case Short:
				type = (unsigned char)2;
				break;
			case Long:
				type = (unsigned char)3;
				break;
			case Float:
				type = (unsigned char)4;
				break;
			case Double:
				type = (unsigned char)5;
				break;
			default:
				// we should not get here
				type = (unsigned char)1;
				break;
		}
		
		// init the m_VariableStartingLocations array
		m_VariableStartingLocations[i] = totalDataSize;
		totalDataSize += sizes[type]*m_DimX*m_DimY*m_DimZ*m_NumTimeSteps;

		strncpy(name, m_VariableNames[i].ascii(), 63);
		name[63] = '\0';

		m_File.writeBlock((char *)&type, sizeof(unsigned char));
		m_File.writeBlock(name, 64*sizeof(char));
	}

	return true;
}

//! Writes char data to the file from the supplied buffer
bool RawVFileImpl::writeCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Char) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// write out the data
	if ((int)numSamples != m_File.writeBlock((char*)buffer, numSamples)) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes short data to the file from the supplied buffer
bool RawVFileImpl::writeShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Short) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// swap
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	// write out the data
	if ((int)numSamples*sizeof(unsigned short) != m_File.writeBlock((char*)buffer, numSamples*sizeof(unsigned short))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes long data to the file from the supplied buffer
bool RawVFileImpl::writeLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Long) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// swap
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	// write out the data
	if ((int)numSamples*sizeof(unsigned int) != m_File.writeBlock((char*)buffer, numSamples*sizeof(unsigned int))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes float data to the file from the supplied buffer
bool RawVFileImpl::writeFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Float) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// swap
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	// write out the data
	if ((int)numSamples*sizeof(float) != m_File.writeBlock((char*)buffer, numSamples*sizeof(float))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes double data to the file from the supplied buffer
bool RawVFileImpl::writeDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Double) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// swap
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	// write out the data
	if ((int)numSamples*sizeof(double) != m_File.writeBlock((char*)buffer, numSamples*sizeof(double))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Returns a new loader which can be used to load a file
VolumeFile* RawVFileImpl::getNewVolumeFileLoader() const
{
	return new RawVFileImpl;
}

//! Sets the position of the next read.
bool RawVFileImpl::protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset)
{
	if (variable>=m_NumVariables || timeStep>=m_NumTimeSteps) {
		return false;
	}
	else {
		m_File.at(m_VariableStartingLocations[variable] + (m_DimX*m_DimY*m_DimZ*timeStep + offset)*getBytesPerPixel(variable));
		return true;
	}
}

bool RawVFileImpl::readHeader(Q_ULLONG fileSize)
{
	float floatVals[4];
	unsigned int uintVals[3];

	// read in the magic value
	m_File.readBlock((char*)uintVals, sizeof(unsigned int)*1);
	if (isLittleEndian()) swapByteOrder(uintVals, 1);
	if (uintVals[0] != 0xBAADBEEF) {
		return false;
	}

	// read in the dimensions
	m_File.readBlock((char*)uintVals, sizeof(unsigned int)*3);
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	m_DimX = uintVals[0];
	m_DimY = uintVals[1];
	m_DimZ = uintVals[2];

	// read in the number of vars and timesteps
	m_File.readBlock((char*)uintVals, sizeof(unsigned int)*2);
	if (isLittleEndian()) swapByteOrder(uintVals, 2);
	m_NumTimeSteps= uintVals[0];
	unsigned int numVars = uintVals[1];
	makeSpaceForVariablesInfo(numVars);
	m_VariableStartingLocations = new Q_ULLONG[numVars];

	// read in the mins
	m_File.readBlock((char*)floatVals, sizeof(float)*4);
	if (isLittleEndian()) swapByteOrder(floatVals, 4);
	m_MinX = floatVals[0];
	m_MinY = floatVals[1];
	m_MinZ = floatVals[2];
	m_MinT = floatVals[3];

	// read in the maxs
	m_File.readBlock((char*)floatVals, sizeof(float)*4);
	if (isLittleEndian()) swapByteOrder(floatVals, 4);
	m_MaxX = floatVals[0];
	m_MaxY = floatVals[1];
	m_MaxZ = floatVals[2];
	m_MaxT = floatVals[3];

	// read in the var names and types
	unsigned int v;
	unsigned char vartype;
	char varname[64];
	unsigned int sizes[] = {0, 1, 2, 4, 4, 8};
	Type types[] = {Char, Char, Short, Long, Float, Double};

	// start at the end of the header
	Q_ULLONG totalDataSize = (Q_ULLONG)headerSize(m_NumVariables);

	for (v=0; v<numVars; v++) {
		m_VariableStartingLocations[v] = totalDataSize;
		m_File.readBlock((char*)&vartype, sizeof(unsigned char)*1);
		if (vartype<1 || vartype>5) { // invalid type
			return false;
		}
		else {
			totalDataSize += sizes[vartype]*m_DimX*m_DimY*m_DimZ*m_NumTimeSteps;
		}
		//XXX check for null character
		m_File.readBlock((char*)varname, sizeof(char)*64);
		m_VariableNames[v] = varname;
		m_VariableTypes[v] = types[vartype];
	}

	// check the fileSize
	if (totalDataSize != m_File.size()) {
		// the size does not match the header information
		return false;
	}

	// everything checks out
	return true;
}

void RawVFileImpl::close()
{
	if (m_Attached) {
		m_File.close();
		m_Attached = false;
		delete [] m_VariableStartingLocations;
		m_VariableStartingLocations = 0;
	}
}

unsigned int RawVFileImpl::headerSize(unsigned int numVariables) const
{
	return 56+65*numVariables;
}

