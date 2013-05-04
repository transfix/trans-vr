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

// RawIVFileImpl.cpp: implementation of the RawIVFileImpl class.
//
//////////////////////////////////////////////////////////////////////
#include <qfileinfo.h>
#include <qfile.h>
#include <VolumeFileTypes/RawIVFileImpl.h>
#include <ByteOrder/ByteSwapping.h>
#include <math.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RawIVFileImpl::RawIVFileImpl()
{
	m_Attached = false;
}

RawIVFileImpl::~RawIVFileImpl()
{

}

//! Returns true if the file given by fileName is of the correct type.
bool RawIVFileImpl::checkType(const QString& fileName)
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
	
	float floatVals[3];
	unsigned int uintVals[3];

	unsigned int dimX, dimY, dimZ;
	float minX, minY, minZ;
	float maxX, maxY, maxZ;
	float spanX, spanY, spanZ;

	// read in the mins
	file.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	minX = floatVals[0];
	minY = floatVals[1];
	minZ = floatVals[2];

	// read in the maxs
	file.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	maxX = floatVals[0];
	maxY = floatVals[1];
	maxZ = floatVals[2];

	// ignore num verts and num cells, redundant
	file.readBlock((char*)uintVals, sizeof(unsigned int)*2);

	// read in the dimensions
	file.readBlock((char*)uintVals, sizeof(unsigned int)*3);
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	dimX = uintVals[0];
	dimY = uintVals[1];
	dimZ = uintVals[2];

	// ignore the "origin"  not sure what it means...probably redundant
	file.readBlock((char*)floatVals, sizeof(float)*3);

	// read in the span...redundant but will use it to check min and max
	file.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	spanX = floatVals[0];
	spanY = floatVals[1];
	spanZ = floatVals[2];

	// check min + span*dim == max to make sure file is consistant
	if ( !(
		fabs(minX + floatVals[0]*(double)(dimX-1) - maxX) < 0.0001 &&
		fabs(minY + floatVals[1]*(double)(dimY-1) - maxY) < 0.0001 &&
		fabs(minZ + floatVals[2]*(double)(dimZ-1) - maxZ) < 0.0001)) {

		// inconsistant, junk it all and replace with simple defaults
		minX = 0; minY = 0; minZ = 0;
		maxX = minX + (double)(dimX-1)*spanX;
		maxY = minY + (double)(dimY-1)*spanY;
		maxZ = minZ + (double)(dimZ-1)*spanZ;

	}
	
	Q_ULLONG numVerts = (Q_ULLONG)dimX*dimY*dimZ;
	//qDebug("dims: %d x %d x %d", dimX, dimY, dimZ);
	//qDebug("numVerts = %d, filesize-68 = %d", numVerts, fileInfo.size()-68);
	if (numVerts == file.size()-68) {
		// success; byte datatype
		return true;
	}
	else if (numVerts*2 == file.size()-68) {
		// success; short datatype
		return true;
	}
	else if (numVerts*4 == file.size()-68) {
		// success; float datatype
		return true;
	}
	else {
		// not byte short or float
		return false;
	}
}

//! Associates this reader with the file given by fileName.
bool RawIVFileImpl::attachToFile(const QString& fileName, Mode mode)
{
	QFileInfo fileInfo(fileName);
	QString absFilePath = fileInfo.absFilePath();

	m_OpenMode = mode;

	if (mode == Read) {
		// first check to make sure it exists
		if (!fileInfo.exists()) {
			qDebug("File does not exist");
			return false;
		}
	
		//QString absFilePath = fileInfo.absFilePath();
	
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
		// open the file
		m_File.setName(absFilePath);
		if (!m_File.open(IO_WriteOnly | IO_Raw)) {
			qDebug("Error opening file");
			return false;
		}

		m_Attached = true;

		return true;
	}
}

//! Sets the type for the given variable v
void RawIVFileImpl::setVariableType(unsigned int variable, Type type)
{
	if (type != Long && type != Double)
		BasicVolumeFileImpl::setVariableType(variable, type);
	else {
		qDebug("Error: Trying to set variable type for a rawiv file to \
Long or Double is not allowed. Defaulting to Char.");
		BasicVolumeFileImpl::setVariableType(variable, Char);
	}
}

//! Reads char data from the file into the supplied buffer
bool RawIVFileImpl::readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
{
	// don't try to read if opened for writing
	if (m_OpenMode == Write) {
		return false;
	}


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
bool RawIVFileImpl::readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
	// don't try to read if opened for writing
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
bool RawIVFileImpl::readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
	// rawiv doesn't do long
	return false;
}

//! Reads float data from the file into the supplied buffer
bool RawIVFileImpl::readFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	// don't try to read if opened for writing
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
bool RawIVFileImpl::readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
	// rawiv doesn't do double
	return false;
}

//! Writes char data to the file from the supplied buffer
bool RawIVFileImpl::writeCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
{
	// don't try to write if opened for reading
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
bool RawIVFileImpl::writeShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
	// don't try to write if opened for reading
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
	if (((int)numSamples*sizeof(unsigned short)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(unsigned short))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes long data to the file from the supplied buffer
bool RawIVFileImpl::writeLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
	// rawiv doesn't support long typed data
	return false;
}

//! Writes float data to the file from the supplied buffer
bool RawIVFileImpl::writeFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	// don't try to write if opened for reading
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
	if (((int)numSamples*sizeof(float)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(float))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes double data to the file from the supplied buffer
bool RawIVFileImpl::writeDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
	// rawiv doesn't support double typed data
	return false;
}


//! Returns a new loader which can be used to load a file
VolumeFile* RawIVFileImpl::getNewVolumeFileLoader() const
{
	return new RawIVFileImpl;
}

//! Sets the position of the next read.
bool RawIVFileImpl::protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset)
{
	if (variable>0||timeStep>0) {
		return false;
	}
	else {
		m_File.at((Q_ULLONG)(68 + offset*bytesPerVoxel()));
		return true;
	}
}

bool RawIVFileImpl::readHeader(Q_ULLONG fileSize)
{
	float floatVals[3];
	unsigned int uintVals[3];

	// read in the mins
	m_File.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	m_MinX = floatVals[0];
	m_MinY = floatVals[1];
	m_MinZ = floatVals[2];

	// read in the maxs
	m_File.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	m_MaxX = floatVals[0];
	m_MaxY = floatVals[1];
	m_MaxZ = floatVals[2];

	// ignore num verts and num cells, redundant
	m_File.readBlock((char*)uintVals, sizeof(unsigned int)*2);

	// read in the dimensions
	m_File.readBlock((char*)uintVals, sizeof(unsigned int)*3);
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	m_DimX = uintVals[0];
	m_DimY = uintVals[1];
	m_DimZ = uintVals[2];

	// ignore the "origin"  not sure what it means...probably redundant
	m_File.readBlock((char*)floatVals, sizeof(float)*3);

	// read in the span...redundant but will use it to check min and max
	double spanX, spanY, spanZ;
	m_File.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	spanX = floatVals[0];
	spanY = floatVals[1];
	spanZ = floatVals[2];

	// check min + span*dim == max to make sure file is consistant
	if ( !(
		fabs(m_MinX + floatVals[0]*(double)(m_DimX-1) - m_MaxX) < 0.0001 &&
		fabs(m_MinY + floatVals[1]*(double)(m_DimY-1) - m_MaxY) < 0.0001 &&
		fabs(m_MinZ + floatVals[2]*(double)(m_DimZ-1) - m_MaxZ) < 0.0001)) {

		// inconsistant, junk it all and replace with simple defaults
		m_MinX = 0; m_MinY = 0; m_MinZ = 0;
		m_MaxX = m_MinX + (double)(m_DimX-1)*spanX;
		m_MaxY = m_MinY + (double)(m_DimY-1)*spanY;
		m_MaxZ = m_MinZ + (double)(m_DimZ-1)*spanZ;

	}

	m_MinT = 0.0;
	m_MaxT = 0.0;
	m_NumTimeSteps = 1;
	makeSpaceForVariablesInfo(1);
	m_VariableNames[0] = "No Name";
	
	Q_ULLONG numVerts = (Q_ULLONG)m_DimX*m_DimY*m_DimZ;
	if (numVerts == fileSize-68) {
		// success; byte datatype
		m_VariableTypes[0] = Char;
		return true;
	}
	else if (numVerts*2 == fileSize-68) {
		// success; short datatype
		m_VariableTypes[0] = Short;
		return true;
	}
	else if (numVerts*4 == fileSize-68) {
		// success; float datatype
		m_VariableTypes[0] = Float;
		return true;
	}
	else {
		// not byte, short or float
		return false;
	}
}

bool RawIVFileImpl::writeHeader()
{
	float floatVals[3];
	unsigned int uintVals[3];
	
	// don't try to write if opened for reading
	if (m_OpenMode == Read)
		return false;

	// seek to the start of the file
	BasicVolumeFileImpl::setPosition(0,0,0); // sets the various VolumeFile related pointers
	m_File.at((Q_ULLONG)0); // puts us at the actual beginning of the file.

	// write out the mins
	floatVals[0] = m_MinX;
	floatVals[1] = m_MinY;
	floatVals[2] = m_MinZ;
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	if (sizeof(float)*3 != m_File.writeBlock((char*)floatVals, sizeof(float)*3))
		return false;

	// write out the maxs
	floatVals[0] = m_MaxX;
	floatVals[1] = m_MaxY;
	floatVals[2] = m_MaxZ;
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	if (sizeof(float)*3 != m_File.writeBlock((char*)floatVals, sizeof(float)*3))
		return false;

	// num verts and num cells
	uintVals[0] = m_DimX * m_DimY * m_DimZ;
	uintVals[1] = (m_DimX-1) * (m_DimY-1) * (m_DimZ-1);
	if (isLittleEndian()) swapByteOrder(uintVals, 2);
	if (sizeof(unsigned int)*2 != m_File.writeBlock((char*)uintVals, sizeof(unsigned int)*2))
		return false;

	// write out the dimensions
	uintVals[0] = m_DimX;
	uintVals[1] = m_DimY;
	uintVals[2] = m_DimZ;
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	if (sizeof(unsigned int)*3 != m_File.writeBlock((char*)uintVals, sizeof(unsigned int)*3))
		return false;

	// write out the min extent as the "origin"
	floatVals[0] = m_MinX;
	floatVals[1] = m_MinY;
	floatVals[2] = m_MinZ;
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	if (sizeof(float)*3 != m_File.writeBlock((char*)floatVals, sizeof(float)*3))
		return false;

	// write out the span
	floatVals[0] = (m_MaxX - m_MinX) / (m_DimX-1);
	floatVals[1] = (m_MaxY - m_MinY) / (m_DimY-1);
	floatVals[2] = (m_MaxZ - m_MinZ) / (m_DimZ-1);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	if (sizeof(float)*3 != m_File.writeBlock((char*)floatVals, sizeof(float)*3))
		return false;

	return true;
}


void RawIVFileImpl::setPosition(Q_ULLONG position)
{
	if (position!=m_CurrentPosition) {
		m_File.at((Q_ULLONG)68 + position*bytesPerVoxel());
	}
	m_CurrentPosition = position;
}

void RawIVFileImpl::close()
{
	if (m_Attached) {
		m_File.close();
		m_Attached = false;
	}
}

unsigned int RawIVFileImpl::bytesPerVoxel() const
{
	unsigned int bpv;
	if (m_VariableTypes[0] == Char) {
		bpv = 1;
	}
	else if (m_VariableTypes[0] == Short) {
		bpv = 2;
	}
	else {
		bpv = 4;
	}
	return bpv;
}

