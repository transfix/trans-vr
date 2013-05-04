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

// MrcFileImpl.cpp: implementation of the MrcFileImpl class.
//
//////////////////////////////////////////////////////////////////////
#include <qfileinfo.h>
#include <VolumeFileTypes/MrcFileImpl.h>
#include <ByteOrder/ByteSwapping.h>
#include <string.h>
#include <math.h>

#ifdef SOLARIS
#include <ieeefp.h>
#endif

// XXX: This is UGLY. Windows does not have this function in its math library.
// This function is only called from the interpretXXXHeader functions
#ifdef WIN32
int finite(float)
{
	return 0;
}
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

void printHeader(MrcHeader &header)
{
	printf("dims: %dx%dx%d\n", header.nx, header.ny, header.nz);
	printf("mode: %d\n", header.mode);
	printf("n(x/y/z)start: %d/%d/%d\n",header.nxstart,header.nystart,header.nzstart);
	
	printf("mx:%d, my:%d, mz:%d\n", header.mx,header.my,header.mz);
	printf("(x/y/z)length: %f/%f/%f\n", header.xlength,header.ylength,header.zlength);
	printf("alpha:%f, beta:%f, gamma:%f\n", header.alpha,header.beta,header.gamma);
	printf("map(c/r/s): %d/%d/%d\n", header.mapc,header.mapr,header.maps);
	
	printf("min: %f\n", header.amin);
	printf("max: %f\n", header.amax);
	printf("mean: %f\n", header.amean);
	printf("ispg = %d\n", header.ispg);
	printf("nsumbt = %d\n", header.nsymbt);
	printf("origin: (%f,%f,%f)\n", header.xorigin,header.yorigin,header.zorigin);
	printf("rms = %f\n", header.rms);
	printf("nlabl = %d\n", header.nlabl);
}

MrcFileImpl::MrcFileImpl()
{
	m_Attached = false;
	m_MustSwap = false;
}

MrcFileImpl::~MrcFileImpl()
{
	close();
}

//! Returns true if the file given by fileName is of the correct type.
bool MrcFileImpl::checkType(const QString& fileName)
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
	if (!file.open(QIODevice::ReadOnly | QIODevice::Unbuffered)) {
		qDebug("Error opening file");
		return false;
	}

	qDebug("MrcFileImpl::checkType()");

	// read the header
	MrcHeader header;
	ExtendedMrcHeader extheader;
	file.readBlock((char*)&header, sizeof(MrcHeader));

	if (!(header.map[0]=='M' && header.map[1]=='A' && header.map[2]=='P')) {
		// not the new MRC style, we must try to guess the
		// endianness
		
		qDebug("old style mrc file");
		if(header.nsymbt > 0)
		  file.readBlock((char*)&extheader, sizeof(ExtendedMrcHeader));

		// first try not swapping
		if (checkHeader(header, file.size())) {
			//printHeader(header);
			return true;
		}
		else {
			// swap and try again
			swapHeader(header);
			if (checkHeader(header, file.size())) {
				//printHeader(header);
				return true;
			}
			else {
				// we dont support wierd or exotic endianness
				return false;
			}
		}
	}

	// XXX: After seeing two different versions of the little endian machine
	// stamp in two weeks (both of which weren't the one Rover was using), I
	// decided that the machine stamp is totally useless. The code now checks
	// the header and swaps if the check fails. This works just fine on all the
	// files I have tested it on.

	// (Give me rawiv any day. All this byte order indecision is a headache.)

	/*qDebug("machine stamp before swap: %x", header.machst);
	// swap the machine stamp
	if (isLittleEndian()) swapByteOrder(header.machst);
	qDebug("machine stamp after swap: %x", header.machst);

	if (!(header.machst == 0x11110000 || header.machst == 0x44410000
																		|| header.machst == 0x44440000)) {
		// unknown machine stamp
		qDebug("unknown machine stamp");
		return false;
	}

	bool mustSwap = (isLittleEndian() && header.machst == 0x11110000) ||
		(!isLittleEndian()
		 && (header.machst == 0x44410000 || header.machst == 0x44440000));
	//bool mustSwap = (!isLittleEndian() && header.machst == 0x11110000) ||
	//	(isLittleEndian() && header.machst == 0x44410000);
	
	//qDebug("mustSwap = %s", mustSwap?"true":"false");
	// swap the 56, 4-byte, values
	//printHeader(header);
	if (mustSwap) swapHeader(header);*/

	if(header.nsymbt > 0)
	  file.readBlock((char*)&extheader, sizeof(ExtendedMrcHeader));

	// Nobody seems to agree about the meaning of the machine stamp,
	// so swap again if the header doesn't check out.
	if (!checkHeader(header, file.size()))
		swapHeader(header);
	
	//printHeader(header);

	return checkHeader(header, file.size());
}

//! Associates this reader with the file given by fileName.
bool MrcFileImpl::attachToFile(const QString& fileName, Mode mode)
{
	QFileInfo fileInfo(fileName);

	// set the mode
	m_OpenMode = mode;

	if (mode == Read) {
		// first check to make sure it exists
		if (!fileInfo.exists()) {
			qDebug("File does not exist");
			return false;
		}

		QString absFilePath = fileInfo.absFilePath();

		m_File.setName(absFilePath);
		if (!m_File.open(QIODevice::ReadOnly | QIODevice::Unbuffered)) {
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
		QString absFilePath = fileInfo.absFilePath();

		m_File.setName(absFilePath);
		if (!m_File.open(QIODevice::WriteOnly | QIODevice::Unbuffered)) {
			qDebug("Error opening file");
			return false;
		}

		m_Attached = true;
		if (isBigEndian())
			m_MustSwap = true;
	}

	return m_Attached;
}

//! Sets the type for the given variable v
void MrcFileImpl::setVariableType(unsigned int variable, Type type)
{
	if (type != Long && type != Double)
		BasicVolumeFileImpl::setVariableType(variable, type);
	else {
		qDebug("Error: Trying to set variable type for a mrc file to \
Long or Double is not allowed. Defaulting to Char.");
		BasicVolumeFileImpl::setVariableType(variable, Char);
	}
}

//! Reads char data from the file into the supplied buffer
bool MrcFileImpl::readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
{
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
bool MrcFileImpl::readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
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
	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads long data from the file into the supplied buffer
bool MrcFileImpl::readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
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
	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads float data from the file into the supplied buffer
bool MrcFileImpl::readFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
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
	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads double data from the file into the supplied buffer
bool MrcFileImpl::readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
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
	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Writes the header to the file
bool MrcFileImpl::writeHeader()
{
	MrcHeader header;

	// fail if this file is open for reading
	if (m_OpenMode == Read)
		return false;

	// fill in the header
	fillHeader(header);

	if (m_MustSwap) swapHeader(header);

	// write it out, return false if the write fails
	return (sizeof(MrcHeader) == m_File.writeBlock((char *)&header, sizeof(MrcHeader)));
}

//! Writes char data to the file from the supplied buffer
bool MrcFileImpl::writeCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
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
bool MrcFileImpl::writeShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Short) {
		// no good
		return false;
	}

	prepareToRead(variable);

	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	// write out the data
	if ((numSamples*sizeof(unsigned short)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(unsigned short))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes long data to the file from the supplied buffer
bool MrcFileImpl::writeLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Long) {
		// no good
		return false;
	}

	prepareToRead(variable);

	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	// write out the data
	if (((int)numSamples*sizeof(unsigned int)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(unsigned int))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes float data to the file from the supplied buffer
bool MrcFileImpl::writeFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Float) {
		// no good
		return false;
	}

	prepareToRead(variable);

	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	// write out the data
	if (((int)numSamples*sizeof(float)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(float))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes double data to the file from the supplied buffer
bool MrcFileImpl::writeDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Double) {
		// no good
		return false;
	}

	prepareToRead(variable);

	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	// write out the data
	if (((int)numSamples*sizeof(double)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(double))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Returns a new loader which can be used to load a file
VolumeFile* MrcFileImpl::getNewVolumeFileLoader() const
{
	return new MrcFileImpl;
}

//! Sets the position of the next read.
bool MrcFileImpl::protectedSetPosition(unsigned int variable, unsigned int timeStep, qulonglong offset)
{
	if (variable>=m_NumVariables || timeStep>=m_NumTimeSteps) {
		return false;
	}
	else {
		m_File.at(1024 + (m_DimX*m_DimY*m_DimZ*timeStep + offset)*getBytesPerPixel(variable));
		return true;
	}
}

bool MrcFileImpl::readHeader(qulonglong fileSize)
{
	qDebug("MrcFileImpl::readHeader()");
	// read the header
	MrcHeader header;
	ExtendedMrcHeader extheader;
	m_File.readBlock((char*)&header, sizeof(MrcHeader));

	if (!(header.map[0]=='M' && header.map[1]=='A' && header.map[2]=='P')) {
		// not the new MRC style, we must try to guess the
		// endianness

	  if(header.nsymbt > 0)
	    m_File.readBlock((char*)&extheader, sizeof(ExtendedMrcHeader));

		// first try not swapping
		if (checkHeader(header, fileSize)) {
			m_MustSwap = false;
			return interpretOldHeader(header, fileSize);;
		}
		else {
			// swap and try again
			swapHeader(header);
			if (checkHeader(header, fileSize)) {
				m_MustSwap = true;
				return interpretOldHeader(header, fileSize);;
			}
			else {
				// we dont support wierd or exotic endianness
				return false;
			}
		}
	}

	// new MRC file format
	

	// swap the machine stamp
	/*if (isLittleEndian()) swapByteOrder(header.machst);

	if (!(header.machst == 0x11110000 || header.machst == 0x44410000
																		|| header.machst == 0x44440000)) {
		// unknown machine stamp
		qDebug("unknown machine stamp");
		return false;
	}

	m_MustSwap = (isLittleEndian() && header.machst == 0x11110000) ||
		(!isLittleEndian()
		 && (header.machst == 0x44410000 || header.machst == 0x44440000));
	
	// swap the 56, 4-byte, values
	if (m_MustSwap) swapHeader(header);*/

	if(header.nsymbt > 0)
	  m_File.readBlock((char*)&extheader, sizeof(ExtendedMrcHeader));

	// Nobody seems to agree about the meaning of the machine stamp,
	// so swap again if the header doesn't check out.
	if (!checkHeader(header, fileSize)) {
		// we change our mind about swapping
		//m_MustSwap = !m_MustSwap;
		m_MustSwap = true;
		// and swap the header again
		swapHeader(header);
	}
	
	//m_MustSwap = (isLittleEndian() && header.machst == 0x11110000) ||
	//	(!isLittleEndian() && header.machst == 0x44410000);
	//m_MustSwap = false;
	
	// swap the 56, 4-byte, values
	//if (m_MustSwap) swapHeader(header);

	if (!checkHeader(header, fileSize)) {
		return false;
	}


	return interpretNewHeader(header, fileSize);
}

bool MrcFileImpl::checkHeader(MrcHeader& header, qulonglong fileSize)
{
	qulonglong sizes[] = {1, 2, 4};

	//qDebug("MrcFileImpl::checkHeader()");

	// check for the details we dont support
	if (header.mode<0 || header.mode>2) {
		// we dont support this type or MRC file for now		
		qDebug("unsupported mrc file. (mode = %x)", header.mode);
		return false;
	}

	// check the fileSize
	if(header.nsymbt == 0)
	  {
	    if (sizes[header.mode]*header.nx*header.ny*header.nz + 1024 != fileSize) {
	      // the size does not match the header information
	      qDebug("bad mrc file? (file size != size given in header)");
	      return false;
	    }
	  }
	else
	  {
	    if (sizes[header.mode]*header.nx*header.ny*header.nz + sizeof(MrcHeader) + sizeof(ExtendedMrcHeader) != fileSize) {
	      // the size does not match the header information
	      qDebug("bad mrc file? (file size != size given in header)");
	      return false;
	    }
	  }

	// everything checks out, return true
	return true;
}

bool MrcFileImpl::interpretNewHeader(MrcHeader& header, qulonglong fileSize)
{
	Type types[] = {Char, Short, Float};
	//unsigned int sizes[] = {1, 2, 4};

	m_DimX = header.nx;
	m_DimY = header.ny;
	m_DimZ = header.nz;
	m_NumTimeSteps = 1;
	makeSpaceForVariablesInfo(1);
	m_VariableNames[0] = "No Name";
	m_VariableTypes[0] = types[header.mode];
	// make sure we aren't using garbage values
	if (!finite(header.xorigin) || !finite(header.yorigin) || !finite(header.zorigin)) {
		m_MinX = 0.0;
		m_MinY = 0.0;
		m_MinZ = 0.0;
	}
	else {
	m_MinX = header.xorigin;
	m_MinY = header.yorigin;
	m_MinZ = header.zorigin;
	}
	m_MinT = 0.0;

	// we need to double check the meaning of xlength
	// (plus some extra paranoia)
	if (header.xlength<=0.0 || header.ylength<=0.0 || header.zlength<=0.0
		|| !finite(header.xlength) || !finite(header.ylength) || !finite(header.zlength)) {
		// hmm, this is wierd
		m_MaxX = m_MinX + m_DimX*1.0;
		m_MaxY = m_MinY + m_DimY*1.0;
		m_MaxZ = m_MinZ + m_DimZ*1.0;
	}
	else {
		m_MaxX = m_MinX + header.xlength;
		m_MaxY = m_MinY + header.ylength;
		m_MaxZ = m_MinZ + header.zlength;
	}
	m_MaxT = 0.0;

	// everything checks out, return true
	return true;
}

bool MrcFileImpl::interpretOldHeader(MrcHeader& header, qulonglong fileSize)
{
	Type types[] = {Char, Short, Float};
	//unsigned int sizes[] = {1, 2, 4};

	m_DimX = header.nx;
	m_DimY = header.ny;
	m_DimZ = header.nz;
	m_NumTimeSteps = 1;
	makeSpaceForVariablesInfo(1);
	m_VariableNames[0] = "No Name";
	m_VariableTypes[0] = types[header.mode];
	m_MinX = 0.0;
	m_MinY = 0.0;
	m_MinZ = 0.0;
	m_MinT = 0.0;

	// we need to double check the meaning of xlength
	// (plus some extra paranoia)
	if (header.xlength<=0.0 || header.ylength<=0.0 || header.zlength<=0.0
		|| !finite(header.xlength) || !finite(header.ylength) || !finite(header.zlength)) {
		// hmm, this is wierd
		m_MaxX = m_MinX + m_DimX*1.0;
		m_MaxY = m_MinY + m_DimY*1.0;
		m_MaxZ = m_MinZ + m_DimZ*1.0;
	}
	else {
		m_MaxX = m_MinX + header.xlength;
		m_MaxY = m_MinY + header.ylength;
		m_MaxZ = m_MinZ + header.zlength;
	}
	m_MaxT = 0.0;

	// everything checks out, return true
	return true;
}

void MrcFileImpl::swapHeader(MrcHeader& header)
{
	//qDebug("swap header");
	swapByteOrder(header.nx);
	swapByteOrder(header.ny);
	
	swapByteOrder(header.nz);
	
	//qDebug("header.mode = %x", header.mode);
	swapByteOrder(header.mode);
	//qDebug("header.mode = %x", header.mode);
	
	swapByteOrder(header.nxstart);
	swapByteOrder(header.nystart);
	swapByteOrder(header.nzstart);
	
	swapByteOrder(header.mx);
	swapByteOrder(header.my);
	swapByteOrder(header.mz);
	
	swapByteOrder(header.xlength);
	swapByteOrder(header.ylength);
	swapByteOrder(header.zlength);
	
	swapByteOrder(header.alpha);
	swapByteOrder(header.beta);
	swapByteOrder(header.gamma);
	
	swapByteOrder(header.mapc);
	swapByteOrder(header.mapr);
	swapByteOrder(header.maps);
	
	swapByteOrder(header.amin);
	swapByteOrder(header.amax);
	swapByteOrder(header.amean);
	
	swapByteOrder(header.ispg);
	swapByteOrder(header.nsymbt);
	
	swapByteOrder(header.extra, 25);
	
	swapByteOrder(header.xorigin);
	swapByteOrder(header.yorigin);
	swapByteOrder(header.zorigin);

	swapByteOrder(header.rms);
	
	swapByteOrder(header.nlabl);
	
}

void MrcFileImpl::fillHeader(MrcHeader& header)
{
	// fill in the header's fields
	header.nx = m_DimX;
	header.ny = m_DimY;
	header.nz = m_DimZ;
	
	switch (m_VariableTypes[0])
	{
		case Char:
			header.mode = 0;
			break;
		case Short:
			header.mode = 1;
			break;
		case Float:
			header.mode = 2;
			break;
		default:
			// wha???
			// we shouldn't get here. (famous last words, I know)
			break;
	}
	
	// start coord, defaults to (0,0,0)
	header.nxstart = 0;   
	header.nystart = 0;      
	header.nzstart = 0; 
	
	// the dimensions again
	header.mx = m_DimX;
	header.my = m_DimY;
	header.mz = m_DimZ;
	
	// dimensions of a cell (span) 
	// (supposed to be in angstroms, but no guarantees)
	header.xlength = m_MaxX - m_MinX;
	header.ylength = m_MaxY - m_MinY;
	header.zlength = m_MaxZ - m_MinZ;
	
	// cell angles, all 90 deg
	header.alpha = 90.0;
	header.beta = 90.0;
	header.gamma = 90.0;
	
	// axis order
	header.mapc = 1; // number of axis corresponding to columns (X)
	header.mapr = 2; // number of axis corresponding to rows (Y)
	header.maps = 3; // number of axis corresponding to sections (Z)
	
	// min, max and mean... just put 0.0
	header.amin = 0.0; // minimum density value
	header.amax = 0.0; // maximum density value
	header.amean = 0.0; // mean density value
	
	header.ispg = 0; // space group number (0 for images)
	header.nsymbt = 0; // # of bytes for symmetry operators
	
	memset(header.extra, 0, 25*sizeof(int)); // user defined storage space
	
	// mesh origin
	header.xorigin = m_MinX; // X phase origin
	header.yorigin = m_MinY; // Y phase origin
	header.zorigin = m_MinZ; // Z phase origin

	// character string 'MAP '
	header.map[0] = 'M';
	header.map[1] = 'A';
	header.map[2] = 'P';
	header.map[3] = ' ';

	// machine stamp
	if (isLittleEndian()) {
		header.machst = 0x44410000;
		// swap it to big endian
		swapByteOrder(header.machst);
	}
	else
		header.machst = 0x11110000;

	header.rms = 0.0; // rms deviation of map from mean density
	header.nlabl = 1; // # of labels being used in the MRC header
	
	// zero the labels
	for (int i=0; i < 10; i++)
		memset(header.label[i], 0, 80);
	// fill in the first label
	strcpy(header.label[0], "Created by Volume Rover");

}

void MrcFileImpl::close()
{
	if (m_Attached) {
		m_File.close();
		m_Attached = false;
	}
}

