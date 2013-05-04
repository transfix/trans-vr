/*
  Copyright 2002-2005 The University of Texas at Austin
  
	Authors: John Wiggins <prok@ices.utexas.edu>
           Anthony Thane <thanea@ices.utexas.edu>
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

// PifFileImpl.cpp: implementation of the PifFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#include <qfileinfo.h>
#include <VolumeFileTypes/PifFileImpl.h>
#include <ByteOrder/ByteSwapping.h>
#include <string.h>
#include <stdlib.h>

/*static void printFileHeader(PifFileHeader& header)
{
	qDebug("file_id = \'%x%x%x%x%x%x%x%x\'", header.file_id[0], header.file_id[1],
				header.file_id[2], header.file_id[3], header.file_id[4],
				header.file_id[5], header.file_id[6], header.file_id[7]);
	qDebug("RealScaleFactor = %s", header.RealScaleFactor);
	qDebug("numImages = %d", header.numImages);
	qDebug("endianNess = %d", header.endianNess);
	qDebug("genProgram = %s", header.genProgram);
	qDebug("htype = %d", header.htype);
	qDebug("dims: %dx%dx%d", header.nx, header.ny, header.nz);
	qDebug("mode = %d", header.mode);
}*/

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

PifFileImpl::PifFileImpl()
{
	m_Attached = false;
	
	m_ScaleFactor = 1.0;
	m_MustScale = false;
	m_BytesPerVoxel = 1;
}

PifFileImpl::~PifFileImpl()
{
	close();
}

//! Returns true if the file given by fileName is of the correct type.
bool PifFileImpl::checkType(const QString& fileName)
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

	qDebug("PifFileImpl::checkType()");

	// read the header
	PifFileHeader header;
	file.readBlock((char*)&header, sizeof(PifFileHeader));
		
	//printFileHeader(header);

	if (checkHeader(header, file.size())) {
		return true;
	}
	else {
		// swap and try again
		swapFileHeader(header);
		//printFileHeader(header);
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

//! Associates this reader with the file given by fileName.
bool PifFileImpl::attachToFile(const QString& fileName, Mode mode)
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
		// XXX: we do not support writing
		return false;

		// open the file
		QString absFilePath = fileInfo.absFilePath();

		m_File.setName(absFilePath);
		if (!m_File.open(IO_WriteOnly | IO_Raw)) {
			qDebug("Error opening file");
			return false;
		}

		m_Attached = true;
	}

	return m_Attached;
}

//! Sets the type for the given variable v
void PifFileImpl::setVariableType(unsigned int variable, Type type)
{
	if (type != Long && type != Double)
		BasicVolumeFileImpl::setVariableType(variable, type);
	else {
		qWarning("Error: Trying to set variable type for a pif file to \
Long or Double is not allowed. Defaulting to Char.");
		BasicVolumeFileImpl::setVariableType(variable, Char);
	}
}

//! Reads char data from the file into the supplied buffer
bool PifFileImpl::readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
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
bool PifFileImpl::readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
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
bool PifFileImpl::readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
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
bool PifFileImpl::readFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Write)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Float) {
		// no good
		return false;
	}

	prepareToRead(variable);

	if (!m_MustScale) {
		// normal read
		// read in the data
		if (((int)numSamples*sizeof(float)) != m_File.readBlock((char*)buffer, numSamples*sizeof(float))) {
			return false;
		}
		if (m_MustSwap) swapByteOrder(buffer, numSamples);
	}
	else {
		// read short or int and scale to float
		unsigned int i;

		if (m_BytesPerVoxel == 2) {
			// Yes, this isn't pretty. But then, neither is PIF.
			short *readbuf = new short [numSamples];

			if (((int)numSamples*sizeof(short)) != m_File.readBlock((char *)readbuf,
numSamples*sizeof(short))) {
				delete [] readbuf;
				return false;
			}
			
			if (m_MustSwap) swapByteOrder((unsigned short *)readbuf, numSamples);

			// convert to float
			for (i=0; i < numSamples; i++)
				buffer[i] = (float)readbuf[i];
		
			//if (m_MustSwap) swapByteOrder(readbuf, numSamples);
			//if (m_MustSwap) swapByteOrder(buffer, numSamples);
			
			// scale
			for (i=0; i < numSamples; i++)
				//buffer[i] = m_ScaleFactor * readbuf[i];
				buffer[i] *= m_ScaleFactor;
			
			// done!
			delete [] readbuf;
		}
		else { // m_BytesPerVoxel == 4
			if (((int)numSamples*sizeof(int)) != m_File.readBlock((char *)buffer,
numSamples*sizeof(int))) {
				return false;
			}
		
			if (m_MustSwap) swapByteOrder(buffer, numSamples);
			
			// scale to float
			for (i=0; i < numSamples; i++)
				buffer[i] = m_ScaleFactor * ((int *)buffer)[i];
		}
	}

	incrementPosition(numSamples);
	return true;
}

//! Reads double data from the file into the supplied buffer
bool PifFileImpl::readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
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
	if (((int)numSamples*sizeof(double)) != m_File.readBlock((char*)buffer, numSamples*sizeof(double))) {
		return false;
	}
	if (m_MustSwap) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Writes the header to the file
bool PifFileImpl::writeHeader()
{
	// XXX: this is wrong. writing isn't really supported.
	return false;

	PifFileHeader fheader;
	PifImageHeader iheader;

	// fail if this file is open for reading
	if (m_OpenMode == Read)
		return false;

	// fill in the header
	fillHeaders(fheader, iheader);

	// write it out, return false if the write fails
	return (sizeof(PifFileHeader) == m_File.writeBlock((char *)&fheader, sizeof(PifFileHeader)));
}

//! Writes char data to the file from the supplied buffer
bool PifFileImpl::writeCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
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
bool PifFileImpl::writeShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Short) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// write out the data
	if (((int)numSamples*sizeof(unsigned short)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(unsigned short))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes long data to the file from the supplied buffer
bool PifFileImpl::writeLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Long) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// write out the data
	if (((int)numSamples*sizeof(unsigned int)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(unsigned int))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes float data to the file from the supplied buffer
bool PifFileImpl::writeFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Float) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// write out the data
	if (((int)numSamples*sizeof(float)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(float))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Writes double data to the file from the supplied buffer
bool PifFileImpl::writeDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
	if (m_OpenMode == Read)
		return false;

	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Double) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// write out the data
	if (((int)numSamples*sizeof(double)) != m_File.writeBlock((char*)buffer, numSamples*sizeof(double))) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Returns a new loader which can be used to load a file
VolumeFile* PifFileImpl::getNewVolumeFileLoader() const
{
	return new PifFileImpl;
}

//! Sets the position of the next read.
bool PifFileImpl::protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset)
{
	if (variable>=m_NumVariables || timeStep>=m_NumTimeSteps) {
		return false;
	}
	else {
		Q_ULLONG cOff = 512 + 512*(variable+1);
		if (m_MustScale && m_BytesPerVoxel == 2)
			cOff += (m_DimX*m_DimY*m_DimZ*variable + offset)*2;
		else
			cOff += (m_DimX*m_DimY*m_DimZ*variable + offset)*getBytesPerPixel(variable);
		m_File.at(cOff);
		//m_File.at(512 + 512*(variable+1) + (m_DimX*m_DimY*m_DimZ*variable + offset)*getBytesPerPixel(variable));
		return true;
	}
}

bool PifFileImpl::readHeader(Q_ULLONG fileSize)
{
	qDebug("PifFileImpl::readHeader()");
	// read the header
	PifFileHeader fheader;
	PifImageHeader iheader;

	m_File.readBlock((char*)&fheader, sizeof(PifFileHeader));
	m_File.readBlock((char*)&iheader, sizeof(PifImageHeader));

	// first try not swapping
	if (checkHeader(fheader, fileSize)) {
		m_MustSwap = false;
		return interpretHeaders(fheader, iheader, fileSize);;
	}
	else {
		// swap and try again
		swapFileHeader(fheader);
		swapImageHeader(iheader);
		if (checkHeader(fheader, fileSize)) {
			m_MustSwap = true;
			return interpretHeaders(fheader, iheader, fileSize);;
		}
		else {
			// we dont support wierd or exotic endianness
			return false;
		}
	}
}

bool PifFileImpl::checkHeader(PifFileHeader& header, Q_ULLONG fileSize)
{
	Q_ULLONG sizes[] = {1, 2, 4,0,0,0,0,2,0,4,0,0,0,0,0,0,0,0,0,0,2,4};

	//qDebug("PifFileFileImpl::checkHeader()");

	// check for the details we dont support
	if (header.htype == 0) {
		// we dont support this type or PIF file for now		
		qDebug("unsupported pif file. (htype = %d)", header.htype);
		return false;
	}

	if (!(header.mode == 0 || header.mode == 1 || header.mode == 2
		|| header.mode == 7 || header.mode == 9 || header.mode == 20
		|| header.mode == 21)) {
		// we dont support this type or PIF file for now		
		qDebug("unsupported pif file. (mode = %d)", header.mode);
		return false;
	}

	// check the fileSize
	if ((sizes[header.mode]*header.nx*header.ny*header.nz*header.numImages
			+ 512 + 512*header.numImages) != fileSize) {
		// the size does not match the header information
		qDebug("bad pif file? (file size != size given in header)");
		return false;
	}

	// everything checks out, return true
	return true;
}

bool PifFileImpl::interpretHeaders(PifFileHeader& fheader, PifImageHeader& iheader, Q_ULLONG fileSize)
{
	Type types[] = {Char, Short, Float};
	//Q_ULLONG sizes[] = {1, 2, 4};

	if (fheader.mode == 9)
		fheader.mode = 2;
	else if (fheader.mode == 2 || fheader.mode == 21) {
		m_MustScale = true;
		m_BytesPerVoxel = 4;
		fheader.mode = 2;
	}
	else if (fheader.mode == 7 || fheader.mode == 20) {
		m_MustScale = true;
		m_BytesPerVoxel = 2;
		fheader.mode = 2;
	}

	if (m_MustScale)
		m_ScaleFactor = (float)strtod(fheader.RealScaleFactor, NULL);

	m_DimX = fheader.nx;
	m_DimY = fheader.ny;
	m_DimZ = fheader.nz;
	m_NumTimeSteps = 1;
	makeSpaceForVariablesInfo(1);
	m_VariableNames[0] = "No Name";
	m_VariableTypes[0] = types[fheader.mode];
	m_MinX = 0.0;
	m_MinY = 0.0;
	m_MinZ = 0.0;
	m_MinT = 0.0;

	// we need to double check the meaning of xlength
	if (iheader.xlength<=0.0 || iheader.ylength<=0.0 || iheader.zlength<=0.0) {
		// hmm, this is wierd
		qDebug("image header cell sizes are zero or negative");
		m_MaxX = m_MinX + m_DimX*1.0;
		m_MaxY = m_MinY + m_DimY*1.0;
		m_MaxZ = m_MinZ + m_DimZ*1.0;
	}
	else {
		m_MaxX = m_MinX + iheader.xlength;
		m_MaxY = m_MinY + iheader.ylength;
		m_MaxZ = m_MinZ + iheader.zlength;
	}
	m_MaxT = 0.0;

	// everything checks out, return true
	return true;
}

void PifFileImpl::swapFileHeader(PifFileHeader& header)
{
	//qDebug("swap header");
	swapByteOrder(header.numImages);
	swapByteOrder(header.endianNess);
	swapByteOrder(header.htype);

	swapByteOrder(header.nx);
	swapByteOrder(header.ny);
	swapByteOrder(header.nz);
	
	//qDebug("header.mode = %x", header.mode);
	swapByteOrder(header.mode);
	//qDebug("header.mode = %x", header.mode);
}

void PifFileImpl::swapImageHeader(PifImageHeader& header)
{
	//qDebug("swap header");
	swapByteOrder(header.nx);
	swapByteOrder(header.ny);
	swapByteOrder(header.nz);
	
	//qDebug("header.mode = %x", header.mode);
	swapByteOrder(header.mode);
	//qDebug("header.mode = %x", header.mode);

	swapByteOrder(header.bkgnd);
	swapByteOrder(header.packRadius);
	
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
	
	swapByteOrder(header.min);
	swapByteOrder(header.max);
	swapByteOrder(header.mean);
	
	swapByteOrder(header.ispg);
	swapByteOrder(header.nsymbt);
	swapByteOrder(header.xorigin);
	swapByteOrder(header.yorigin);
	
	swapByteOrder(header.aoverb);
	swapByteOrder(header.map_abang);

	swapByteOrder(header.dela);
	swapByteOrder(header.delb);
	swapByteOrder(header.delc);
	
	swapByteOrder(header.t_matrix, 6);
	
	swapByteOrder(header.dthe);
	swapByteOrder(header.dphi_90);
	swapByteOrder(header.symmetry);
	swapByteOrder(header.binFactor);
	
	swapByteOrder(header.a_star);
	swapByteOrder(header.b_star);
	swapByteOrder(header.c_star);
	swapByteOrder(header.alp_star);
	swapByteOrder(header.bet_star);
	swapByteOrder(header.gam_star);
	
	swapByteOrder(header.pixelSize);
}

void PifFileImpl::fillHeaders(PifFileHeader& fheader, PifImageHeader& iheader)
{
	// XXX: this is incomplete. it also completely ignore the fact that
	// there must be a separate image header for each variable.

	// fill in the headers' fields
	fheader.numImages = m_NumVariables;
	fheader.htype = 1;

	fheader.nx = iheader.nx = m_DimX;
	fheader.ny = iheader.ny = m_DimY;
	fheader.nz = iheader.nz = m_DimZ;
	
	switch (m_VariableTypes[0])
	{
		case Char:
			fheader.mode = iheader.mode = 0;
			break;
		case Short:
			fheader.mode = iheader.mode = 1;
			break;
		case Float:
			fheader.mode = iheader.mode = 2;
			break;
		default:
			// wha???
			// we shouldn't get here. (famous last words, I know)
			break;
	}
	
	// start coord, defaults to (0,0,0)
	iheader.nxstart = 0;   
	iheader.nystart = 0;      
	iheader.nzstart = 0; 
	
	// the dimensions again
	iheader.mx = m_DimX;
	iheader.my = m_DimY;
	iheader.mz = m_DimZ;
	
	// dimensions of a cell (span) 
	// (supposed to be in angstroms, but no guarantees)
	iheader.xlength = m_MaxX - m_MinX;
	iheader.ylength = m_MaxY - m_MinY;
	iheader.zlength = m_MaxZ - m_MinZ;
	
	// cell angles, all 90 deg
	iheader.alpha = 90.0;
	iheader.beta = 90.0;
	iheader.gamma = 90.0;
	
	// axis order
	iheader.mapc = 1; // number of axis corresponding to columns (X)
	iheader.mapr = 2; // number of axis corresponding to rows (Y)
	iheader.maps = 3; // number of axis corresponding to sections (Z)
	
	// min, max and mean... just put 0.0
	iheader.min = 0.0; // minimum density value
	iheader.max = 0.0; // maximum density value
	iheader.mean = 0.0; // mean density value
	
	iheader.ispg = 0; // space group number (0 for images)
	iheader.nsymbt = 0; // # of bytes for symmetry operators
	
	// machine endianness 
	if (isLittleEndian())
		fheader.endianNess = 0;
	else
		fheader.endianNess = 1;
}

void PifFileImpl::close()
{
	if (m_Attached) {
		m_File.close();
		m_Attached = false;
	}
}

