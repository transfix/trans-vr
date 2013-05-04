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

// RawIVSource.cpp: implementation of the RawIVSource class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/RawIVSource.h>
#include <math.h>
#include <qdir.h>
#include <qfileinfo.h>
#include <qstringlist.h>
#include <qdatetime.h>
#include <qapplication.h>
#include <stdlib.h>
#include <ByteOrder/ByteSwapping.h>
#include <VolumeFileTypes/RawIVFileImpl.h>

const Q_INT32 CACHE_RECORD_VERSION_NUMBER = 6;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RawIVSource::RawIVSource(const QString& filename, const QString& cacheDir) 
: m_File(filename), m_CacheDir(cacheDir), m_FileInfo(filename)
{
	setDefaults();
}

RawIVSource::~RawIVSource()
{
	close();
	destroyBuffer();
}


bool RawIVSource::open(QWidget* parent)
{
	if (!m_IsOpen) {
		if (m_File.open(IO_ReadOnly | IO_Raw) && readHeader() && allocateBuffer( getDimX()*getDimY()*3 /* 3 slices */)) {
			// success
			m_IsOpen = true;
			// check for and create cache if needed
			// use progressbar to show progress
			m_ProgressDialog = new QProgressDialog("Creating a cache of preprocess work. \nThis happens the first time you load a file, or when a file has changed", 
				"Cancel", determineCacheWorkAmount(), parent, "Creating cache", true);
			m_ProgressValue = 0;
			try {
				if (!createCache(m_FileInfo.fileName(), m_CacheDir.absPath())) {
					// failed to create cache
					m_IsOpen = false;
					m_File.close();
					setError("Error creating cache\nMake sure there is enough harddisk space for the cache", true);
				}
			}
			catch(const UserCancelException&) {
				m_IsOpen = false;
				m_File.close();
				setError("Error creating cache\nCancelled by user", true);
			}
			delete m_ProgressDialog;
			m_ProgressDialog = 0;
		}
		else {
			// failure
			m_IsOpen = false;
			m_File.close();
			setError("Error opening file", true);
		}
	}
	return m_IsOpen;
}

void RawIVSource::close()
{
	if (m_IsOpen) {
		m_File.close();
		m_IsOpen = false;
	}
}

void RawIVSource::fillThumbnail(char* data, 
								unsigned int xDim, unsigned int yDim, unsigned int zDim, 
								uint variable, uint timeStep)
{
/* will not be implemented.  I will remove this from the base class later */	
}

void RawIVSource::fillData(char* data, uint xMin, uint yMin, uint zMin,
						   uint xMax, uint yMax, uint zMax,
						   uint xDim, uint yDim, uint zDim,
						   uint variable, uint timeStep)
{
	unsigned int strideX = (xDim==1?1:(xMax-xMin)/(xDim-1));
	unsigned int strideY = (yDim==1?1:(yMax-yMin)/(yDim-1));
	unsigned int strideZ = (zDim==1?1:(zMax-zMin)/(zDim-1));
	unsigned int newDimX = getDimX();
	unsigned int newDimY = getDimY();
	unsigned int newDimZ = getDimZ();
	unsigned int level = 0;
	QFile file;

	// check that strides are equal
	if (strideX!=strideY || strideY!=strideZ) {
		setError("Invalid query.  The stride in each dimension must be identical");
	}

	// find the correct level while keeping track of the level dimensions
	if (strideX==1) {
		// perform extraction from level 0
		if (m_DataType == DataTypeByte) {
			qDebug("Extract from level 0");
			QTime t;
			t.start();
			extractFromRawIVFile(data, xMin, yMin, zMin, xMax, yMax, zMax);
			qDebug("Elapsed time extracting from file: %d", t.elapsed());
		}
		else {
			// open the file
			file.setName(mipmapLevelFileName(level));
			file.open(IO_ReadOnly | IO_Raw);

			// perform extraction
			qDebug("Extract from level 0");
			QTime t;
			t.start();
			extractFromCacheFile(data, xMin, yMin, zMin, xMax, yMax, zMax, getDimX(), getDimY(), getDimZ(), file);
			qDebug("Elapsed time extracting from file: %d", t.elapsed());
			// close the file
			file.close();
		}
	}
	else {
		// determine the correct level
		while (strideX!=1) {
			// calculate the new dimensions
			newDimX = (newDimX+1)>>1;
			newDimY = (newDimY+1)>>1;
			newDimZ = (newDimZ+1)>>1;
			// increment the target level
			level++;
			// calculate the new stride
			strideX = strideX>>1;
			// calculate the new query min and max
			xMin = xMin >> 1;
			yMin = yMin >> 1;
			zMin = zMin >> 1;
			xMax = xMax >> 1;
			yMax = yMax >> 1;
			zMax = zMax >> 1;

		}

		// open the file
		file.setName(mipmapLevelFileName(level));
		file.open(IO_ReadOnly);
		
		// perform extraction
		qDebug("Extract from level %d", level);
		extractFromCacheFile(data, xMin, yMin, zMin, xMax, yMax, zMax, newDimX, newDimY, newDimZ, file);
	
		// close the file
		file.close();

	}
}

VolumeSource::DownLoadFrequency RawIVSource::interactiveUpdateHint() const
{
	return DLFDelayed;
}

void RawIVSource::setDefaults()
{
	m_IsOpen = false;
	m_Buffer = 0;
	m_BufferSize = 0;
	m_DataType = DataTypeByte;
	m_ProgressDialog = 0;
	m_ProgressValue = 0;
}

bool RawIVSource::allocateBuffer(unsigned int size)
{
	// only reallocate buffer if the current buffer is not big enough
	if (size > m_BufferSize) {
		destroyBuffer();
		return forceAllocateBuffer(size);
	}
	else {
		return true;
	}
}

bool RawIVSource::forceAllocateBuffer(unsigned int size)
{
	m_Buffer = new unsigned char[size];
	if (m_Buffer) {
		m_BufferSize = size;
		return true;
	}
	else {
		m_BufferSize = 0;
		return false;
	}
}

void RawIVSource::destroyBuffer()
{
	delete [] m_Buffer;
	m_Buffer = 0;
	m_BufferSize = 0;
}

bool RawIVSource::readHeader()
{
	float floatVals[3];
	unsigned int uintVals[3];

	// read in the mins
	//fread(floatVals, sizeof(float), 3, fp);
	m_File.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	setMinX(floatVals[0]);
	setMinY(floatVals[1]);
	setMinZ(floatVals[2]);

	// read in the maxs
	//fread(floatVals, sizeof(float), 3, fp);
	m_File.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	setMaxX(floatVals[0]);
	setMaxY(floatVals[1]);
	setMaxZ(floatVals[2]);

	// ignore num verts and num cells, redundant
	//fread(uintVals, sizeof(unsigned int), 2, fp);
	m_File.readBlock((char*)uintVals, sizeof(unsigned int)*2);

	// read in the dimensions
	//fread(uintVals, sizeof(unsigned int), 3, fp);
	m_File.readBlock((char*)uintVals, sizeof(unsigned int)*3);
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	setDimX(uintVals[0]);
	setDimY(uintVals[1]);
	setDimZ(uintVals[2]);

	// ignore the "origin"  not sure what it means...probably redundant
	//fread(floatVals, sizeof(float), 3, fp);
	m_File.readBlock((char*)floatVals, sizeof(float)*3);

	// read in the span...redundant but will use it to check min and max
	//fread(floatVals, sizeof(float), 3, fp);
	m_File.readBlock((char*)floatVals, sizeof(float)*3);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);

	// check min + span*dim == max to make sure file is consistant
	if ( !(
		fabs(m_MinX + floatVals[0]*(double)(m_DimX-1) - m_MaxX) < 0.0001 &&
		fabs(m_MinY + floatVals[1]*(double)(m_DimY-1) - m_MaxY) < 0.0001 &&
		fabs(m_MinZ + floatVals[2]*(double)(m_DimZ-1) - m_MaxZ) < 0.0001)) {

		// inconsistant, junk it all and replace with simple defaults
		m_MinX = 0; m_MinY = 0; m_MinZ = 0;
		m_MaxX = m_MinX + (double)(m_DimX-1)*floatVals[0];
		m_MaxY = m_MinY + (double)(m_DimY-1)*floatVals[1];
		m_MaxZ = m_MinZ + (double)(m_DimZ-1)*floatVals[2];

	}
	
	if (getNumVerts() == m_FileInfo.size()-68) {
		// success; byte datatype
		m_DataType = DataTypeByte;
		return true;
	}
	else if (getNumVerts()*2 == m_FileInfo.size()-68) {
		// success; short datatype
		m_DataType = DataTypeShort;
		return true;
	}
	else if (getNumVerts()*4 == m_FileInfo.size()-68) {
		// success; float datatype
		m_DataType = DataTypeFloat;
		return true;
	}
	else {
		// not byte short of float
		return false;
	}
}

void RawIVSource::extractFromRawIVFile(char* data, uint xMin, uint yMin, uint zMin, 
									   uint xMax, uint yMax, uint zMax)
{
	uint width = xMax-xMin+1;
	uint height = yMax-yMin+1;
	uint depth = zMax-zMin+1;
	uint j,k;

	// loop through each slice
	for (k=0; k<depth; k++) {
		// loop through each line
		for (j=0; j<height; j++) {
			// go to the start of the "scanline"
			m_File.at((zMin+k)*getDimX()*getDimY()+((yMin+j)*getDimX())+xMin+68);

			// read in a line
			m_File.readBlock(data+(k*width*height + j*width), width);
		}
	}
}

void RawIVSource::extractFromCacheFile(char* data, uint xMin, uint yMin, uint zMin, 
									   uint xMax, uint yMax, uint zMax, 
									   uint newDimX, uint newDimY, uint newDimZ, QFile& file)
{
	uint width = xMax-xMin+1;
	uint height = yMax-yMin+1;
	uint depth = zMax-zMin+1;
	uint j,k;

	// loop through each slice
	for (k=0; k<depth; k++) {
		// loop through each line
		for (j=0; j<height; j++) {
			// go to the start of the "scanline"
			file.at((zMin+k)*newDimX*newDimY+((yMin+j)*newDimX)+xMin);

			// read in a line
			file.readBlock(data+(k*width*height + j*width), width);
		}
	}
}
bool RawIVSource::createCache(const QString& fileName, const QString& cacheRoot)
{
	QDir dir(cacheRoot);

	if (!dir.exists()) {
		setError("The cache directory does not exist", true);
	}

	// check to see if a cache dir for the file exists
	if (!dir.exists(fileName)) {
		dir.mkdir(fileName);
		dir.cd(fileName);
		createNewCache(m_FileInfo, dir);
		if (cacheOutdated(m_FileInfo, dir)) {
			// there was a problem creating the cache
			return false;
		}
		else {
			return true;
		}
	}
	else {
		dir.cd(fileName);
		// check to see if the cache is up to date with respect to the rawiv file
		if (cacheOutdated(m_FileInfo, dir)) {
			// clear the directory
			clearCache(dir);
			// re-create the cache
			createNewCache(m_FileInfo, dir);
			if (cacheOutdated(m_FileInfo, dir)) {
				// there was a problem creating the cache
				return false;
			}
			else {
				return true;
			}
		}
		else { // cache already existsed, just return true
			return true;
		}
	}
}

void RawIVSource::createNewCache( QFileInfo fileInfo, QDir cacheDir)
{

	// create the record file
	createCacheRecordFile(fileInfo, cacheDir);

	// create the mipmap levels
	createMipmapLevels();

}

void RawIVSource::createCacheRecordFile(QFileInfo fileInfo, QDir cacheDir)
{

	// open the file
	QString filename(cacheDir.absPath() + "/" + fileInfo.baseName() + "_Record" + ".cache");
	QFile file(filename);
	file.open(IO_WriteOnly);

	// set up the datastream
	QDataStream stream( &file );

	// output the magic string
	stream << QString("Volume Cache Record");

	// output the version
	stream << (Q_INT32) CACHE_RECORD_VERSION_NUMBER;

	// output the modified date and time
	QDateTime dateTime = fileInfo.lastModified();
	stream << dateTime;

	// output the absolute file path
	stream << fileInfo.absFilePath();

	file.close();
}

void RawIVSource::clearCache(QDir cacheDir)
{
	// get the filtered list of files
	QStringList files = cacheDir.entryList("*.cache");

	QString file;

	// loop though the list of files
	for (QStringList::Iterator it = files.begin(); it != files.end(); ++it) {
		file = *it;
		//delete each file
		cacheDir.remove(file);
	}
}

void RawIVSource::createMipmapLevels()
{
	QFile source, target;
	bool moreLevels;
	unsigned int targetLevel;
	unsigned int newDimX;
	unsigned int newDimY;
	unsigned int newDimZ;

	QTime t;
	t.start();

	if ( (getDimX()>1 || getDimY()>1 || getDimZ()>1) ) {
		if (m_DataType==DataTypeByte) { // byte data, there is no level 0 mipmap
			targetLevel = 1;
			target.setName(mipmapLevelFileName(targetLevel));
			target.open( IO_WriteOnly );
			moreLevels = createMipmapLevelFromRawIVGaussian(target);
			target.close();

			// calculate the new dimensions
			newDimX = (getDimX()+1)>>1;
			newDimY = (getDimY()+1)>>1;
			newDimZ = (getDimZ()+1)>>1;
			// set targetlevel to 2
			targetLevel++;
		}
		else if (m_DataType==DataTypeShort) { // create a level 0 mipmap
			targetLevel = 0;
			target.setName(mipmapLevelFileName(targetLevel));
			target.open( IO_WriteOnly );
			moreLevels = createMipmapLevelZeroFromShortRawIV(target);
			target.close();

			// calculate the new dimensions
			newDimX = getDimX();
			newDimY = getDimY();
			newDimZ = getDimZ();
			// set targetlevel to 1
			targetLevel++;

		}
		else { // DataTypeFloat
			targetLevel = 0;
			target.setName(mipmapLevelFileName(targetLevel));
			target.open( IO_WriteOnly );
			moreLevels = createMipmapLevelZeroFromFloatRawIV(target);
			target.close();

			// calculate the new dimensions
			newDimX = getDimX();
			newDimY = getDimY();
			newDimZ = getDimZ();
			// set targetlevel to 1
			targetLevel++;
		}

		while (moreLevels) {
			// open the source and target files
			target.setName(mipmapLevelFileName(targetLevel));
			target.open( IO_WriteOnly );
			source.setName(mipmapLevelFileName(targetLevel-1));
			source.open( IO_ReadOnly );

			// create the level
			moreLevels = createMipmapLevelFromCacheFileGaussian(source, target, newDimX, newDimY, newDimZ);

			// close the files
			target.close();
			source.close();

			// calculate the new dimensions
			newDimX = (newDimX+1)>>1;
			newDimY = (newDimY+1)>>1;
			newDimZ = (newDimZ+1)>>1;
			// increment the target level
			targetLevel++;
		}

	}

	qDebug("Timer to create mipmap levels: %d", t.elapsed());
}

static inline void checkMin(float& currentMin, bool& minSet, float val)
{
	if (minSet) {
		currentMin = (val<currentMin?val:currentMin);
	}
	else {
		currentMin = val;
		minSet = true;
	}
}

static inline void checkMax(float& currentMax, bool& maxSet, float val)
{
	if (maxSet) {
		currentMax = (val>currentMax?val:currentMax);
	}
	else {
		currentMax = val;
		maxSet = true;
	}
}

static inline unsigned char convertToByte(unsigned short val, unsigned short min, unsigned short max)
{
	int tempval;
	if (max-min==0) {
		tempval = 0;
	}
	else {
		tempval = (val-min) * 255 / (max-min);
		tempval = (tempval<255?tempval:255);
		tempval = (tempval>0?tempval:0);
	}
	return (unsigned char)tempval;
}

static inline void checkMin(unsigned short& currentMin, bool& minSet, unsigned short val)
{
	if (minSet) {
		currentMin = (val<currentMin?val:currentMin);
	}
	else {
		currentMin = val;
		minSet = true;
	}
}

static inline void checkMax(unsigned short& currentMax, bool& maxSet, unsigned short val)
{
	if (maxSet) {
		currentMax = (val>currentMax?val:currentMax);
	}
	else {
		currentMax = val;
		maxSet = true;
	}
}

static inline unsigned char convertToByte(float val, float min, float max)
{
	/*XXX test code here
	if (val==0) {
		val = 1;
	}
	else {
		val = log(val);
	}
	*/

	int tempval;
	if (max-min==0.0) {
		tempval = 0;
	}
	else {
		tempval = (val-min) * 255 / (max-min);
		tempval = (tempval<255?tempval:255);
		tempval = (tempval>0?tempval:0);
	}
	return (unsigned char)tempval;
}

static const int buffSize = 256;

bool RawIVSource::createMipmapLevelZeroFromShortRawIV(QFile& targetFile)
{
	unsigned short min, max;
	bool minSet = false, maxSet = false;
	unsigned int k,j;
	unsigned short buffer[buffSize];
	unsigned char byteBuffer[buffSize];

	// determine min and max
	qDebug("Finding min and max");
	m_File.at(68);

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_File.readBlock((char*)buffer, buffSize*sizeof(unsigned short));
		if (isLittleEndian()) swapByteOrder(buffer, buffSize);
		for (j=0; j<buffSize; j++) {
			checkMin(min, minSet, buffer[j]);
			checkMax(max, maxSet, buffer[j]);
		}
		incrementProgress(buffSize);
	}
	// do remainder
	m_File.readBlock((char*)buffer, (getNumVerts()%buffSize)*sizeof(unsigned short));
	if (isLittleEndian()) swapByteOrder(buffer, (getNumVerts()%buffSize));
	for (j=0; j<(getNumVerts()%buffSize); j++) {
		checkMin(min, minSet, buffer[j]);
		checkMax(max, maxSet, buffer[j]);
	}
	incrementProgress(getNumVerts()%buffSize);

	// convert to byte
	qDebug("Converting to byte");
	m_File.at(68);
	targetFile.at(0);

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_File.readBlock((char*)buffer, buffSize*sizeof(unsigned short));
		if (isLittleEndian()) swapByteOrder(buffer, buffSize);
		for (j=0; j<buffSize; j++) {
			byteBuffer[j] = convertToByte(buffer[j], min, max);
		}
		targetFile.writeBlock((char*)byteBuffer, buffSize*sizeof(unsigned char));
		incrementProgress(buffSize);
	}
	// do remainder
	m_File.readBlock((char*)buffer, (getNumVerts()%buffSize)*sizeof(unsigned short));
	if (isLittleEndian()) swapByteOrder(buffer, (getNumVerts()%buffSize));
	for (j=0; j<(getNumVerts()%buffSize); j++) {
		byteBuffer[j] = convertToByte(buffer[j], min, max);
	}
	targetFile.writeBlock((char*)byteBuffer, (getNumVerts()%buffSize)*sizeof(unsigned char));
	incrementProgress(getNumVerts()%buffSize);

	return (getDimX()>1 || getDimY()>1 || getDimZ()>1);
}

bool RawIVSource::createMipmapLevelZeroFromFloatRawIV(QFile& targetFile)
{
	float min, max;
	bool minSet = false, maxSet = false;
	unsigned int k,j;
	float buffer[buffSize];
	unsigned char byteBuffer[buffSize];

	// determine min and max
	qDebug("Finding min and max");
	m_File.at(68);

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_File.readBlock((char*)buffer, buffSize*sizeof(float));
		if (isLittleEndian()) swapByteOrder(buffer, buffSize);
		for (j=0; j<buffSize; j++) {
			checkMin(min, minSet, (buffer[j]));
			checkMax(max, maxSet, (buffer[j]));
		}
		incrementProgress(buffSize);
	}
	// do remainder
	m_File.readBlock((char*)buffer, (getNumVerts()%buffSize)*sizeof(float));
	if (isLittleEndian()) swapByteOrder(buffer, (getNumVerts()%buffSize));
	for (j=0; j<(getNumVerts()%buffSize); j++) {
		checkMin(min, minSet, (buffer[j]));
		checkMax(max, maxSet, (buffer[j]));
	}
	incrementProgress(getNumVerts()%buffSize);

	/*XXX test code here
	max = log(max);
	if (min==0.0) {
		min = 1;
	}
	else {
		min = log(min);
	}
	*/

	// convert to byte
	qDebug("Converting to byte");
	m_File.at(68);
	targetFile.at(0);

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_File.readBlock((char*)buffer, buffSize*sizeof(float));
		if (isLittleEndian()) swapByteOrder(buffer, buffSize);
		for (j=0; j<buffSize; j++) {
			byteBuffer[j] = ( convertToByte(buffer[j], min, max) );
		}
		targetFile.writeBlock((char*)byteBuffer, buffSize*sizeof(unsigned char));
		incrementProgress(buffSize);
	}
	// do remainder
	m_File.readBlock((char*)buffer, (getNumVerts()%buffSize)*sizeof(float));
	if (isLittleEndian()) swapByteOrder(buffer, (getNumVerts()%buffSize));
	for (j=0; j<(getNumVerts()%buffSize); j++) {
		byteBuffer[j] = ( convertToByte(buffer[j], min, max) );
	}
	targetFile.writeBlock((char*)byteBuffer, (getNumVerts()%buffSize)*sizeof(unsigned char));
	incrementProgress(getNumVerts()%buffSize);

	return (getDimX()>1 || getDimY()>1 || getDimZ()>1);
}

static inline unsigned int resample(unsigned int * samples)
{
	return ((
		samples[0] +
		samples[1] +
		samples[2] +
		samples[3] +
		samples[4] +
		samples[5] +
		samples[6] +
		samples[7])>>3);
}

bool RawIVSource::createMipmapLevelFromRawIVGaussian(QFile& targetFile)
{
	// weighted average of 27 voxels

	// we will have 3 slices in memory at a time,
	// however we will normally read in 2 slices at once

	// prepare the new dimensions by dividing by 2
	unsigned int newDimX = (getDimX()+1)>>1;
	unsigned int newDimY = (getDimY()+1)>>1;
	unsigned int newDimZ = (getDimZ()+1)>>1;

	// the weights
	unsigned int weights[27] = {
		1,2,1,
		2,4,2,
		1,2,1,

		2,4,2,
		4,20,4,
		2,4,2,

		1,2,1,
		2,4,2,
		1,2,1};

	// buffers for the slices
	// note that m_Buffer has space for 3 slices
	unsigned char* slice0 = m_Buffer;
	unsigned char* slice1 = m_Buffer+getDimX()*getDimY();
	unsigned char* slice2 = m_Buffer+2*getDimX()*getDimY();
	unsigned char* temp;

	unsigned int positiveI, positiveJ;
	unsigned int negativeI, negativeJ;
	unsigned int sliceNum, i, j;
	unsigned int sum;
	unsigned int average;

	// set the file position
	m_File.at(68);
	for (sliceNum=0; sliceNum<getDimZ(); sliceNum+=2) {
		if (sliceNum==0) { // top border
			// read in 3 slices
			m_File.readBlock((char*)slice0, getDimX()*getDimY());
			m_File.readBlock((char*)slice1, getDimX()*getDimY());
			m_File.readBlock((char*)slice2, getDimX()*getDimY());
		}
		else if (sliceNum+1==getDimZ()) { // bottom border
			// swap slices and read in last slice twice
			temp = slice2;
			slice2 = slice0;
			slice0 = slice2;
			m_File.readBlock((char*)slice1, 1*getDimX()*getDimY());
			m_File.at(sliceNum*getDimX()*getDimY()+68);
			m_File.readBlock((char*)slice2, 1*getDimX()*getDimY());
		}
		else { // normal case
			// swap slices and read in two new slices
			temp = slice2;
			slice2 = slice0;
			slice0 = slice2;
			m_File.readBlock((char*)slice1, getDimX()*getDimY());
			m_File.readBlock((char*)slice2, getDimX()*getDimY());
		}
		// perform averaging
		for (j=0; j<getDimY(); j+=2) {
			// check to see one of the samples will be out of bounds
			if (j==0) {
				// repeat a sample
				negativeJ = 0;
			}
			else {
				// normal case
				negativeJ = j-1;
			}

			if (j+1 >= getDimY()) {
				// repeat a sample
				positiveJ = j;
			}
			else {
				// normal case
				positiveJ = j+1;
			}

			for (i=0; i<getDimX(); i+=2) {
				// check to see one of the samples will be out of bounds
				if (i==0) {
					// repeat a sample
					negativeI = 0;
				}
				else {
					// normal case
					negativeI = i-1;
				}

				if (i+1 >= getDimX()) {
					// repeat a sample
					positiveI = i;
				}
				else {
					// normal case
					positiveI = i+1;
				}

				sum = 
					weights[ 0]*slice0[getDimX()*negativeJ + negativeI] +
					weights[ 1]*slice0[getDimX()*negativeJ + i] +
					weights[ 2]*slice0[getDimX()*negativeJ + positiveI] +
					weights[ 3]*slice0[getDimX()*j + negativeI] +
					weights[ 4]*slice0[getDimX()*j + i] +
					weights[ 5]*slice0[getDimX()*j + positiveI] +
					weights[ 6]*slice0[getDimX()*positiveJ + negativeI] +
					weights[ 7]*slice0[getDimX()*positiveJ + i] +
					weights[ 8]*slice0[getDimX()*positiveJ + positiveI] +
					weights[ 9]*slice1[getDimX()*negativeJ + negativeI] +
					weights[10]*slice1[getDimX()*negativeJ + i] +
					weights[11]*slice1[getDimX()*negativeJ + positiveI] +
					weights[12]*slice1[getDimX()*j + negativeI] +
					weights[13]*slice1[getDimX()*j + i] +
					weights[14]*slice1[getDimX()*j + positiveI] +
					weights[15]*slice1[getDimX()*positiveJ + negativeI] +
					weights[16]*slice1[getDimX()*positiveJ + i] +
					weights[17]*slice1[getDimX()*positiveJ + positiveI] +
					weights[18]*slice2[getDimX()*negativeJ + negativeI] +
					weights[19]*slice2[getDimX()*negativeJ + i] +
					weights[20]*slice2[getDimX()*negativeJ + positiveI] +
					weights[21]*slice2[getDimX()*j + negativeI] +
					weights[22]*slice2[getDimX()*j + i] +
					weights[23]*slice2[getDimX()*j + positiveI] +
					weights[24]*slice2[getDimX()*positiveJ + negativeI] +
					weights[25]*slice2[getDimX()*positiveJ + i] +
					weights[26]*slice2[getDimX()*positiveJ + positiveI];

				average = sum/76;
				// write out the result
				targetFile.writeBlock((char*)&average, 1);


			}

		}

		incrementProgress(getDimX()*getDimY()*2);

	}
	return (newDimX>1 || newDimY>1 || newDimZ>1);
}

bool RawIVSource::createMipmapLevelFromCacheFileGaussian(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int offset)
{
	// weighted average of 27 voxels

	// we will have 3 slices in memory at a time,
	// however we will normally read in 2 slices at once

	// prepare the new dimensions by dividing by 2
	unsigned int newDimX = (dimX+1)>>1;
	unsigned int newDimY = (dimY+1)>>1;
	unsigned int newDimZ = (dimZ+1)>>1;

	// the weights
	unsigned int weights[27] = {
		1,2,1,
		2,4,2,
		1,2,1,

		2,4,2,
		4,20,4,
		2,4,2,

		1,2,1,
		2,4,2,
		1,2,1};

	// buffers for the slices
	// note that m_Buffer has space for 3 slices
	unsigned char* slice0 = m_Buffer;
	unsigned char* slice1 = m_Buffer+getDimX()*getDimY();
	unsigned char* slice2 = m_Buffer+2*getDimX()*getDimY();
	unsigned char* temp;

	unsigned int positiveI, positiveJ;
	unsigned int negativeI, negativeJ;
	unsigned int sliceNum, i, j;
	unsigned int sum;
	unsigned int average;

	if (dimX==0 || dimY==0 || dimZ==0) {
		qDebug("WARNING: problem with dimensions.  dimX = %d.  dimY = %d, dimZ = %d", dimX, dimY, dimZ);
	}



	// set the file position
	source.at(offset);
	for (sliceNum=0; sliceNum<dimZ; sliceNum+=2) {
		if (sliceNum==0) { // top border
			// read in 3 slices
			source.readBlock((char*)slice0, dimX*dimY);
			source.readBlock((char*)slice1, dimX*dimY);
			source.readBlock((char*)slice2, dimX*dimY);
		}
		else if (sliceNum+1==dimZ) { // bottom border
			// swap slices and read in last slice twice
			temp = slice2;
			slice2 = slice0;
			slice0 = slice2;
			source.readBlock((char*)slice1, 1*dimX*dimY);
			source.at(sliceNum*dimX*dimY+offset);
			source.readBlock((char*)slice2, 1*dimX*dimY);
		}
		else { // normal case
			// swap slices and read in two new slices
			temp = slice2;
			slice2 = slice0;
			slice0 = slice2;
			source.readBlock((char*)slice1, dimX*dimY);
			source.readBlock((char*)slice2, dimX*dimY);
		}
		// perform averaging
		for (j=0; j<dimY; j+=2) {
			// check to see one of the samples will be out of bounds
			if (j==0) {
				// repeat a sample
				negativeJ = 0;
			}
			else {
				// normal case
				negativeJ = j-1;
			}

			if (j+1 >= dimY) {
				// repeat a sample
				positiveJ = j;
			}
			else {
				// normal case
				positiveJ = j+1;
			}

			for (i=0; i<dimX; i+=2) {
				// check to see one of the samples will be out of bounds
				if (i==0) {
					// repeat a sample
					negativeI = 0;
				}
				else {
					// normal case
					negativeI = i-1;
				}

				if (i+1 >= dimX) {
					// repeat a sample
					positiveI = i;
				}
				else {
					// normal case
					positiveI = i+1;
				}

				sum = 
					weights[ 0]*slice0[dimX*negativeJ + negativeI] +
					weights[ 1]*slice0[dimX*negativeJ + i] +
					weights[ 2]*slice0[dimX*negativeJ + positiveI] +
					weights[ 3]*slice0[dimX*j + negativeI] +
					weights[ 4]*slice0[dimX*j + i] +
					weights[ 5]*slice0[dimX*j + positiveI] +
					weights[ 6]*slice0[dimX*positiveJ + negativeI] +
					weights[ 7]*slice0[dimX*positiveJ + i] +
					weights[ 8]*slice0[dimX*positiveJ + positiveI] +
					weights[ 9]*slice1[dimX*negativeJ + negativeI] +
					weights[10]*slice1[dimX*negativeJ + i] +
					weights[11]*slice1[dimX*negativeJ + positiveI] +
					weights[12]*slice1[dimX*j + negativeI] +
					weights[13]*slice1[dimX*j + i] +
					weights[14]*slice1[dimX*j + positiveI] +
					weights[15]*slice1[dimX*positiveJ + negativeI] +
					weights[16]*slice1[dimX*positiveJ + i] +
					weights[17]*slice1[dimX*positiveJ + positiveI] +
					weights[18]*slice2[dimX*negativeJ + negativeI] +
					weights[19]*slice2[dimX*negativeJ + i] +
					weights[20]*slice2[dimX*negativeJ + positiveI] +
					weights[21]*slice2[dimX*j + negativeI] +
					weights[22]*slice2[dimX*j + i] +
					weights[23]*slice2[dimX*j + positiveI] +
					weights[24]*slice2[dimX*positiveJ + negativeI] +
					weights[25]*slice2[dimX*positiveJ + i] +
					weights[26]*slice2[dimX*positiveJ + positiveI];

				average = sum/76;
				// write out the result
				target.writeBlock((char*)&average, 1);

			}

		}

		incrementProgress(dimX*dimY*2);

	}
	return (newDimX>1 || newDimY>1 || newDimZ>1);}


bool RawIVSource::createMipmapLevelFromRawIV(QFile& targetFile)
{
	// average every eight voxels to form one voxel in the next mipmap level

	// we will read in 2 slices at a time, and perform the averaging

	// prepare the new dimensions by dividing by 2
	unsigned int newDimX = (getDimX()+1)>>1;
	unsigned int newDimY = (getDimY()+1)>>1;
	unsigned int newDimZ = (getDimZ()+1)>>1;


	unsigned int sliceNum, i, j;
	unsigned int positiveI, positiveJ;
	unsigned int samples[8];
	unsigned char average;
	// set the file position
	m_File.at(68);
	for (sliceNum=0; sliceNum<getDimZ(); sliceNum+=2) {

		// set the file position  commented out because it might not be necessary
		//m_File.at(sliceNum*getDimX()*getDimY()+68);
		if (sliceNum+1 >= getDimZ() ) {
			// we only have one slice left
			// read it in twice
			m_File.readBlock((char*)m_Buffer, 1*getDimX()*getDimY());
			m_File.at(sliceNum*getDimX()*getDimY()+68);
			m_File.readBlock((char*)m_Buffer+getDimX()+getDimY(), 1*getDimX()*getDimY());
		}
		else {
			// read in 2 slices
			m_File.readBlock((char*)m_Buffer, 2*getDimX()*getDimY());

		}

		// perform averaging
		for (j=0; j<getDimY(); j+=2) {
			// check to see one of the samples will be out of bounds
			if (j+1 >= getDimY()) {
				// repeat a sample
				positiveJ = j;
			}
			else {
				// normal case
				positiveJ = j+1;
			}
			for (i=0; i<getDimX(); i+=2) {
				// check to see one of the samples will be out of bounds
				if (i+1 >= getDimX()) {
					// repeat a sample
					positiveI = i;
				}
				else {
					// normal case
					positiveI = i+1;
				}

				// add up the eight samples
				samples[0] = m_Buffer[j*getDimX() + i];
				samples[1] = m_Buffer[j*getDimX() + positiveI];
				samples[2] = m_Buffer[positiveJ*getDimX() + i];
				samples[3] = m_Buffer[positiveJ*getDimX() + positiveI];
				samples[4] = m_Buffer[getDimX()*getDimY() + j*getDimX() + i];
				samples[5] = m_Buffer[getDimX()*getDimY() + j*getDimX() + positiveI];
				samples[6] = m_Buffer[getDimX()*getDimY() + positiveJ*getDimX() + i];
				samples[7] = m_Buffer[getDimX()*getDimY() + positiveJ*getDimX() + positiveI];

				// divide by 8
				average = resample(samples);

				// write out the result
				targetFile.writeBlock((char*)&average, 1);
			}
		}
		incrementProgress(getDimX()*getDimY()*2);
	}
	return (newDimX>1 || newDimY>1 || newDimZ>1);

}

bool RawIVSource::createMipmapLevelFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ)
{
	// average every eight voxels to form one voxel in the next mipmap level

	// we will read in 2 rows at a time, and perform the averaging

	// prepare the new dimensions by dividing by 2
	unsigned int newDimX = (dimX+1)>>1;
	unsigned int newDimY = (dimY+1)>>1;
	unsigned int newDimZ = (dimZ+1)>>1;

	unsigned int samples[8];

	if (dimX==0 || dimY==0 || dimZ==0) {
		qDebug("WARNING: problem with dimensions.  dimX = %d.  dimY = %d, dimZ = %d", dimX, dimY, dimZ);
	}


	unsigned int sliceNum, i, j;
	unsigned int positiveI, positiveJ;
	unsigned char average;
	// set the file position
	source.at(0);
	for (sliceNum=0; sliceNum<dimZ; sliceNum+=2) {

		// set the file position commented out cause may not be necesary
		// source.at(sliceNum*dimX*dimY);

		if (sliceNum+1 >= dimZ ) {
			// we only have one slice left
			// read it in twice
			source.readBlock((char*)m_Buffer, 1*dimX*dimY);
			source.at(sliceNum*dimX*dimY);
			source.readBlock((char*)m_Buffer+dimX+dimY, 1*dimX*dimY);
		}
		else {
			// read in 2 slices
			source.readBlock((char*)m_Buffer, 2*dimX*dimY);

		}

		// perform averaging
		for (j=0; j<dimY; j+=2) {
			// check to see one of the samples will be out of bounds
			if (j+1 >= dimY) {
				// repeat a sample
				positiveJ = j;
			}
			else {
				// normal case
				positiveJ = j+1;
			}
			for (i=0; i<dimX; i+=2) {
				// check to see one of the samples will be out of bounds
				if (i+1 >= dimX) {
					// repeat a sample
					positiveI = i;
				}
				else {
					// normal case
					positiveI = i+1;
				}

				// add up the eight samples
				 
				samples[0] = m_Buffer[j*dimX + i];
				samples[1] = m_Buffer[j*dimX + positiveI];
				samples[2] = m_Buffer[positiveJ*dimX + i];
				samples[3] = m_Buffer[positiveJ*dimX + positiveI];
				samples[4] = m_Buffer[dimX*dimY + j*dimX + i];
				samples[5] = m_Buffer[dimX*dimY + j*dimX + positiveI];
				samples[6] = m_Buffer[dimX*dimY + positiveJ*dimX + i];
				samples[7] = m_Buffer[dimX*dimY + positiveJ*dimX + positiveI];

				// divide by 8
				average = resample(samples);

				// write out the result
				target.writeBlock((char*)&average, 1);
			}
		}
		incrementProgress(dimX*dimY*2);
	}
	return (newDimX>1 || newDimY>1 || newDimZ>1);

}

bool RawIVSource::cacheOutdated(QFileInfo fileInfo, QDir cacheDir)
{
	// check for the existance of the record file
	QString fileName(cacheDir.absPath() + "/" + fileInfo.baseName() + "_Record" + ".cache");
	QFileInfo recordInfo(fileName);
	if (!recordInfo.exists()) {
		// no record file means cache is out of date
		return true;
	}
	else {
		// open the record file
		QFile recordFile(fileName);
		recordFile.open(IO_ReadOnly);
		QDataStream stream( &recordFile );

		// read in the magic string and check it
		QString MagicString;
		stream >> MagicString;
		if (MagicString != QString("Volume Cache Record")) {
			return true;
		}

		// check to make sure its the current version
		Q_INT32 version;
		stream >> version;
		if (version != CACHE_RECORD_VERSION_NUMBER) {
			return true;
		}

		// check the modified date
		QDateTime dateTime;
		stream >> dateTime;
		if (dateTime != fileInfo.lastModified()) {
			return true;
		}

		// check the absolute path
		QString absFilePath;
		stream >> absFilePath;
		if (absFilePath != fileInfo.absFilePath()) {
			return true;
		}

		recordFile.close();

		// check the mipmapLevels
		unsigned int level, dimX, dimY, dimZ;
		bool moreLevels;
		if (m_DataType==DataTypeByte) {
			level = 1;
			dimX = (getDimX()+1)>>1;
			dimY = (getDimY()+1)>>1;
			dimZ = (getDimZ()+1)>>1;
			moreLevels = (getDimX()!=1 || getDimY()!=1 || getDimZ()!=1);
		}
		else {
			level = 0;
			dimX = getDimX();
			dimY = getDimY();
			dimZ = getDimZ();
			moreLevels = true; // there is at least a level 0
		}


		while (moreLevels) {
			if (!checkMipmapLevel(level, dimX, dimY, dimZ)) {
				// problem with mipmap file
				return true; // cache is outdated
			}
			else {
				moreLevels = (dimX!=1 || dimY!=1 || dimZ!=1);
				// get ready for next level
				level++;
				dimX = (dimX+1)>>1;
				dimY = (dimY+1)>>1;
				dimZ = (dimZ+1)>>1;
			}
		}



		// everything checks out, cache is not outdated
		return false;
	}
}

bool RawIVSource::checkMipmapLevel(uint level, uint dimX, uint dimY, uint dimZ)
{
	QFileInfo fileInfo(mipmapLevelFileName(level));
	return (fileInfo.exists() && fileInfo.size()==dimX*dimY*dimZ);
}

void RawIVSource::incrementProgress(int amount)
{
	static int delay = 0;
	m_ProgressValue+=amount;
	delay+=amount;
	//qDebug("Current: %d", m_ProgressValue);
	if (delay>500000) {
		m_ProgressDialog->setProgress(m_ProgressValue);
		qApp->processEvents();
		delay = 0;
		if (m_ProgressDialog->wasCancelled()) {
			throw UserCancelException();
		}
	}
}

int RawIVSource::determineCacheWorkAmount()
{
	bool moreLevels;
	unsigned int targetLevel;
	unsigned int newDimX;
	unsigned int newDimY;
	unsigned int newDimZ;
	unsigned int totalAmount = 0;


	if ( (getDimX()>1 || getDimY()>1 || getDimZ()>1) ) {
		if (m_DataType==DataTypeByte) { // byte data, there is no level 0 mipmap
			targetLevel = 1;
			totalAmount+=getDimZ()*getDimX()*getDimY();
			// calculate the new dimensions
			newDimX = (getDimX()+1)>>1;
			newDimY = (getDimY()+1)>>1;
			newDimZ = (getDimZ()+1)>>1;
			// set targetlevel to 2
			targetLevel++;
		}
		else if (m_DataType==DataTypeShort) { // create a level 0 mipmap
			targetLevel = 0;
			moreLevels = getDimX()>1||getDimY()>1||getDimZ()>1;
			totalAmount+=2*getDimZ()*getDimX()*getDimY();

			// calculate the new dimensions
			newDimX = getDimX();
			newDimY = getDimY();
			newDimZ = getDimZ();
			// set targetlevel to 1
			targetLevel++;

		}
		else { // DataTypeFloat
			targetLevel = 0;
			moreLevels = getDimX()>1||getDimY()>1||getDimZ()>1;
			totalAmount+=2*getDimZ()*getDimX()*getDimY();

			// calculate the new dimensions
			newDimX = getDimX();
			newDimY = getDimY();
			newDimZ = getDimZ();
			// set targetlevel to 1
			targetLevel++;
		}

		while (moreLevels) {

			// create the level
			moreLevels = newDimX>1||newDimY>1||newDimZ>1;

			totalAmount+=newDimZ*newDimX*newDimY;

			// calculate the new dimensions
			newDimX = (newDimX+1)>>1;
			newDimY = (newDimY+1)>>1;
			newDimZ = (newDimZ+1)>>1;
			// increment the target level
			targetLevel++;
		}

	}
	qDebug("total: %d", totalAmount);
	return totalAmount;

}
