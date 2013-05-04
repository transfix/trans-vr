/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane 2002-2003 <thanea@ices.utexas.edu>
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

//////////////////////////////////////////////////////////////////////
//
// VolumeFileSource.cpp: implementation of the VolumeFileSource class.
//
//////////////////////////////////////////////////////////////////////

#include <boost/current_function.hpp>

#include <qstringlist.h>
#include <qdatetime.h>
#include <qapplication.h>

#include <Q3ProgressDialog>

#include <VolumeFileTypes/VolumeFileSource.h>
#include <math.h>
#include <stdlib.h>

// libcontour include for contour spectrum
#include <Contour/contour.h>
#include <Contour/datasetreg3.h>
// libcontourtree include for contour tree
#include <contourtree/computeCT.h>

#include <ByteOrder/ByteSwapping.h>


const Q_INT32 CACHE_RECORD_VERSION_NUMBER = 8;

//#if !defined(QT_LARGEFILE_SUPPORT)
//#error "Largefile support is not enabled in Qt!"
//#endif

//#if !WIN32
//#ifndef _LARGEFILE64_SOURCE
//#error "wtf!?"
//#endif
//#endif


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeFileSource::VolumeFileSource(const QString& filename, const QString& cacheDir) 
  :  m_CacheDir(cacheDir), m_FileInfo(filename)//, m_VolumeFile(0)
{
	setDefaults();
	//qDebug("sizeof(off_t) = %d", sizeof(off_t));
}

VolumeFileSource::~VolumeFileSource()
{
	close();
	destroyBuffer();

#ifdef VFS_USE_THREADING
	for (int i=0;  i<m_WorkerList.count(); i++) {
		WorkerThread *curr = m_WorkerList.at(i);

		// wait for the thread to finish (why isn't there a kill() method?)
		if (curr->running())
			curr->wait();

		// delete the thread
		delete curr;
		curr = NULL;
	}
#endif
}


bool VolumeFileSource::open(QWidget* parent)
{
	if (!m_IsOpen) {
	  //get loader calls return a VolumeFile object allocated with new that we must delete, so use boost::scoped_ptr
	  m_VolumeFile.reset(VolumeFileFactory::ms_MainFactory.getLoader(m_FileInfo.absFilePath()));
		if (!m_VolumeFile) qDebug("failed to get file loader");
		if (m_VolumeFile && m_VolumeFile->attachToFile(m_FileInfo.absFilePath()) && readHeader() && allocateBuffer( getDimX()*getDimY()*3 /* 3 slices */)) {
			// success
			m_IsOpen = true;
			// check for and create cache if needed
			// use progressbar to show progress
			m_ProgressDialog = new Q3ProgressDialog("Creating a cache of preprocess work. \nThis happens the first time you load a file, or when a file has changed", 
				"Cancel", 100, parent, "Creating cache", true);
			m_ProgressValue = 0;
			m_OnePercentProgress = determineCacheWorkAmount() / 100;
			try {
				if (!createCache(m_FileInfo.fileName(), m_CacheDir.absPath())) {
					// failed to create cache
					close();
					setError(QString("%1: Error creating cache\nMake sure there is enough harddisk space for the cache").arg(BOOST_CURRENT_FUNCTION), true);
				}
			}
			catch(const UserCancelException&) {
				close();
				setError(QString("%1: Error creating cache\nCancelled by user").arg(BOOST_CURRENT_FUNCTION), true);
			}
			delete m_ProgressDialog;
			m_ProgressDialog = 0;
		}
		else { 
			// failure
			close();
			setError(QString("%1: Error opening file").arg(BOOST_CURRENT_FUNCTION), true);
		}
	}
	return m_IsOpen;
}

void VolumeFileSource::close()
{
	if (m_IsOpen) {
	  //delete m_VolumeFile;
	  //m_VolumeFile = 0;
	  m_VolumeFile.reset();
		m_IsOpen = false;
	}

	if (m_FuncMin) {
		delete [] m_FuncMin;
		m_FuncMin = 0;
	}
	if (m_FuncMax) {
		delete [] m_FuncMax;
		m_FuncMax = 0;
	}
}

void VolumeFileSource::fillThumbnail(char*, unsigned int, unsigned int, unsigned int, uint, uint)
{
/* will not be implemented.  I will remove this from the base class later */	
}

void VolumeFileSource::fillData(char* data, uint xMin, uint yMin, uint zMin,
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
#ifdef LARGEFILE_KLUDGE
	PFile file;
#else
	QFile file;
#endif

#if 0
	//special case for 2D images (volumes with zDim == 1)
	if(strideZ==1)
	  {
	    zMax = zMin = 0;
	    strideZ = strideX; //fake it
	  }
#endif

	// check that strides are equal
	if (strideX!=strideY || strideY!=strideZ) {
	  setError(QString("%1: Invalid query.  The stride in each dimension must be identical (stride: %2 %3 %4)").arg(BOOST_CURRENT_FUNCTION).arg(strideX).arg(strideY).arg(strideZ));
	}

	// find the correct level while keeping track of the level dimensions
	if (strideX==1) {
		// perform extraction from level 0
		if (m_VolumeFile->getVariableType(variable) == VolumeFile::Char) {
			qDebug("Extract from level 0");
			QTime t;
			t.start();
			qDebug("extractFromRawIVFile()");
			extractFromRawIVFile(data, xMin, yMin, zMin, xMax, yMax, zMax, variable, timeStep);
			qDebug("Elapsed time extracting from file: %d", t.elapsed());
		}
		else {
			// open the file
			file.setName(mipmapLevelFileName(level, variable, timeStep));
			file.open(QIODevice::ReadOnly | QIODevice::Unbuffered);

			// perform extraction
			qDebug("Extract from level 0");
			QTime t;
			t.start();
			qDebug("extractFromCacheFile(%d, %d, %d, %d, %d, %d, %d, %d, %d)", xMin, yMin, zMin, xMax, yMax, zMax, getDimX(), getDimY(), getDimZ());
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
		file.setName(mipmapLevelFileName(level, variable, timeStep));
		file.open(QIODevice::ReadOnly);
		printf("VolumeFileSource::fillData()\n");
		// perform extraction
		qDebug("Extract from level %d", level);
		QTime t;
		t.start();
		qDebug("extractFromCacheFile(%d, %d, %d, %d, %d, %d, %d, %d, %d)", xMin, yMin, zMin, xMax, yMax, zMax, newDimX, newDimY, newDimZ);
		extractFromCacheFile(data, xMin, yMin, zMin, xMax, yMax, zMax, newDimX, newDimY, newDimZ, file);
		qDebug("Elapsed time extracting from file: %d", t.elapsed());
		// close the file
		file.close();

	}
}

bool VolumeFileSource::fillGradientData(char* data,
		uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep)
{
	unsigned int strideX = (xDim==1?1:(xMax-xMin)/(xDim-1));
	unsigned int strideY = (yDim==1?1:(yMax-yMin)/(yDim-1));
	unsigned int strideZ = (zDim==1?1:(zMax-zMin)/(zDim-1));
	unsigned int newDimX = getDimX();
	unsigned int newDimY = getDimY();
	unsigned int newDimZ = getDimZ();
	unsigned int level = 0;

	printf("/nVolumeFileSource::fillGradientData()\n\n");

#ifdef LARGEFILE_KLUDGE
	PFile file;
#else
	QFile file;
#endif

	// check that strides are equal
	if (strideX!=strideY || strideY!=strideZ) {
	  setError(QString("%1: Invalid query.  The stride in each dimension must be identical (stride: %2 %3 %4)").arg(BOOST_CURRENT_FUNCTION).arg(strideX).arg(strideY).arg(strideZ));
		return false;
	}

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
	file.setName(gradientLevelFileName(level, variable, timeStep));
	printf("\nGradient data file name: %s\n\n", gradientLevelFileName(level, variable, timeStep).data());
	file.open(QIODevice::ReadOnly);

	// perform extraction
	qDebug("Extract gradient from level %d", level);
	extractFromGradientCacheFile(data, xMin, yMin, zMin, xMax, yMax, zMax, newDimX, newDimY, newDimZ, file);

	// close the file
	file.close();

	return true;
}

bool VolumeFileSource::readRawData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, uint variable, uint timeStep)
{
	// read data directly from the VolumeFile instance
	uint width = xMax-xMin+1;
	uint height = yMax-yMin+1;
	uint depth = zMax-zMin+1;
	uint j,k, readDims;
	char *readBuf;
	bool ret = true;

	// the dimensions of a clipped slice
	readDims = height * getDimX();

#if 0
	// this is the original read method
	for (k=0; k<depth; k++) {
		// loop through each line
		for (j=0; j<height; j++) {
			// read a line
			ret = m_VolumeFile->readData((*)data+(k*width*height + j*width), 
				(zMin+k)*getDimX()*getDimY()+((yMin+j)*getDimX())+xMin,
				width, variable, timeStep);
		}
	}

	// This read method should be more disk friendly than doing a
	// bunch of scanline sized reads (ouch!).
	
	// this is the read offset for a slice
	// (zMin+k)*getDimX()*getDimY()+yMin*getDimX()
	// and the size is
	// readDims
	
	readBuf = new char [readDims*sizeof()];
	// loop through each slice
	for (k=0; k < depth; k++) {
		// read the clipped slice
		ret = m_VolumeFile->readData((*)readBuf,
																 (zMin+k)*getDimX()*getDimY()+yMin*getDimX(),
																 readDims, variable, timeStep);
		// copy from the readBuf to the data pointer
		for (j=0; j < height; j++) {
			memcpy((*)data + (k*width*height + j*width),
						 (*)readBuf + (j*getDimX()) + xMin,
						 width*sizeof());
		}
	}
	delete [] readBuf;
#endif
	
#if 1
	// pick the read function based on data type
	switch(m_VolumeFile->getVariableType(variable))
	{
		case VolumeFile::Char:
		{
			readBuf = new char [readDims*sizeof(unsigned char)];
			// loop through each slice
			for (k=0; k < depth && ret; k++) {
				// read the clipped slice
				ret = m_VolumeFile->readCharData((unsigned char*)readBuf,
																 ((qulonglong)(zMin+k)*getDimX()*getDimY())+yMin*getDimX(),
																 readDims, variable, timeStep);
				// copy from the readBuf to the data pointer
				for (j=0; j < height; j++) {
					memcpy((unsigned char*)data + (k*width*height + j*width),
								 (unsigned char*)readBuf + (j*getDimX()) + xMin,
								 width*sizeof(unsigned char));
				}
			}
			delete [] readBuf;
			break;
		}
		case VolumeFile::Short:
		{
			readBuf = new char [readDims*sizeof(unsigned short)];
			// loop through each slice
			for (k=0; k < depth; k++) {
				// read the clipped slice
				ret = m_VolumeFile->readShortData((unsigned short*)readBuf,
																 ((qulonglong)(zMin+k)*getDimX()*getDimY())+yMin*getDimX(),
																 readDims, variable, timeStep);
				// copy from the readBuf to the data pointer
				for (j=0; j < height; j++) {
					memcpy((unsigned short*)data + (k*width*height + j*width),
								 (unsigned short*)readBuf + (j*getDimX()) + xMin,
								 width*sizeof(unsigned short));
				}
			}
			delete [] readBuf;
			break;
		}
		case VolumeFile::Long:
		{
			readBuf = new char [readDims*sizeof(unsigned int)];
			// loop through each slice
			for (k=0; k < depth; k++) {
				// read the clipped slice
				ret = m_VolumeFile->readLongData((unsigned int*)readBuf,
																 ((qulonglong)(zMin+k)*getDimX()*getDimY())+yMin*getDimX(),
																 readDims, variable, timeStep);
				// copy from the readBuf to the data pointer
				for (j=0; j < height; j++) {
					memcpy((unsigned int*)data + (k*width*height + j*width),
								 (unsigned int*)readBuf + (j*getDimX()) + xMin,
								 width*sizeof(unsigned int));
				}
			}
			delete [] readBuf;
			break;
		}
		case VolumeFile::Float:
		{
			readBuf = new char [readDims*sizeof(float)];
			// loop through each slice
			for (k=0; k < depth; k++) {
				// read the clipped slice
				ret = m_VolumeFile->readFloatData((float*)readBuf,
																 ((qulonglong)(zMin+k)*getDimX()*getDimY())+yMin*getDimX(),
																 readDims, variable, timeStep);
				// copy from the readBuf to the data pointer
				for (j=0; j < height; j++) {
					memcpy((float*)data + (k*width*height + j*width),
								 (float*)readBuf + (j*getDimX()) + xMin,
								 width*sizeof(float));
				}
			}
			delete [] readBuf;
			break;
		}
		case VolumeFile::Double:
		{
			readBuf = new char [readDims*sizeof(double)];
			// loop through each slice
			for (k=0; k < depth; k++) {
				// read the clipped slice
				ret = m_VolumeFile->readDoubleData((double*)readBuf,
																 ((qulonglong)(zMin+k)*getDimX()*getDimY())+yMin*getDimX(),
																 readDims, variable, timeStep);
				// copy from the readBuf to the data pointer
				for (j=0; j < height; j++) {
					memcpy((double*)data + (k*width*height + j*width),
								 (double*)readBuf + (j*getDimX()) + xMin,
								 width*sizeof(double));
				}
			}
			delete [] readBuf;
			break;
		}
		default:
		{
			ret = false;
			break;
		}
	}
#else
	// pick the read function based on data type
	switch(m_VolumeFile->getVariableType(variable))
	{
		case VolumeFile::Char:
		{
			// loop through each slice
			for (k=0; k<depth; k++) {
				// loop through each line
				for (j=0; j<height; j++) {
					// read a line
					ret = m_VolumeFile->readCharData((unsigned char*)data+(k*width*height + j*width), 
						((qulonglong)(zMin+k)*getDimX()*getDimY())+((yMin+j)*getDimX())+xMin,
						width, variable, timeStep);
				}
			}
			break;
		}
		case VolumeFile::Short:
		{
			// loop through each slice
			for (k=0; k<depth; k++) {
				// loop through each line
				for (j=0; j<height; j++) {
					// read a line
					ret = m_VolumeFile->readShortData((unsigned short*)data+(k*width*height + j*width), 
						((qulonglong)(zMin+k)*getDimX()*getDimY())+((yMin+j)*getDimX())+xMin,
						width, variable, timeStep);
				}
			}
			break;
		}
		case VolumeFile::Long:
		{
			// loop through each slice
			for (k=0; k<depth; k++) {
				// loop through each line
				for (j=0; j<height; j++) {
					// read a line
					ret = m_VolumeFile->readLongData((unsigned int*)data+(k*width*height + j*width), 
						((qulonglong)(zMin+k)*getDimX()*getDimY())+((yMin+j)*getDimX())+xMin,
						width, variable, timeStep);
				}
			}
			break;
		}
		case VolumeFile::Float:
		{
			// loop through each slice
			for (k=0; k<depth; k++) {
				// loop through each line
				for (j=0; j<height; j++) {
					// read a line
					ret = m_VolumeFile->readFloatData((float*)data+(k*width*height + j*width), 
						((qulonglong)(zMin+k)*getDimX()*getDimY())+((yMin+j)*getDimX())+xMin,
						width, variable, timeStep);
				}
			}
			break;
		}
		case VolumeFile::Double:
		{
			// loop through each slice
			for (k=0; k<depth; k++) {
				// loop through each line
				for (j=0; j<height; j++) {
					// read a line
					ret = m_VolumeFile->readDoubleData((double*)data+(k*width*height + j*width), 
						((qulonglong)(zMin+k)*getDimX()*getDimY())+((yMin+j)*getDimX())+xMin,
						width, variable, timeStep);
				}
			}
			break;
		}
		default:
		{
			ret = false;
			break;
		}
	}
#endif

	return ret;
}

VolumeSource::DownLoadFrequency VolumeFileSource::interactiveUpdateHint() const
{
	return DLFDelayed;
}

unsigned int VolumeFileSource::getVariableType(unsigned int variable) const
{
	return (unsigned int)m_VolumeFile->getVariableType(variable);
}

QString VolumeFileSource::getVariableName(unsigned int variable) const
{
	return m_VolumeFile->getVariableName(variable);
}

double VolumeFileSource::getFunctionMinimum(unsigned int variable, unsigned int timeStep) const
{
	return m_FuncMin[variable*m_NumTimeSteps + timeStep];
}

double VolumeFileSource::getFunctionMaximum(unsigned int variable, unsigned int timeStep) const
{
	return m_FuncMax[variable*m_NumTimeSteps + timeStep];
}

QString VolumeFileSource::getContourSpectrumFileName(unsigned int variable, unsigned int timeStep)
{
	return contourSpectrumFileName(variable, timeStep);
}

QString VolumeFileSource::getContourTreeFileName(unsigned int variable, unsigned int timeStep)
{
	return contourTreeFileName(variable, timeStep);
}

#ifdef VFS_USE_THREADING
void VolumeFileSource::cleanUpWorkerThread(int thid)
{
	int i;

	for (i=0; i<m_WorkerList.count(); i++) {
		WorkerThread *curr = m_WorkerList.at(i);

		if (curr->getID() == thid) {
			// remove the thread and delete it
			m_WorkerList.remove(curr);
			delete curr;
			
			// all done
			break;
		}
	}
}
#endif

void VolumeFileSource::computeContourSpectrum(QObject *obj, unsigned int variable, unsigned int timeStep)
{
#ifdef VFS_USE_THREADING
	ContourSpectrumArgs *data = new ContourSpectrumArgs;
	int i, thid = (getNumTimeSteps() * timeStep + variable+1)*1; // a unique-ish id
	bool alreadyRunning = false;
	
	// make sure a thread with the same id is not running
	for (i=0; i < m_WorkerList.count(); i++) {
		WorkerThread *curr = m_WorkerList.at(i);

		if (curr != NULL && curr->getID() == thid) {
			alreadyRunning = true;
			break;
		}
	}
	
	// start the computation if it is not already running
	if (!alreadyRunning) {
		// fill the argument data structure
		data = new ContourSpectrumArgs;
		data->thisPtr = this;
		data->var = variable;
		data->timeStep = timeStep;

		// create the worker thread
		WorkerThread *drone = new WorkerThread(obj, thid, callCreateContourSpectrum, (void *)data);
		// start it
		drone->start();

		// then add it to some list of running threads
		m_WorkerList.append(drone);
	}
#else
	createContourSpectrum(variable, timeStep);
#endif
}

void VolumeFileSource::computeContourTree(QObject *obj, unsigned int variable, unsigned int timeStep)
{
	// XXX This code is not safe.
				// Until createContourXXX functions are made threadsafe, this code
				// should not be enabled. Specifically, the functions make changes to
				// shared variables m_Buffer and m_BufferSize (possibly others). 
	// XXX Addendum: I think it's fixed now. Keep an eye on this space.
#ifdef VFS_USE_THREADING
	ContourTreeArgs *data = new ContourTreeArgs;
	int i, thid = (getNumTimeSteps() * timeStep + variable+1)*2; // a unique-ish id
	bool alreadyRunning = false;
	
	// make sure a thread with the same id is not running
	for (i=0; i < m_WorkerList.count(); i++) {
		WorkerThread *curr = m_WorkerList.at(i);

		if (curr != NULL && curr->getID() == thid) {
			alreadyRunning = true;
			break;
		}
	}
	
	// start the computation if it is not already running
	if (!alreadyRunning) {
		// fill the argument data structure
		data = new ContourTreeArgs;
		data->thisPtr = this;
		data->var = variable;
		data->timeStep = timeStep;

		// create the worker thread
		WorkerThread *drone = new WorkerThread(obj, thid, callCreateContourTree, (void *)data);
		// start it
		drone->start();

		// then add it to some list of running threads
		m_WorkerList.append(drone);
	}
#else
	createContourTree(variable, timeStep);
#endif
}

void VolumeFileSource::callCreateContourSpectrum(void *args)
{
	ContourSpectrumArgs *data = (ContourSpectrumArgs *)args;

	data->thisPtr->createContourSpectrum(data->var, data->timeStep);

	delete data;
}

void VolumeFileSource::callCreateContourTree(void *args)
{
	ContourTreeArgs *data = (ContourTreeArgs *)args;

	data->thisPtr->createContourTree(data->var, data->timeStep);

	delete data;
}

QString VolumeFileSource::getFilter()
{
	return VolumeFileFactory::ms_MainFactory.getFilterString();
}

void VolumeFileSource::setDefaults()
{
	m_IsOpen = false;
	m_Buffer = 0;
	m_BufferSize = 0;
	m_ProgressDialog = 0;
	m_ProgressValue = 0;
	m_OnePercentProgress = 0;
	//m_VolumeFile = 0;
	m_FuncMin = 0;
	m_FuncMax = 0;
}

bool VolumeFileSource::allocateBuffer(unsigned int size)
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

bool VolumeFileSource::forceAllocateBuffer(unsigned int size)
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

void VolumeFileSource::destroyBuffer()
{
	if (m_Buffer)
		delete [] m_Buffer;
	m_Buffer = 0;
	m_BufferSize = 0;
}

bool VolumeFileSource::readHeader()
{

	// get the mins
	setMinX(m_VolumeFile->getMinX());
	setMinY(m_VolumeFile->getMinY());
	setMinZ(m_VolumeFile->getMinZ());
	setMinT(m_VolumeFile->getMinT());

	// get the maxs
	setMaxX(m_VolumeFile->getMaxX());
	setMaxY(m_VolumeFile->getMaxY());
	setMaxZ(m_VolumeFile->getMaxZ());
	setMaxT(m_VolumeFile->getMaxT());

	// get the dimensions
	setDimX(m_VolumeFile->getDimX());
	setDimY(m_VolumeFile->getDimY());
	setDimZ(m_VolumeFile->getDimZ());

	qDebug("Dimensions: %dx%dx%d", getDimX(),getDimY(),getDimZ());

	m_NumVariables = m_VolumeFile->getNumVariables();
	m_NumTimeSteps = m_VolumeFile->getNumTimeSteps();

	// init the function min/max arrays
	m_FuncMin = new double [m_NumVariables*m_NumTimeSteps];
	m_FuncMax = new double [m_NumVariables*m_NumTimeSteps];

	return true;
}

void VolumeFileSource::extractFromRawIVFile(char* data, uint xMin, uint yMin, uint zMin, 
									   uint xMax, uint yMax, uint zMax, unsigned int variable, unsigned int timeStep)
{
	uint width = xMax-xMin+1;
	uint height = yMax-yMin+1;
	uint depth = zMax-zMin+1;
	uint j,k;

	// loop through each slice
	for (k=0; k<depth; k++) {
		// loop through each line
		for (j=0; j<height; j++) {
			// read a line
			m_VolumeFile->readCharData((unsigned char*)data+(k*width*height + j*width), 
				((qulonglong)(zMin+k)*getDimX()*getDimY())+((yMin+j)*getDimX())+xMin,
				width, variable, timeStep);
		}
	}
}

void VolumeFileSource::extractFromCacheFile(char* data, uint xMin, uint yMin, uint zMin, 
									   uint xMax, uint yMax, uint zMax, 
#ifdef LARGEFILE_KLUDGE
									   uint newDimX, uint newDimY, uint newDimZ, PFile& file)
#else
									   uint newDimX, uint newDimY, uint newDimZ, QFile& file)
#endif
{
	uint width = xMax-xMin+1;
	uint height = yMax-yMin+1;
	uint depth = zMax-zMin+1;
	uint j,k;
#ifndef LARGEFILE_KLUDGE
	qDebug("Cache file: %s",file.name().ascii());
#endif
	// loop through each slice
	for (k=0; k<depth; k++) {
		// loop through each line
		for (j=0; j<height; j++) {
			// go to the start of the "scanline"
			if(file.at(((qulonglong)(zMin+k))*newDimX*newDimY+((yMin+j)*newDimX)+xMin)==false)
			{
				qDebug("Error seeking in cache!");
				return;
			}

			// read in a line
			if(file.readBlock(data+(k*width*height + j*width), width)==-1)
			{
				qDebug("Error reading cache!");
				return;
			}
		}
	}
}
	
void VolumeFileSource::extractFromGradientCacheFile(char* data,
		uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, 
#ifdef LARGEFILE_KLUDGE
		uint newDimX, uint newDimY, uint newDimZ, PFile& file)
#else
		uint newDimX, uint newDimY, uint newDimZ, QFile& file)
#endif
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
#ifdef LARGEFILE_KLUDGE
			file.at(((off_t)((zMin+k)*newDimX*newDimY+((yMin+j)*newDimX)+xMin))*4);
#else
			file.at(((qulonglong)((zMin+k)*newDimX*newDimY+((yMin+j)*newDimX)+xMin))*4);
#endif

			// read in a line
			file.readBlock(data+(k*width*height + j*width)*4, width*4);
		}
	}
}

bool VolumeFileSource::createCache(const QString& fileName, const QString& cacheRoot)
{
	QDir dir(cacheRoot);

	if (!dir.exists()) {
	  setError(QString("%1: The cache directory does not exist").arg(BOOST_CURRENT_FUNCTION), true);
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

void VolumeFileSource::createNewCache( QFileInfo fileInfo, QDir cacheDir)
{
	// create the mipmap levels
	unsigned int v, t;
	for (v=0; v<m_VolumeFile->getNumVariables(); v++) {
		for (t=0; t<m_VolumeFile->getNumTimeSteps(); t++) {
			// mipmap levels
			createMipmapLevels(v, t);
			// contour spectrum
			//createContourSpectrum(v, t);
		}
	}

	// create the record file
	createCacheRecordFile(fileInfo, cacheDir);

}

void VolumeFileSource::createCacheRecordFile(QFileInfo fileInfo, QDir cacheDir)
{

	// open the file
	QString filename(cacheDir.absPath() + "/" + fileInfo.baseName() + "_Record" + ".cache");
	QFile file(filename);
	file.open(QIODevice::WriteOnly);

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

	// output the min/max for each function
	unsigned int v, t;
	for (v=0; v<m_VolumeFile->getNumVariables(); v++)
		for (t=0; t<m_VolumeFile->getNumTimeSteps(); t++) {
			stream << m_FuncMin[v*m_VolumeFile->getNumTimeSteps()+t];
			stream << m_FuncMax[v*m_VolumeFile->getNumTimeSteps()+t];
		}

	file.close();
}

void VolumeFileSource::clearCache(QDir cacheDir)
{
	// get the filtered list of files
	QStringList files = cacheDir.entryList("*.cache; *.spectrum; *.tree; *.grad");

	QString file;

	// loop though the list of files
	for (QStringList::Iterator it = files.begin(); it != files.end(); ++it) {
		file = *it;
		//delete each file
		cacheDir.remove(file);
	}
}

void VolumeFileSource::createMipmapLevels(unsigned int variable, unsigned int timeStep)
{
#ifdef LARGEFILE_KLUDGE
	PFile source, target;
#else
	QFile source, target;
#endif
	bool moreLevels;
	unsigned int targetLevel;
	unsigned int newDimX;
	unsigned int newDimY;
	unsigned int newDimZ;

	QTime t;
	t.start();

	if ( (getDimX()>1 || getDimY()>1 || getDimZ()>1) ) {
		if (m_VolumeFile->getVariableType(variable)==VolumeFile::Char) { // byte data, there is no level 0 mipmap
			// but there is a level 0 gradient
			/*target.setName(gradientLevelFileName(0, variable, timeStep));
			target.open( IO_WriteOnly );
			createGradientFromCharRawiv(target, variable, timeStep);
			target.close();*/
			// create level 1
			targetLevel = 1;
			target.setName(mipmapLevelFileName(targetLevel, variable, timeStep));
			target.open( QIODevice::WriteOnly );
			moreLevels = createMipmapLevelFromRawIVGaussian(target, variable, timeStep);
			target.close();

			// calculate the new dimensions
			newDimX = (getDimX()+1)>>1;
			newDimY = (getDimY()+1)>>1;
			newDimZ = (getDimZ()+1)>>1;
			
			// and then get the gradient for level 1
			// (the new dimensions are valid for the level 1 mipmap)
			/*source.setName(mipmapLevelFileName(targetLevel, variable, timeStep));
			source.open( IO_ReadOnly );
			target.setName(gradientLevelFileName(targetLevel, variable, timeStep));
			target.open( IO_WriteOnly );
			createGradientFromCacheFile(source, target, newDimX, newDimY, newDimZ);
			source.close();
			target.close();*/

			// set targetlevel to 2
			targetLevel++;
		}
		else if (m_VolumeFile->getVariableType(variable)==VolumeFile::Short) { // create a level 0 mipmap
			targetLevel = 0;
			target.setName(mipmapLevelFileName(targetLevel, variable, timeStep));
			target.open( QIODevice::WriteOnly );
			moreLevels = createMipmapLevelZeroFromShortRawIV(target, variable, timeStep);
			target.close();
			// create a level 0 gradient
			/*target.setName(gradientLevelFileName(targetLevel, variable, timeStep));
			target.open( IO_WriteOnly );
			createGradientFromShortRawiv(target, variable, timeStep);
			target.close();*/

			// calculate the new dimensions
			newDimX = getDimX();
			newDimY = getDimY();
			newDimZ = getDimZ();
			// set targetlevel to 1
			targetLevel++;

		}
		else { // DataTypeFloat
			targetLevel = 0;
			target.setName(mipmapLevelFileName(targetLevel, variable, timeStep));
			target.open( QIODevice::WriteOnly );
			moreLevels = createMipmapLevelZeroFromFloatRawIV(target, variable, timeStep);
			target.close();
			// create a level 0 gradient
			/*target.setName(gradientLevelFileName(targetLevel, variable, timeStep));
			target.open( IO_WriteOnly );
			createGradientFromFloatRawiv(target, variable, timeStep);
			target.close();*/

			// calculate the new dimensions
			newDimX = getDimX();
			newDimY = getDimY();
			newDimZ = getDimZ();
			// set targetlevel to 1
			targetLevel++;
		}

		while (moreLevels) {
			// open the source and target files
			target.setName(mipmapLevelFileName(targetLevel, variable, timeStep));
			target.open( QIODevice::WriteOnly );
			source.setName(mipmapLevelFileName(targetLevel-1, variable, timeStep));
			source.open( QIODevice::ReadOnly );

			// create the level
			moreLevels = createMipmapLevelFromCacheFileGaussian(source, target, newDimX, newDimY, newDimZ);

			// close the files
			target.close();
			source.close();

			// calculate the new dimensions
			newDimX = (newDimX+1)>>1;
			newDimY = (newDimY+1)>>1;
			newDimZ = (newDimZ+1)>>1;
			
			// and then get the gradient for this level
			/*source.setName(mipmapLevelFileName(targetLevel, variable, timeStep));
			source.open( IO_ReadOnly );
			target.setName(gradientLevelFileName(targetLevel, variable, timeStep));
			target.open( IO_WriteOnly );
			createGradientFromCacheFile(source, target, newDimX, newDimY, newDimZ);
			source.close();
			target.close();*/

			// increment the target level
			targetLevel++;
		}

	}

	qDebug("Timer to create mipmap levels: %d", t.elapsed());
}

void VolumeFileSource::createContourSpectrum(unsigned int variable, unsigned int timeStep)
{
	// no largefile kludge needed.
	QFile source, target;
	bool spectrumDone=false;
	unsigned int sourceLevel;
	qulonglong newDimX;
	qulonglong newDimY;
	qulonglong newDimZ;

	if ( (getDimX()>1 || getDimY()>1 || getDimZ()>1) ) {
		// always start with the level 0 mipmap
		sourceLevel = 0;
		// get the dimensions
		newDimX = getDimX();
		newDimY = getDimY();
		newDimZ = getDimZ();

		while ( !spectrumDone ) {
			if ( (newDimX*newDimY*newDimZ)>(128*128*128) ) {
				// the dimensions for this level are too large, try the next level
				// calculate the new dimensions
				newDimX = (newDimX+1)>>1;
				newDimY = (newDimY+1)>>1;
				newDimZ = (newDimZ+1)>>1;
				// increment sourceLevel
				sourceLevel++;
			}
			else {
				// generate the contour spectrum
				// open the source and target files
				target.setName(contourSpectrumFileName(variable, timeStep));
				source.setName(mipmapLevelFileName(sourceLevel, variable, timeStep));

				// create only if the target does not already exist
				if (!target.exists()) {
					// open our files
					target.open( QIODevice::WriteOnly );
					source.open( QIODevice::ReadOnly );

					qDebug("Computing Contour Spectrum... (Volume Dims: %lux%lux%lu)",
													newDimX,newDimY,newDimZ);

					// calculate the contour spectrum
					if ( sourceLevel==0
							&& m_VolumeFile->getVariableType(variable)==VolumeFile::Char ) {
						// character datasets at level 0 are a special case
						// calculate the contour spectrum from the volume file
						createContourSpectrumFromRawIV(target, variable, timeStep);
					}
					else {
						// calculate the contour spectrum from a cache file
						createContourSpectrumFromCacheFile(source, target, (uint)newDimX, (uint)newDimY, (uint)newDimZ);
					}

					// close the files
					target.close();
					source.close();
				}
				
				// we're done
				spectrumDone = true;
			}
		}
	}
}

void VolumeFileSource::createContourTree(unsigned int variable, unsigned int timeStep)
{
	// no largefile kludge needed.
	QFile source, target;
	bool treeDone=false;
	unsigned int sourceLevel;
	qulonglong newDimX;
	qulonglong newDimY;
	qulonglong newDimZ;

	if ( (getDimX()>1 || getDimY()>1 || getDimZ()>1) ) {
		// always start with the level 0 mipmap
		sourceLevel = 0;
		// get the dimensions
		newDimX = getDimX();
		newDimY = getDimY();
		newDimZ = getDimZ();

		while ( !treeDone ) {
			if ( (newDimX*newDimY*newDimZ)>(128*128*128) ) {
				// the dimensions for this level are too large, try the next level
				// calculate the new dimensions
				newDimX = (newDimX+1)>>1;
				newDimY = (newDimY+1)>>1;
				newDimZ = (newDimZ+1)>>1;
				// increment sourceLevel
				sourceLevel++;
			}
			else {
				// generate the contour tree
				// open the source and target files
				target.setName(contourTreeFileName(variable, timeStep));
				source.setName(mipmapLevelFileName(sourceLevel, variable, timeStep));

				// create only if the target does not already exist
				if(!target.exists()) {
					// open the files
					target.open( QIODevice::WriteOnly );
					source.open( QIODevice::ReadOnly );

					qDebug("Computing Contour Tree... (Volume Dims: %lux%lux%lu)",
													newDimX,newDimY,newDimZ);

					// calculate the contour tree
					if ( sourceLevel==0
							&& m_VolumeFile->getVariableType(variable)==VolumeFile::Char ) {
						// character datasets at level 0 are a special case
						// calculate the contour tree from the volume file
						createContourTreeFromRawIV(target, variable, timeStep);
					}
					else {
						// calculate the contour tree from a cache file
						createContourTreeFromCacheFile(source, target, (uint)newDimX, (uint)newDimY, (uint)newDimZ);
					}
	
					// close the files
					target.close();
					source.close();
				}
				
				// we're done
				treeDone = true;
			}
		}
	}
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

static const qulonglong buffSize = 256;
//
// HEY LOOK! A GLOBAL VARIABLE HAS JUST BEEN DECLARED!
// ( did you notice? )
//

#ifdef LARGEFILE_KLUDGE
bool VolumeFileSource::createMipmapLevelZeroFromShortRawIV(PFile& targetFile, unsigned int variable, unsigned int timeStep)
#else
bool VolumeFileSource::createMipmapLevelZeroFromShortRawIV(QFile& targetFile, unsigned int variable, unsigned int timeStep)
#endif
{
	unsigned short min, max;
	bool minSet = false, maxSet = false;
	qulonglong k,j;
	unsigned short buffer[buffSize];
	unsigned char byteBuffer[buffSize];

	// determine min and max
	qDebug("Finding min and max");
	m_VolumeFile->setPosition(variable, timeStep, 0);

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_VolumeFile->readShortData(buffer, buffSize, variable);
		for (j=0; j<buffSize; j++) {
			checkMin(min, minSet, buffer[j]);
			checkMax(max, maxSet, buffer[j]);
		}
		incrementProgress(buffSize);
	}
	// do remainder
	m_VolumeFile->readShortData(buffer, (getNumVerts()%buffSize), variable);
	for (j=0; j<(getNumVerts()%buffSize); j++) {
		checkMin(min, minSet, buffer[j]);
		checkMax(max, maxSet, buffer[j]);
	}
	incrementProgress(getNumVerts()%buffSize);

	// save the min/max values
	m_FuncMin[variable*m_NumTimeSteps + timeStep] = (double)min;
	m_FuncMax[variable*m_NumTimeSteps + timeStep] = (double)max;

	// convert to byte
	qDebug("Converting to byte");
	m_VolumeFile->setPosition(variable, timeStep, 0);
#ifdef LARGEFILE_KLUDGE
	targetFile.at((off_t)0);
#else
	targetFile.at(0);
#endif

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_VolumeFile->readShortData(buffer, buffSize, variable);
		for (j=0; j<buffSize; j++) {
			byteBuffer[j] = convertToByte(buffer[j], min, max);
		}
		targetFile.writeBlock((char*)byteBuffer, buffSize*sizeof(unsigned char));
		incrementProgress(buffSize);
	}
	// do remainder
	m_VolumeFile->readShortData(buffer, (getNumVerts()%buffSize), variable);
	for (j=0; j<(getNumVerts()%buffSize); j++) {
		byteBuffer[j] = convertToByte(buffer[j], min, max);
	}
	targetFile.writeBlock((char*)byteBuffer, (getNumVerts()%buffSize)*sizeof(unsigned char));
	incrementProgress(getNumVerts()%buffSize);

	return (getDimX()>1 || getDimY()>1 || getDimZ()>1);
}

#ifdef LARGEFILE_KLUDGE
bool VolumeFileSource::createMipmapLevelZeroFromFloatRawIV(PFile& targetFile, unsigned int variable, unsigned int timeStep)
#else
bool VolumeFileSource::createMipmapLevelZeroFromFloatRawIV(QFile& targetFile, unsigned int variable, unsigned int timeStep)
#endif
{
	float min, max;
	bool minSet = false, maxSet = false;
	qulonglong k,j;
	float buffer[buffSize];
	unsigned char byteBuffer[buffSize];

	// determine min and max
	qDebug("Finding min and max");
	m_VolumeFile->setPosition(variable, timeStep, 0);

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_VolumeFile->readFloatData(buffer, buffSize, variable);
		for (j=0; j<buffSize; j++) {
			checkMin(min, minSet, (buffer[j]));
			checkMax(max, maxSet, (buffer[j]));
		}
		incrementProgress(buffSize);
	}
	// do remainder
	m_VolumeFile->readFloatData(buffer, (getNumVerts()%buffSize), variable);
	for (j=0; j<(getNumVerts()%buffSize); j++) {
		checkMin(min, minSet, (buffer[j]));
		checkMax(max, maxSet, (buffer[j]));
	}
	incrementProgress(getNumVerts()%buffSize);

	// save the min/max values
	m_FuncMin[variable*m_NumTimeSteps + timeStep] = (double)min;
	m_FuncMax[variable*m_NumTimeSteps + timeStep] = (double)max;

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
	qDebug("Converting to byte (min/max: %f/%f)", min, max);
	m_VolumeFile->setPosition(variable, timeStep, 0);
#ifdef LARGEFILE_KLUDGE
	targetFile.at((off_t)0);
#else
	targetFile.at(0);
#endif

	for (k=0; k<getNumVerts()/buffSize; k++) {
		m_VolumeFile->readFloatData(buffer, buffSize, variable);
		for (j=0; j<buffSize; j++) {
			byteBuffer[j] = ( convertToByte(buffer[j], min, max) );
		}
		targetFile.writeBlock((char*)byteBuffer, buffSize*sizeof(unsigned char));
		incrementProgress(buffSize);
	}
	// do remainder
	m_VolumeFile->readFloatData(buffer, (getNumVerts()%buffSize), variable);
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

#ifdef LARGEFILE_KLUDGE
bool VolumeFileSource::createMipmapLevelFromRawIVGaussian(PFile& targetFile, unsigned int variable, unsigned int timeStep)
#else
bool VolumeFileSource::createMipmapLevelFromRawIVGaussian(QFile& targetFile, unsigned int variable, unsigned int timeStep)
#endif
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

	unsigned int positiveI, positiveJ;
	unsigned int negativeI, negativeJ;
	qulonglong sliceNum, i, j;
	unsigned int sum;
	unsigned int average;
	unsigned char c_avg;

	// "fake" the minimum and maximum
	// save the min/max values
	m_FuncMin[variable*m_NumTimeSteps + timeStep] = 0.0;
	m_FuncMax[variable*m_NumTimeSteps + timeStep] = 255.0;


	// set the file position
	m_VolumeFile->setPosition(variable, timeStep, 0);
	for (sliceNum=0; sliceNum<getDimZ(); sliceNum+=2) {
		if (sliceNum==0) { // top border
			// read in 3 slices
			m_VolumeFile->readCharData(slice0, getDimX()*getDimY(), variable);
			m_VolumeFile->readCharData(slice1, getDimX()*getDimY(), variable);
			m_VolumeFile->readCharData(slice2, getDimX()*getDimY(), variable);
		}
		else if (sliceNum+1==getDimZ()) { // bottom border
			// swap slices and read in last slice twice
			unsigned char* temp = slice2;
			slice2 = slice0;
			slice0 = temp;
			m_VolumeFile->readCharData(slice1, getDimX()*getDimY(), variable);
			m_VolumeFile->setPosition(variable, timeStep, sliceNum*getDimX()*getDimY());
			m_VolumeFile->readCharData(slice2, getDimX()*getDimY(), variable);
		}
		else { // normal case
			// swap slices and read in two new slices
			unsigned char* temp = slice2;
			slice2 = slice0;
			slice0 = temp;
			m_VolumeFile->readCharData(slice1, getDimX()*getDimY(), variable);
			m_VolumeFile->readCharData(slice2, getDimX()*getDimY(), variable);
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

				//target.writeBlock((char*)&average, 1);

				//fixed for big endian!
				c_avg = (unsigned char)average;
				targetFile.writeBlock((char*)&c_avg, 1);
			}

		}

		incrementProgress(getDimX()*getDimY()*2);

	}
	return (newDimX>1 || newDimY>1 || newDimZ>1);
}

#ifdef LARGEFILE_KLUDGE
bool VolumeFileSource::createMipmapLevelFromCacheFileGaussian(PFile& source, PFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int offset)
#else
bool VolumeFileSource::createMipmapLevelFromCacheFileGaussian(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int offset)
#endif
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

	unsigned int positiveI, positiveJ;
	unsigned int negativeI, negativeJ;
	qulonglong sliceNum, i, j;
	unsigned int sum;
	unsigned int average;
	unsigned char c_avg;

	if (dimX==0 || dimY==0 || dimZ==0) {
		qDebug("WARNING: problem with dimensions.  dimX = %d.  dimY = %d, dimZ = %d", dimX, dimY, dimZ);
	}



	// set the file position
#ifdef LARGEFILE_KLUDGE
	source.at((off_t)offset);
#else
	source.at(offset);
#endif
	for (sliceNum=0; sliceNum<dimZ; sliceNum+=2) {
		if (sliceNum==0) { // top border
			// read in 3 slices
			source.readBlock((char*)slice0, dimX*dimY);
			source.readBlock((char*)slice1, dimX*dimY);
			source.readBlock((char*)slice2, dimX*dimY);
		}
		else if (sliceNum+1==dimZ) { // bottom border
			// swap slices and read in last slice twice
			unsigned char* temp = slice2;
			slice2 = slice0;
			slice0 = temp;
			source.readBlock((char*)slice1, 1*dimX*dimY);
#ifdef LARGEFILE_KLUDGE
			source.at((off_t)sliceNum*dimX*dimY+(qulonglong)offset);
#else
			source.at(sliceNum*dimX*dimY+(qulonglong)offset);
#endif
			source.readBlock((char*)slice2, 1*dimX*dimY);
		}
		else { // normal case
			// swap slices and read in two new slices
			unsigned char* temp = slice2;
			slice2 = slice0;
			slice0 = temp;
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
				// !! holy crap this is slow (writing 1 byte at a time)
				// addendum: it appears that writeBlock is buffered. -> not that slow
				//target.writeBlock((char*)&average, 1);

				//fixed for big endian!
				c_avg = (unsigned char)average;
				target.writeBlock((char*)&c_avg, 1);
			}

		}

		incrementProgress(dimX*dimY*2);

	}
	return (newDimX>1 || newDimY>1 || newDimZ>1);
}

#ifdef LARGEFILE_KLUDGE
bool VolumeFileSource::createMipmapLevelFromRawIV(PFile& targetFile, unsigned int variable, unsigned int timeStep)
#else
bool VolumeFileSource::createMipmapLevelFromRawIV(QFile& targetFile, unsigned int variable, unsigned int timeStep)
#endif
{
	// average every eight voxels to form one voxel in the next mipmap level

	// we will read in 2 slices at a time, and perform the averaging

	// prepare the new dimensions by dividing by 2
	unsigned int newDimX = (getDimX()+1)>>1;
	unsigned int newDimY = (getDimY()+1)>>1;
	unsigned int newDimZ = (getDimZ()+1)>>1;


	qulonglong sliceNum, i, j;
	unsigned int positiveI, positiveJ;
	unsigned int samples[8];
	unsigned char average;
	// set the file position
	m_VolumeFile->setPosition(variable, timeStep, 0);
	for (sliceNum=0; sliceNum<getDimZ(); sliceNum+=2) {

		if (sliceNum+1 >= getDimZ() ) {
			// we only have one slice left
			// read it in twice
			m_VolumeFile->readCharData(m_Buffer, 1*getDimX()*getDimY(), variable);
			m_VolumeFile->setPosition(variable, timeStep, sliceNum*getDimX()*(qulonglong)getDimY());
			m_VolumeFile->readCharData(m_Buffer+getDimX()+getDimY(), 1*getDimX()*getDimY(), variable);
		}
		else {
			// read in 2 slices
			m_VolumeFile->readCharData(m_Buffer, 2*getDimX()*getDimY(), variable);

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

#ifdef LARGEFILE_KLUDGE
bool VolumeFileSource::createMipmapLevelFromCacheFile(PFile& source, PFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ)
#else
bool VolumeFileSource::createMipmapLevelFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ)
#endif
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


	qulonglong sliceNum, i, j;
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
#ifdef LARGEFILE_KLUDGE
			source.at((off_t)sliceNum*dimX*(qulonglong)dimY);
#else
			source.at(sliceNum*dimX*(qulonglong)dimY);
#endif
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

#ifdef LARGEFILE_KLUDGE
void VolumeFileSource::createGradientFromCharRawiv(PFile& targetFile, unsigned int variable, unsigned int timeStep)
#else
void VolumeFileSource::createGradientFromCharRawiv(QFile& targetFile, unsigned int variable, unsigned int timeStep)
#endif
{
	// make sure there is enough space for 3 slices of char data
	allocateBuffer( getDimX()*getDimY()*3 /* 3 slices */);

	int sliceNum, x=0, y=0, inputindex=0;
	int negXOffset, posXOffset;
	int negYOffset, posYOffset;
	int negZOffset, posZOffset;
	unsigned int sliceSize = getDimX()*getDimY(), width=getDimX();
	unsigned char grad[4];
	double length, dx,dy,dz;


	// set the file position
	m_VolumeFile->setPosition(variable, timeStep, 0);

	for (sliceNum=0; sliceNum<getDimZ(); sliceNum++) {

		// prepare the slice buffer
		if (sliceNum == 0) {
			// this is the first slice
			// zero out the first slice in the buffer
			memset(m_Buffer, 0, sliceSize);
			// and read the first two slices of the volume
			m_VolumeFile->readCharData(m_Buffer+sliceSize, 2*sliceSize, variable);
		}	
		else if (sliceNum+1 >= getDimZ() ) {
			// this is the last slice
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize, sliceSize);
			memcpy(m_Buffer+sliceSize, m_Buffer+sliceSize*2, sliceSize);
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// then zero the last slice in the buffer
			memset(m_Buffer+sliceSize*2, 0, sliceSize);
		}
		else {
			// some slice in the middle of the volume
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize, sliceSize);
			memcpy(m_Buffer+sliceSize, m_Buffer+sliceSize*2, sliceSize);
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// read in the next slice
			m_VolumeFile->readCharData(m_Buffer+sliceSize*2, sliceSize, variable);
		}
		
		// compute the gradient for this slice
		//
		// get the z offset
		if (sliceNum == 0) { // border offset
			negZOffset = 0; posZOffset = sliceSize;
		}
		else if (sliceNum == getDimZ()-1) { // border offset
			negZOffset = -sliceSize; posZOffset = 0;
		}
		else { // normal offset
			negZOffset = -sliceSize; posZOffset = sliceSize;
		}
		//
		for (y=0; y < getDimY(); y++) {
			// get the y offset
			if (y==0) { // border offset
				negYOffset = 0; posYOffset = width;
			}
			else if (y==getDimY()-1) { // border offset
				negYOffset = -width; posYOffset = 0;
			}				
			else { // normal offset
				negYOffset = -width; posYOffset = width;
			}

			// do border case
			negXOffset = 0; posXOffset = 1;
			inputindex = sliceSize + y*width + x;
	
			dx = (double)m_Buffer[inputindex+negXOffset] - (double)m_Buffer[inputindex+posXOffset];
			dy = (double)m_Buffer[inputindex+negYOffset] - (double)m_Buffer[inputindex+posYOffset];
			dz = (double)m_Buffer[inputindex+negZOffset] - (double)m_Buffer[inputindex+posZOffset]; 		
			length = sqrt(dx*dx+dy*dy+dz*dz);

			grad[0] = (unsigned char)(dx/length * 127.0)+127;
			grad[1] = (unsigned char)(dy/length * 127.0)+127;
			grad[2] = (unsigned char)(dz/length * 127.0)+127;
			grad[3] = 255;
			
			// write out the result
			targetFile.writeBlock((char*)grad, 4);
			
			// do normal case
			negXOffset = -1; posXOffset = 1;
			for (x=1, inputindex = sliceSize + y*width + 1;
				x<width-1; 
				x++, inputindex++) {
		
				dx = (double)m_Buffer[inputindex+negXOffset] - (double)m_Buffer[inputindex+posXOffset];
				dy = (double)m_Buffer[inputindex+negYOffset] - (double)m_Buffer[inputindex+posYOffset];
				dz = (double)m_Buffer[inputindex+negZOffset] - (double)m_Buffer[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);

				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
			
				// write out the result
				targetFile.writeBlock((char*)grad, 4);
			}
			
			if (width > 1) {
				// do border case
				negXOffset = -1; posXOffset = 0;
				// use the inputindex coming off the for loop
	
				dx = (double)m_Buffer[inputindex+negXOffset] - (double)m_Buffer[inputindex+posXOffset];
				dy = (double)m_Buffer[inputindex+negYOffset] - (double)m_Buffer[inputindex+posYOffset];
				dz = (double)m_Buffer[inputindex+negZOffset] - (double)m_Buffer[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);

				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
			
				// write out the result
				targetFile.writeBlock((char*)grad, 4);
			}
		}
		incrementProgress(sliceSize);
	}
}

#ifdef LARGEFILE_KLUDGE
void VolumeFileSource::createGradientFromShortRawiv(PFile& targetFile, unsigned int variable, unsigned int timeStep)
#else
void VolumeFileSource::createGradientFromShortRawiv(QFile& targetFile, unsigned int variable, unsigned int timeStep)
#endif
{
	// make sure there is enough space for 3 slices of short data
	allocateBuffer( getDimX()*getDimY()*3*sizeof(short) /* 3 slices */);

	int sliceNum, x=0, y=0, inputindex=0;
	int negXOffset, posXOffset;
	int negYOffset, posYOffset;
	int negZOffset, posZOffset;
	unsigned int sliceSize = getDimX()*getDimY(), width=getDimX();
	unsigned char grad[4];
	double length, dx,dy,dz;

	// set the file position
	m_VolumeFile->setPosition(variable, timeStep, 0);

	for (sliceNum=0; sliceNum<getDimZ(); sliceNum++) {

		// prepare the slice buffer
		if (sliceNum == 0) {
			// this is the first slice
			// zero out the first slice in the buffer
			memset(m_Buffer, 0, sliceSize*sizeof(short));
			// and read the first two slices of the volume
			m_VolumeFile->readShortData((unsigned short *)(m_Buffer+sliceSize*sizeof(short)), 2*sliceSize, variable);
		}	
		else if (sliceNum+1 >= getDimZ() ) {
			// this is the last slice
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize*sizeof(short), sliceSize*sizeof(short));
			memcpy(m_Buffer+sliceSize*sizeof(short), m_Buffer+sliceSize*2*sizeof(short), sliceSize*sizeof(short));
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// then zero the last slice in the buffer
			memset(m_Buffer+sliceSize*2*sizeof(short), 0, sliceSize*sizeof(short));
		}
		else {
			// some slice in the middle of the volume
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize*sizeof(short), sliceSize*sizeof(short));
			memcpy(m_Buffer+sliceSize*sizeof(short), m_Buffer+sliceSize*2*sizeof(short), sliceSize*sizeof(short));
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// read in the next slice
			m_VolumeFile->readShortData((unsigned short *)(m_Buffer+sliceSize*2*sizeof(short)), sliceSize, variable);
		}
		
		// compute the gradient for this slice
		//
		// get the z offset
		if (sliceNum == 0) { // border offset
			negZOffset = 0; posZOffset = sliceSize;
		}
		else if (sliceNum == getDimZ()-1) { // border offset
			negZOffset = -sliceSize; posZOffset = 0;
		}
		else { // normal offset
			negZOffset = -sliceSize; posZOffset = sliceSize;
		}
		//
		for (y=0; y < getDimY(); y++) {
			// get the y offset
			if (y==0) { // border offset
				negYOffset = 0; posYOffset = width;
			}
			else if (y==getDimY()-1) { // border offset
				negYOffset = -width; posYOffset = 0;
			}				
			else { // normal offset
				negYOffset = -width; posYOffset = width;
			}

			// do border case
			negXOffset = 0; posXOffset = 1;
			inputindex = sliceSize + y*width + x;
	
			dx = (double)((unsigned short *)m_Buffer)[inputindex+negXOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posXOffset];
			dy = (double)((unsigned short *)m_Buffer)[inputindex+negYOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posYOffset];
			dz = (double)((unsigned short *)m_Buffer)[inputindex+negZOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posZOffset]; 		
			length = sqrt(dx*dx+dy*dy+dz*dz);

			grad[0] = (unsigned char)(dx/length * 127.0)+127;
			grad[1] = (unsigned char)(dy/length * 127.0)+127;
			grad[2] = (unsigned char)(dz/length * 127.0)+127;
			grad[3] = 255;
			
			// write out the result
			targetFile.writeBlock((char*)grad, 4);
			
			// do normal case
			negXOffset = -1; posXOffset = 1;
			for (x=1, inputindex = sliceSize + y*width + 1;
				x<width-1; 
				x++, inputindex++) {
		
				dx = (double)((unsigned short *)m_Buffer)[inputindex+negXOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posXOffset];
				dy = (double)((unsigned short *)m_Buffer)[inputindex+negYOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posYOffset];
				dz = (double)((unsigned short *)m_Buffer)[inputindex+negZOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);

				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
			
				// write out the result
				targetFile.writeBlock((char*)grad, 4);
			}
			
			if (width > 1) {
				// do border case
				negXOffset = -1; posXOffset = 0;
				// use the inputindex coming off the for loop
		
				dx = (double)((unsigned short *)m_Buffer)[inputindex+negXOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posXOffset];
				dy = (double)((unsigned short *)m_Buffer)[inputindex+negYOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posYOffset];
				dz = (double)((unsigned short *)m_Buffer)[inputindex+negZOffset] - (double)((unsigned short *)m_Buffer)[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);
	
				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
				
				// write out the result
				targetFile.writeBlock((char*)grad, 4);
			}
		}
		incrementProgress(sliceSize);
	}
}

#ifdef LARGEFILE_KLUDGE
void VolumeFileSource::createGradientFromFloatRawiv(PFile& targetFile, unsigned int variable, unsigned int timeStep)
#else
void VolumeFileSource::createGradientFromFloatRawiv(QFile& targetFile, unsigned int variable, unsigned int timeStep)
#endif
{
	// make sure there is enough space for 3 slices of float data
	allocateBuffer( getDimX()*getDimY()*3*sizeof(float) /* 3 slices */);

	int sliceNum, x=0, y=0, inputindex=0;
	int negXOffset, posXOffset;
	int negYOffset, posYOffset;
	int negZOffset, posZOffset;
	unsigned int sliceSize = getDimX()*getDimY(), width=getDimX();
	unsigned char grad[4];
	double length, dx,dy,dz;

	// set the file position
	m_VolumeFile->setPosition(variable, timeStep, 0);

	for (sliceNum=0; sliceNum<getDimZ(); sliceNum++) {

		// prepare the slice buffer
		if (sliceNum == 0) {
			// this is the first slice
			// zero out the first slice in the buffer
			memset(m_Buffer, 0, sliceSize*sizeof(float));
			// and read the first two slices of the volume
			m_VolumeFile->readFloatData((float *)(m_Buffer+sliceSize*sizeof(float)), 2*sliceSize, variable);
		}	
		else if (sliceNum+1 >= getDimZ() ) {
			// this is the last slice
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize*sizeof(float), sliceSize*sizeof(float));
			memcpy(m_Buffer+sliceSize*sizeof(float), m_Buffer+sliceSize*2*sizeof(float), sliceSize*sizeof(float));
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// then zero the last slice in the buffer
			memset(m_Buffer+sliceSize*2*sizeof(float), 0, sliceSize*sizeof(float));
		}
		else {
			// some slice in the middle of the volume
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize*sizeof(float), sliceSize*sizeof(float));
			memcpy(m_Buffer+sliceSize*sizeof(float), m_Buffer+sliceSize*2*sizeof(float), sliceSize*sizeof(float));
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// read in the next slice
			m_VolumeFile->readFloatData((float *)(m_Buffer+sliceSize*2*sizeof(float)), sliceSize, variable);
		}
		
		// compute the gradient for this slice
		//
		// get the z offset
		if (sliceNum == 0) { // border offset
			negZOffset = 0; posZOffset = sliceSize;
		}
		else if (sliceNum == getDimZ()-1) { // border offset
			negZOffset = -sliceSize; posZOffset = 0;
		}
		else { // normal offset
			negZOffset = -sliceSize; posZOffset = sliceSize;
		}
		//
		for (y=0; y < getDimY(); y++) {
			// get the y offset
			if (y==0) { // border offset
				negYOffset = 0; posYOffset = width;
			}
			else if (y==getDimY()-1) { // border offset
				negYOffset = -width; posYOffset = 0;
			}				
			else { // normal offset
				negYOffset = -width; posYOffset = width;
			}

			// do border case
			negXOffset = 0; posXOffset = 1;
			inputindex = sliceSize + y*width + x;
	
			dx = (double)((float *)m_Buffer)[inputindex+negXOffset] - (double)((float *)m_Buffer)[inputindex+posXOffset];
			dy = (double)((float *)m_Buffer)[inputindex+negYOffset] - (double)((float *)m_Buffer)[inputindex+posYOffset];
			dz = (double)((float *)m_Buffer)[inputindex+negZOffset] - (double)((float *)m_Buffer)[inputindex+posZOffset]; 		
			length = sqrt(dx*dx+dy*dy+dz*dz);

			grad[0] = (unsigned char)(dx/length * 127.0)+127;
			grad[1] = (unsigned char)(dy/length * 127.0)+127;
			grad[2] = (unsigned char)(dz/length * 127.0)+127;
			grad[3] = 255;
			
			// write out the result
			targetFile.writeBlock((char*)grad, 4);
			
			// do normal case
			negXOffset = -1; posXOffset = 1;
			for (x=1, inputindex = sliceSize + y*width + 1;
				x<width-1; 
				x++, inputindex++) {
		
				dx = (double)((float *)m_Buffer)[inputindex+negXOffset] - (double)((float *)m_Buffer)[inputindex+posXOffset];
				dy = (double)((float *)m_Buffer)[inputindex+negYOffset] - (double)((float *)m_Buffer)[inputindex+posYOffset];
				dz = (double)((float *)m_Buffer)[inputindex+negZOffset] - (double)((float *)m_Buffer)[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);

				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
			
				// write out the result
				targetFile.writeBlock((char*)grad, 4);
			}
			
			if (width > 1) {
				// do border case
				negXOffset = -1; posXOffset = 0;
				// use the inputindex coming off the for loop
		
				dx = (double)((float *)m_Buffer)[inputindex+negXOffset] - (double)((float *)m_Buffer)[inputindex+posXOffset];
				dy = (double)((float *)m_Buffer)[inputindex+negYOffset] - (double)((float *)m_Buffer)[inputindex+posYOffset];
				dz = (double)((float *)m_Buffer)[inputindex+negZOffset] - (double)((float *)m_Buffer)[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);

				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
			
				// write out the result
				targetFile.writeBlock((char*)grad, 4);
			}
		}
		incrementProgress(sliceSize);
	}
}

#ifdef LARGEFILE_KLUDGE
void VolumeFileSource::createGradientFromCacheFile(PFile& source, PFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ)
#else
void VolumeFileSource::createGradientFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ)
#endif
{
	// make sure there is enough space for 3 slices of char data
	allocateBuffer(dimX*dimY*3 /* 3 slices */);

	int sliceNum, x=0, y=0, inputindex=0;
	int negXOffset, posXOffset;
	int negYOffset, posYOffset;
	int negZOffset, posZOffset;
	unsigned int sliceSize = dimX*dimY, width=dimX;
	unsigned char grad[4];
	double length, dx,dy,dz;

	// set the file position
	source.at(0);

	for (sliceNum=0; sliceNum<dimZ; sliceNum++) {

		// prepare the slice buffer
		if (sliceNum == 0) {
			// this is the first slice
			// zero out the first slice in the buffer
			memset(m_Buffer, 0, sliceSize);
			// and read the first two slices of the volume
			source.readBlock((char*)m_Buffer+sliceSize, 2*sliceSize);
		}	
		else if (sliceNum+1 >= dimZ ) {
			// this is the last slice
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize, sliceSize);
			memcpy(m_Buffer+sliceSize, m_Buffer+sliceSize*2, sliceSize);
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// then zero the last slice in the buffer
			memset(m_Buffer+sliceSize*2, 0, sliceSize);
		}
		else {
			// some slice in the middle of the volume
			// move the last two slices to the beginning of the buffer
			memcpy(m_Buffer, m_Buffer+sliceSize, sliceSize);
			memcpy(m_Buffer+sliceSize, m_Buffer+sliceSize*2, sliceSize);
			//memmove(m_Buffer, m_Buffer+sliceSize, 2*sliceSize);
			// read in the next slice
			source.readBlock((char*)m_Buffer+sliceSize*2, sliceSize);
		}
		
		// compute the gradient for this slice
		//
		// get the z offset
		if (sliceNum == 0) { // border offset
			negZOffset = 0; posZOffset = sliceSize;
		}
		else if (sliceNum == dimZ-1) { // border offset
			negZOffset = -sliceSize; posZOffset = 0;
		}
		else { // normal offset
			negZOffset = -sliceSize; posZOffset = sliceSize;
		}
		//
		for (y=0; y < dimY; y++) {
			// get the y offset
			if (y == 0) { // border offset
				negYOffset = 0; posYOffset = width;
			}
			else if (y == dimY-1) { // border offset
				negYOffset = -width; posYOffset = 0;
			}				
			else { // normal offset
				negYOffset = -width; posYOffset = width;
			}

			// do border case
			negXOffset = 0; posXOffset = 1;
			inputindex = sliceSize + y*width + x;
	
			dx = (double)m_Buffer[inputindex+negXOffset] - (double)m_Buffer[inputindex+posXOffset];
			dy = (double)m_Buffer[inputindex+negYOffset] - (double)m_Buffer[inputindex+posYOffset];
			dz = (double)m_Buffer[inputindex+negZOffset] - (double)m_Buffer[inputindex+posZOffset]; 		
			length = sqrt(dx*dx+dy*dy+dz*dz);

			grad[0] = (unsigned char)(dx/length * 127.0)+127;
			grad[1] = (unsigned char)(dy/length * 127.0)+127;
			grad[2] = (unsigned char)(dz/length * 127.0)+127;
			grad[3] = 255;
			
			// write out the result
			target.writeBlock((char*)grad, 4);
			
			// do normal case
			negXOffset = -1; posXOffset = 1;
			for (x=1, inputindex = sliceSize + y*width + 1;
				x<width-1; 
				x++, inputindex++) {
		
				dx = (double)m_Buffer[inputindex+negXOffset] - (double)m_Buffer[inputindex+posXOffset];
				dy = (double)m_Buffer[inputindex+negYOffset] - (double)m_Buffer[inputindex+posYOffset];
				dz = (double)m_Buffer[inputindex+negZOffset] - (double)m_Buffer[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);

				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
			
				// write out the result
				target.writeBlock((char*)grad, 4);
			}
			
			if (width > 1) {
				// do border case
				negXOffset = -1; posXOffset = 0;
				// use the inputindex coming off the for loop
	
				dx = (double)m_Buffer[inputindex+negXOffset] - (double)m_Buffer[inputindex+posXOffset];
				dy = (double)m_Buffer[inputindex+negYOffset] - (double)m_Buffer[inputindex+posYOffset];
				dz = (double)m_Buffer[inputindex+negZOffset] - (double)m_Buffer[inputindex+posZOffset]; 		
				length = sqrt(dx*dx+dy*dy+dz*dz);

				grad[0] = (unsigned char)(dx/length * 127.0)+127;
				grad[1] = (unsigned char)(dy/length * 127.0)+127;
				grad[2] = (unsigned char)(dz/length * 127.0)+127;
				grad[3] = 255;
			
				// write out the result
				target.writeBlock((char*)grad, 4);
			}
		}
		incrementProgress(sliceSize);
	}
}

void VolumeFileSource::createContourSpectrumFromRawIV(QFile& target, unsigned int variable, unsigned int timeStep)
{
	// cache the size of m_Buffer
	//unsigned int bufSize = m_BufferSize;
	// make sure there is enough space in m_Buffer
	//allocateBuffer(getDimX()*getDimY()*getDimZ());

	// set up variables
	//unsigned char* dataBuffer = m_Buffer;
	unsigned char* dataBuffer = new unsigned char [getDimX()*getDimY()*getDimZ()];
	int i, array_size=256, dim[3] = {getDimX(),getDimY(),getDimZ()};
	float isoval[256],area[256],min_vol[256],max_vol[256],gradient[256];
	float span[3], orig[3];
 	ConDataset* the_data;
	Signature	*sig;

	// set the file position
	m_VolumeFile->setPosition(variable, timeStep, 0);
	// read data
	m_VolumeFile->readCharData(dataBuffer, getDimX()*getDimY()*getDimZ(), variable);

	// compute origin and span
	orig[0] = getMinX();
	orig[1] = getMinY();
	orig[2] = getMinZ();
	span[0] = (getMaxX()-getMinX()) / (float)(getDimX()-1);
	span[1] = (getMaxY()-getMinY()) / (float)(getDimY()-1);
	span[2] = (getMaxZ()-getMinZ()) / (float)(getDimZ()-1);
	
	// make a libcontour variable out of dataBuffer
 	the_data = newDatasetReg(CONTOUR_UCHAR, CONTOUR_REG_3D, 1, 1, dim,dataBuffer);
	((Datareg3 *)the_data->data->getData(0))->setOrig(orig);
	((Datareg3 *)the_data->data->getData(0))->setSpan(span);
	// compute the contour spectrum
	sig = getSignatureFunctions(the_data, 0,0);
	// the_data has(n't?) a copy of dataBuffer
	delete [] dataBuffer;

	// break sig into its constituent functions
	for (i=0;i<array_size;i++) {
		isoval[i]=sig[0].fx[i];
		area[i]=sig[0].fy[i];
		min_vol[i]=sig[1].fy[i];
		max_vol[i]=sig[2].fy[i];
		gradient[i]=sig[3].fy[i];
	}

	// write the spectrum functions to disk
	target.writeBlock((char *)isoval, 256*sizeof(float));
	target.writeBlock((char *)area, 256*sizeof(float));
	target.writeBlock((char *)min_vol, 256*sizeof(float));
	target.writeBlock((char *)max_vol, 256*sizeof(float));
	target.writeBlock((char *)gradient, 256*sizeof(float));

	// reset m_Buffer to how we found it (if its size changed)
	//if ( bufSize < (getDimX()*getDimY()*getDimZ()) )
	//{
	//	destroyBuffer();
	//	forceAllocateBuffer(bufSize);
	//}

	// clean up
	delete the_data;
	delete sig;

	// increment the progress bar
	//incrementProgress(getDimX()*getDimY()*getDimZ());
}

void VolumeFileSource::createContourSpectrumFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ)
{
	// cache the size of m_Buffer
	//unsigned int bufSize = m_BufferSize;
	// make sure there is enough space in m_Buffer
	//allocateBuffer(dimX*dimY*dimZ);

	// set up variables
	//unsigned char* dataBuffer = m_Buffer;
	unsigned char* dataBuffer = new unsigned char [dimX*dimY*dimZ];
	int i, array_size=256, dim[3] = {dimX,dimY,dimZ};
	float isoval[256],area[256],min_vol[256],max_vol[256],gradient[256];
	float span[3], orig[3];
 	ConDataset* the_data;
	Signature	*sig;

	// set the file position
	source.at(0);
	// read data
	source.readBlock((char *)dataBuffer, dimX*dimY*dimZ);
	
	// compute origin and span
	orig[0] = getMinX();
	orig[1] = getMinY();
	orig[2] = getMinZ();
	span[0] = (getMaxX()-getMinX()) / (float)(dimX-1);
	span[1] = (getMaxY()-getMinY()) / (float)(dimY-1);
	span[2] = (getMaxZ()-getMinZ()) / (float)(dimZ-1);
	
	// make a libcontour variable out of dataBuffer
 	the_data = newDatasetReg(CONTOUR_UCHAR, CONTOUR_REG_3D, 1, 1, dim,dataBuffer);
	((Datareg3 *)the_data->data->getData(0))->setOrig(orig);
	((Datareg3 *)the_data->data->getData(0))->setSpan(span);
	// compute the contour spectrum
	sig = getSignatureFunctions(the_data, 0,0);
	// the_data has(n't?) a copy of dataBuffer
	delete [] dataBuffer;

	// break sig into its constituent functions
	for (i=0;i<array_size;i++) {
		isoval[i]=sig[0].fx[i];
		area[i]=sig[0].fy[i];
		min_vol[i]=sig[1].fy[i];
		max_vol[i]=sig[2].fy[i];
		gradient[i]=sig[3].fy[i];
	}

	// write the spectrum functions to disk
	target.writeBlock((char *)isoval, 256*sizeof(float));
	target.writeBlock((char *)area, 256*sizeof(float));
	target.writeBlock((char *)min_vol, 256*sizeof(float));
	target.writeBlock((char *)max_vol, 256*sizeof(float));
	target.writeBlock((char *)gradient, 256*sizeof(float));

	// reset m_Buffer to how we found it (if its size changed)
	//if ( bufSize < (dimX*dimY*dimZ) )
	//{
	//	destroyBuffer();
	//	forceAllocateBuffer(bufSize);
	//}

	// clean up
	delete the_data;
	delete sig;

	//incrementProgress(dimX*dimY*dimZ);
}

void VolumeFileSource::createContourTreeFromRawIV(QFile& target,
								unsigned int variable, unsigned int timeStep)
{
	// cache the size of m_Buffer
	//unsigned int bufSize = m_BufferSize;
	// make sure there is enough space in m_Buffer
	//allocateBuffer(getDimX()*getDimY()*getDimZ());

	// set up variables
	//unsigned char* dataBuffer = m_Buffer;
	unsigned char* dataBuffer = new unsigned char [getDimX()*getDimY()*getDimZ()];
	int numVerts,numEdges, dim[3] = {getDimX(),getDimY(),getDimZ()};
	CTVTX *verts;
	CTEDGE *edges;

	// set the file position
	m_VolumeFile->setPosition(variable, timeStep, 0);
	// read data
	m_VolumeFile->readCharData(dataBuffer, getDimX()*getDimY()*getDimZ(), variable);
	
	// compute the contour tree
	computeCT(dataBuffer, dim, numVerts, numEdges, &verts, &edges);
	
	qDebug("CT: numverts = %d, numedges = %d", numVerts, numEdges);
	// write the contour tree to disk
	target.writeBlock((char *)&numVerts, sizeof(int));
	target.writeBlock((char *)&numEdges, sizeof(int));
	target.writeBlock((char *)verts, numVerts*sizeof(CTVTX));
	target.writeBlock((char *)edges, numEdges*sizeof(CTEDGE));

	// clean up
	delete [] dataBuffer;
	free(verts);
	free(edges);

	// reset m_Buffer to how we found it (if its size changed)
	//if ( bufSize < (getDimX()*getDimY()*getDimZ()) )
	//{
	//	destroyBuffer();
	//	forceAllocateBuffer(bufSize);
	//}

	// increment the progress bar
	//incrementProgress(getDimX()*getDimY()*getDimZ());
}

void VolumeFileSource::createContourTreeFromCacheFile(QFile& source,
								QFile& target, unsigned int dimX, unsigned int dimY,
								unsigned int dimZ)
{
	// cache the size of m_Buffer
	//unsigned int bufSize = m_BufferSize;
	// make sure there is enough space in m_Buffer
	//allocateBuffer(dimX*dimY*dimZ);

	// set up variables
	//unsigned char* dataBuffer = m_Buffer;
	unsigned char* dataBuffer = new unsigned char [dimX*dimY*dimZ];
	int numVerts,numEdges, dim[3] = {dimX,dimY,dimZ};
	CTVTX *verts;
	CTEDGE *edges;

	// set the file position
	source.at(0);
	// read data
	source.readBlock((char *)dataBuffer, dimX*dimY*dimZ);
	
	// compute the contour tree
	computeCT(dataBuffer, dim, numVerts, numEdges, &verts, &edges);
	
	qDebug("CT: numverts = %d, numedges = %d", numVerts, numEdges);
	// write the contour tree to disk
	target.writeBlock((char *)&numVerts, sizeof(int));
	target.writeBlock((char *)&numEdges, sizeof(int));
	target.writeBlock((char *)verts, numVerts*sizeof(CTVTX));
	target.writeBlock((char *)edges, numEdges*sizeof(CTEDGE));

	// clean up
	delete [] dataBuffer;
	free(verts);
	free(edges);

	// reset m_Buffer to how we found it (if its size changed)
	//if ( bufSize < (dimX*dimY*dimZ) )
	//{
	//	destroyBuffer();
	//	forceAllocateBuffer(bufSize);
	//}

	//incrementProgress(dimX*dimY*dimZ);
}

bool VolumeFileSource::cacheOutdated(QFileInfo fileInfo, QDir cacheDir)
{
	// check for the existance of the record file
	QString fileName(cacheDir.absPath() + "/" + fileInfo.baseName() + "_Record.cache");
	QFileInfo recordInfo(fileName);
	if (!recordInfo.exists()) {
		// no record file means cache is out of date
		return true;
	}
	else {
		// open the record file
		QFile recordFile(fileName);
		recordFile.open(QIODevice::ReadOnly);
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

		// read the min/max function values
		// no check since these would change along with the modification date 
		unsigned int v, t;
		for (v=0; v<m_VolumeFile->getNumVariables(); v++)
			for (t=0; t<m_VolumeFile->getNumTimeSteps(); t++) {
				stream >> m_FuncMin[v*m_VolumeFile->getNumTimeSteps()+t];
				stream >> m_FuncMax[v*m_VolumeFile->getNumTimeSteps()+t];

				qDebug("variable %d / timestep %d min/max: [%f, %f]",v,t,
							m_FuncMin[v*m_VolumeFile->getNumTimeSteps()+t],
							m_FuncMax[v*m_VolumeFile->getNumTimeSteps()+t]);
			}

		recordFile.close();

		for (v=0; v<m_VolumeFile->getNumVariables(); v++) {
			for (t=0; t<m_VolumeFile->getNumTimeSteps(); t++) {
				if (!checkAllMipmapLevels(v,t)) {
					// problem with the cache, return true to rebuild
					return true;
				}
			}
		}

		// everything checks out, cache is not outdated
		return false;
	}
}

bool VolumeFileSource::checkMipmapLevel(uint level, unsigned int variable, unsigned int timeStep, uint dimX, uint dimY, uint dimZ)
{
	QFileInfo mmFileInfo(mipmapLevelFileName(level, variable, timeStep));
//	QFileInfo gFileInfo(gradientLevelFileName(level, variable, timeStep));
#ifdef LARGEFILE_KLUDGE
	PFile mmFile(mmFileInfo.absFilePath());
//	PFile gFile(gFileInfo.absFilePath());
#else
	QFile mmFile(mmFileInfo.absFilePath());
//	QFile gFile(gFileInfo.absFilePath());
#endif
	return (mmFile.open(QIODevice::ReadOnly | QIODevice::Unbuffered)
					&& mmFile.size()==dimX*dimY*((qulonglong)dimZ)
					);//&& gFile.open(IO_ReadOnly | IO_Raw)
					//&& gFile.size()==dimX*dimY*((Q_ULLONG)dimZ)*4);
}

bool VolumeFileSource::checkAllMipmapLevels(unsigned int variable, unsigned int timeStep)
{
	// check the mipmapLevels
	unsigned int level, dimX, dimY, dimZ;
	bool moreLevels;
	if (m_VolumeFile->getVariableType(variable) == VolumeFile::Char) {
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
		if (!checkMipmapLevel(level, variable, timeStep, dimX, dimY, dimZ)) {
			// problem with mipmap file
			return false; // cache is faulty
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

	// if we reach here, cache is good
	return true;
}

void VolumeFileSource::incrementProgress(int amount)
{
	static int delay = 0;
	m_ProgressValue+=amount;
	delay+=amount;
	//qDebug("Current: %d", m_ProgressValue);
	if (delay>500000) {
		m_ProgressDialog->setProgress(m_ProgressValue / m_OnePercentProgress);
		qApp->processEvents();
		delay = 0;
		if (m_ProgressDialog->wasCancelled()) {
			throw UserCancelException();
		}
	}
}

qulonglong VolumeFileSource::determineCacheWorkAmount()
{
	unsigned int v;
	qulonglong totalAmount = 0;

	for (v=0; v<m_VolumeFile->getNumVariables(); v++) {
		totalAmount+=determineWorkForVariable(v);
	}
	qDebug("total: %lu", totalAmount);
	return totalAmount;

}

qulonglong VolumeFileSource::determineWorkForVariable(unsigned int variable)
{
	bool moreLevels, spectrumAdded=false;
	unsigned int targetLevel;
	unsigned int newDimX;
	unsigned int newDimY;
	unsigned int newDimZ;
	qulonglong totalAmount = 0;


	if ( (getDimX()>1 || getDimY()>1 || getDimZ()>1) ) {
		if (m_VolumeFile->getVariableType(variable)==VolumeFile::Char) { // byte data, there is no level 0 mipmap
			targetLevel = 1;
			totalAmount+=2*getDimZ()*getDimX()*((qulonglong)getDimY());
			// calculate the new dimensions
			newDimX = (getDimX()+1)>>1;
			newDimY = (getDimY()+1)>>1;
			newDimZ = (getDimZ()+1)>>1;
			moreLevels = newDimX>1||newDimY>1||newDimZ>1;
			// set targetlevel to 2
			targetLevel++;

		}
		else if (m_VolumeFile->getVariableType(variable)==VolumeFile::Short) { // create a level 0 mipmap
			targetLevel = 0;
			moreLevels = getDimX()>1||getDimY()>1||getDimZ()>1;
			totalAmount+=3*getDimZ()*getDimX()*((qulonglong)getDimY());

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
			totalAmount+=3*getDimZ()*getDimX()*((qulonglong)getDimY());

			// calculate the new dimensions
			newDimX = getDimX();
			newDimY = getDimY();
			newDimZ = getDimZ();
			// set targetlevel to 1
			targetLevel++;
		}

		// check to see if the contour spectrum will be computed at this level
		//if ( (getDimX()*getDimY()*getDimZ())<=(256*256*256) ) {
		//	totalAmount += getDimX()*getDimY()*getDimZ();
		//	spectrumAdded = true;
		//}

		while (moreLevels) {

			// create the level
			moreLevels = newDimX>1||newDimY>1||newDimZ>1;

			totalAmount+=2*newDimZ*newDimX*((qulonglong)newDimY);
			
			// check to see if the contour spectrum will be computed at this level
			//if ( !spectrumAdded && (newDimX*newDimY*newDimZ)<=(256*256*256) ) {
			//	totalAmount += newDimX*newDimY*newDimZ;
			//	spectrumAdded = true;
			//}

			// calculate the new dimensions
			newDimX = (newDimX+1)>>1;
			newDimY = (newDimY+1)>>1;
			newDimZ = (newDimZ+1)>>1;
			// increment the target level
			targetLevel++;
			
		}

	}
	return totalAmount*m_VolumeFile->getNumTimeSteps();
}

