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

// DownLoadManager.cpp: implementation of the DownLoadManager class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/DownLoadManager.h>
#include <math.h>

#include <stdio.h>
#include <qdatetime.h>
#include <Filters/BilateralFilter.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

DownLoadManager::DownLoadManager(unsigned int bufferX, unsigned int bufferY, unsigned int bufferZ)
{
	setDefaults();
	//allocateBuffer(bufferX+2, bufferY+2, bufferZ+2);
	allocateBuffer(bufferX, bufferY, bufferZ);
}

DownLoadManager::~DownLoadManager()
{
}

static inline double clamp(double value, double min, double max)
{
	value = (value<=max ? value : max);
	return  (value>=min ? value : min);
}

bool DownLoadManager::getThumbnail(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time)
{
	/* rewriting this function to aid in isocontouring
	getData(viewer, getMinX(), getMinY(), getMinZ(), getMaxX(), getMaxY(), getMaxZ());
	*/
	double minX = sourceManager->getMinX();
	double minY = sourceManager->getMinY();
	double minZ = sourceManager->getMinZ();
	double maxX = sourceManager->getMaxX();
	double maxY = sourceManager->getMaxY();
	double maxZ = sourceManager->getMaxZ();

	double cwX, cwY, cwZ;
	unsigned int alX, alY, alZ;

	if (sourceManager->hasSource()) {
		ExtractionInformation ei = getExtractionInformation(sourceManager, minX, minY, minZ, maxX, maxY, maxZ);
	
		// request the data
		qDebug("Requesting: from %d, %d, %d    to %d, %d, %d",ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ);
		fillData(sourceManager, dest, ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ, (unsigned int)ei.level, var, time); 
		alX = ((sourceManager->getDimX()-1)>>ei.level);
		alY = ((sourceManager->getDimY()-1)>>ei.level);
		alZ = ((sourceManager->getDimZ()-1)>>ei.level);
		cwX = (sourceManager->getMaxX()-sourceManager->getMinX())/(double)(alX);
		cwY = (sourceManager->getMaxY()-sourceManager->getMinY())/(double)(alY);
		cwZ = (sourceManager->getMaxZ()-sourceManager->getMinZ())/(double)(alZ);

		dest->setMin(
			((double)(ei.startSampleX>>ei.level)*cwX) + sourceManager->getMinX(), 
			((double)(ei.startSampleY>>ei.level)*cwY) + sourceManager->getMinY(), 
			((double)(ei.startSampleZ>>ei.level)*cwZ) + sourceManager->getMinZ());
		dest->setMax(
			((double)(ei.endSampleX>>ei.level)*cwX) + sourceManager->getMinX(), 
			((double)(ei.endSampleY>>ei.level)*cwY) + sourceManager->getMinY(), 
			((double)(ei.endSampleZ>>ei.level)*cwZ) + sourceManager->getMinZ());
		//m_Source->fillData((char*)m_TempBuffer, ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ, ei.widthX, ei.widthY, ei.widthZ);

		// prepare the error flags
		setError(sourceManager->getSource());

		// check for error
		if (!m_Error) {
			// no error, continue normally
			return true;
		}
		else if (m_ErrorMustRestart) {
			// error was detected and must restart connection to source
			sourceManager->resetSource();
			return false;
		}
		else {
			// error, but dont have to restart.  Turn off rendering
			return false;
		}
	}
	else {
		// no source, no point rendering
		return false;
	}
}

bool DownLoadManager::getData(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{

	// make sure min and max are inbounds
	minX = clamp(minX, sourceManager->getMinX(), sourceManager->getMaxX());
	minY = clamp(minY, sourceManager->getMinY(), sourceManager->getMaxY());
	minZ = clamp(minZ, sourceManager->getMinZ(), sourceManager->getMaxZ());
	maxX = clamp(maxX, sourceManager->getMinX(), sourceManager->getMaxX());
	maxY = clamp(maxY, sourceManager->getMinY(), sourceManager->getMaxY());
	maxZ = clamp(maxZ, sourceManager->getMinZ(), sourceManager->getMaxZ());

	double cwX, cwY, cwZ;
	unsigned int alX, alY, alZ;

	if (sourceManager->hasSource()) {
		ExtractionInformation ei = getExtractionInformation(sourceManager, minX, minY, minZ, maxX, maxY, maxZ);

		// request the data
		qDebug("Requesting: from %d, %d, %d    to %d, %d, %d",ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ);
		fillData(sourceManager, dest, ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ, (unsigned int)ei.level, var, time); 
		alX = ((sourceManager->getDimX()-1)>>ei.level);
		alY = ((sourceManager->getDimY()-1)>>ei.level);
		alZ = ((sourceManager->getDimZ()-1)>>ei.level);
		cwX = (sourceManager->getMaxX()-sourceManager->getMinX())/(double)(alX);
		cwY = (sourceManager->getMaxY()-sourceManager->getMinY())/(double)(alY);
		cwZ = (sourceManager->getMaxZ()-sourceManager->getMinZ())/(double)(alZ);

		dest->setMin(
			((double)(ei.startSampleX>>ei.level)*cwX) + sourceManager->getMinX(), 
			((double)(ei.startSampleY>>ei.level)*cwY) + sourceManager->getMinY(), 
			((double)(ei.startSampleZ>>ei.level)*cwZ) + sourceManager->getMinZ());
		dest->setMax(
			((double)(ei.endSampleX>>ei.level)*cwX) + sourceManager->getMinX(), 
			((double)(ei.endSampleY>>ei.level)*cwY) + sourceManager->getMinY(), 
			((double)(ei.endSampleZ>>ei.level)*cwZ) + sourceManager->getMinZ());
		//m_Source->fillData((char*)m_TempBuffer, startSampleX, startSampleY, startSampleZ, endSampleX, endSampleY, endSampleZ, widthX, widthY, widthZ);

		// prepare the error flags
		setError(sourceManager->getSource());

		// check for error
		if (!m_Error) {
			// no error, continue normally
			return true;
		}
		else if (m_ErrorMustRestart) {
			// error was detected and must restart connection to source
			sourceManager->resetSource();
			return false;
		}
		else {
			// error, but dont have to restart.  Turn off rendering
			return false;
		}
	}
	else {
		// no source, no point rendering
		return false;
	}
}

bool DownLoadManager::getGradientData(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{
	// make sure min and max are inbounds
	minX = clamp(minX, sourceManager->getMinX(), sourceManager->getMaxX());
	minY = clamp(minY, sourceManager->getMinY(), sourceManager->getMaxY());
	minZ = clamp(minZ, sourceManager->getMinZ(), sourceManager->getMaxZ());
	maxX = clamp(maxX, sourceManager->getMinX(), sourceManager->getMaxX());
	maxY = clamp(maxY, sourceManager->getMinY(), sourceManager->getMaxY());
	maxZ = clamp(maxZ, sourceManager->getMinZ(), sourceManager->getMaxZ());

	double cwX, cwY, cwZ;
	unsigned int alX, alY, alZ;

	if (sourceManager->hasSource()) {
		ExtractionInformation ei = getExtractionInformation(sourceManager, minX, minY, minZ, maxX, maxY, maxZ);

		// request the data
		qDebug("Requesting: from %d, %d, %d    to %d, %d, %d",ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ);
		fillGradientData(sourceManager, dest, ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ, (unsigned int)ei.level, var, time); 
		alX = ((sourceManager->getDimX()-1)>>ei.level);
		alY = ((sourceManager->getDimY()-1)>>ei.level);
		alZ = ((sourceManager->getDimZ()-1)>>ei.level);
		cwX = (sourceManager->getMaxX()-sourceManager->getMinX())/(double)(alX);
		cwY = (sourceManager->getMaxY()-sourceManager->getMinY())/(double)(alY);
		cwZ = (sourceManager->getMaxZ()-sourceManager->getMinZ())/(double)(alZ);

		dest->setMin(
			((double)(ei.startSampleX>>ei.level)*cwX) + sourceManager->getMinX(), 
			((double)(ei.startSampleY>>ei.level)*cwY) + sourceManager->getMinY(), 
			((double)(ei.startSampleZ>>ei.level)*cwZ) + sourceManager->getMinZ());
		dest->setMax(
			((double)(ei.endSampleX>>ei.level)*cwX) + sourceManager->getMinX(), 
			((double)(ei.endSampleY>>ei.level)*cwY) + sourceManager->getMinY(), 
			((double)(ei.endSampleZ>>ei.level)*cwZ) + sourceManager->getMinZ());
		//m_Source->fillData((char*)m_TempBuffer, startSampleX, startSampleY, startSampleZ, endSampleX, endSampleY, endSampleZ, widthX, widthY, widthZ);

		// prepare the error flags
		setError(sourceManager->getSource());

		// check for error
		if (!m_Error) {
			// no error, continue normally
			return true;
		}
		else if (m_ErrorMustRestart) {
			// error was detected and must restart connection to source
			sourceManager->resetSource();
			return false;
		}
		else {
			// error, but dont have to restart.  Turn off rendering
			return false;
		}
	}
	else {
		// no source, no point rendering
		return false;
	}
}

void DownLoadManager::applyFilter(SourceManager* sourceManager, QWidget* parent, Filter* filter)
{
	/* In transition, I'll fix this later


	VolumeSource* source = sourceManager->getSource();
	VolumeBuffer* buffer = sourceManager->getZoomedInVolume()->getBuffer();
	if (source && buffer->getWidth()*buffer->getHeight()*buffer->getDepth()>0) {
		if (filter->applyFilter(parent, buffer->getBuffer(), 
			buffer->getWidth(),
			buffer->getHeight(),
			buffer->getDepth())) {

			ExtractionInformation ei = getExtractionInformation(sourceManager, 
				buffer->getMinX(),
				buffer->getMinY(),
				buffer->getMinZ(),
				buffer->getMaxX(), 
				buffer->getMaxY(), 
				buffer->getMaxZ());

			upload(sourceManager, sourceManager->getZoomedInVolume(), buffer, ei);


		}
		else { // failed to apply filter, do nothing
		}
	}
	*/
}

bool DownLoadManager::getExtentCoords(SourceManager* sourceManager,
												double minX, double minY, double minZ,
												double maxX, double maxY, double maxZ,
												int& startX, int& startY, int& startZ,
												int& endX, int& endY, int& endZ)
{
	if (sourceManager->hasSource()) {
		ExtractionInformation ei = getExtractionInformation(sourceManager, minX, minY, minZ, maxX, maxY, maxZ);
		
		startX = ei.startSampleX;
		startY = ei.startSampleY;
		startZ = ei.startSampleZ;
		endX = ei.endSampleX;
		endY = ei.endSampleY;
		endZ = ei.endSampleZ;

		return true;
	}
	else return false;
}

bool DownLoadManager::error() const
{
	return m_Error;
}

bool DownLoadManager::errorMustRestart() const
{
	return m_ErrorMustRestart;
}

const QString& DownLoadManager::errorReason() const
{
	return m_ErrorReason;
}

void DownLoadManager::resetError()
{
	m_Error = false;
	m_ErrorMustRestart = false;
	m_ErrorReason = QString("There is no error");
}

void DownLoadManager::setError(const QString& reason, bool mustRestart)
{
	m_Error = true;
	m_ErrorMustRestart = mustRestart;
	m_ErrorReason = reason;
}

void DownLoadManager::setError(VolumeSource* source)
{
	if (source) {
		m_Error = source->error();
		m_ErrorMustRestart = source->errorMustRestart();
		m_ErrorReason = source->errorReason();
		source->resetError();
	}
}

ExtractionInformation DownLoadManager::getExtractionInformation(SourceManager* sourceManager, double minX, double minY, double minZ, double maxX, double maxY, double maxZ) const
{
	ExtractionInformation ei;
	// make sure min and max are inbounds
	minX = clamp(minX, sourceManager->getMinX(), sourceManager->getMaxX());
	minY = clamp(minY, sourceManager->getMinY(), sourceManager->getMaxY());
	minZ = clamp(minZ, sourceManager->getMinZ(), sourceManager->getMaxZ());
	maxX = clamp(maxX, sourceManager->getMinX(), sourceManager->getMaxX());
	maxY = clamp(maxY, sourceManager->getMinY(), sourceManager->getMaxY());
	maxZ = clamp(maxZ, sourceManager->getMinZ(), sourceManager->getMaxZ());


	ei.level = -1;
	do { // loop to find the level that fits within our space
		ei.level++;
		ei.startSampleX = getStartSample(minX, sourceManager->getMinX(), sourceManager->getMaxX(), sourceManager->getDimX(), (unsigned int)ei.level);
		ei.startSampleY = getStartSample(minY, sourceManager->getMinY(), sourceManager->getMaxY(), sourceManager->getDimY(), (unsigned int)ei.level);
		ei.startSampleZ = getStartSample(minZ, sourceManager->getMinZ(), sourceManager->getMaxZ(), sourceManager->getDimZ(), (unsigned int)ei.level);
		ei.endSampleX = getEndSample(maxX, sourceManager->getMinX(), sourceManager->getMaxX(), sourceManager->getDimX(), (unsigned int)ei.level);
		ei.endSampleY = getEndSample(maxY, sourceManager->getMinY(), sourceManager->getMaxY(), sourceManager->getDimY(), (unsigned int)ei.level);
		ei.endSampleZ = getEndSample(maxZ, sourceManager->getMinZ(), sourceManager->getMaxZ(), sourceManager->getDimZ(), (unsigned int)ei.level);
		ei.widthX = ((ei.endSampleX - ei.startSampleX)/(double)(1<<ei.level) + 1);
		ei.widthY = ((ei.endSampleY - ei.startSampleY)/(double)(1<<ei.level) + 1);
		ei.widthZ = ((ei.endSampleZ - ei.startSampleZ)/(double)(1<<ei.level) + 1);
		/*
		// using the textures border
		canvasWidthX = upToPowerOfTwo(widthX-2)+2;
		canvasWidthY = upToPowerOfTwo(widthY-2)+2;
		canvasWidthZ = upToPowerOfTwo(widthZ-2)+2;
	} while (!(canvasWidthX*canvasWidthY*canvasWidthZ <= m_BufferX*m_BufferY*m_BufferZ && 
		viewer->testColorMappedDataWithBorder(canvasWidthX,canvasWidthY,canvasWidthZ)));
		*/
		// not using the texture's border
		ei.canvasWidthX = upToPowerOfTwo(ei.widthX);
		ei.canvasWidthY = upToPowerOfTwo(ei.widthY);
		ei.canvasWidthZ = upToPowerOfTwo(ei.widthZ);
	} while (!testSize(ei.widthX, ei.widthY, ei.widthZ));

	return ei;
}

/*
void DownLoadManager::upload(SourceManager* sourceManager, VolumeWithContours* volumeWithContours, VolumeBuffer* buffer, ExtractionInformation ei)
{
	double minX = buffer->getMinX();
	double minY = buffer->getMinY();
	double minZ = buffer->getMinZ();
	double maxX = buffer->getMaxX();
	double maxY = buffer->getMaxY();
	double maxZ = buffer->getMaxZ();

	copyToUploadableBuffer(sourceManager, buffer, ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ, (unsigned int)ei.level, ei.canvasWidthX, ei.canvasWidthY, ei.canvasWidthZ);
	
	// with border
	//viewer->uploadColorMappedDataWithBorder(m_MainBuffer, canvasWidthX, canvasWidthY, canvasWidthZ);
	//
	// without border
	QTime t;
	t.start();
	
	volumeWithContours->getVolumeRenderer().uploadColorMappedData(m_UploadableBuffer.getBuffer(), ei.canvasWidthX, ei.canvasWidthY, ei.canvasWidthZ);
	qDebug("Time to upload : %d", t.elapsed());
	
	qDebug("Original Size: width: %d, height: %d, depth: %d", ei.widthX, ei.widthY, ei.widthZ);
	qDebug("Uploading data: width: %d, height: %d, depth: %d", ei.canvasWidthX, ei.canvasWidthY, ei.canvasWidthZ);

	volumeWithContours->setAspectRatio(fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ));
	determineTextureSubCube(sourceManager, volumeWithContours, ei.startSampleX, ei.startSampleY, ei.startSampleZ, ei.endSampleX, ei.endSampleY, ei.endSampleZ, 
		(unsigned int)ei.level, 
		minX, minY, minZ, maxX, maxY, maxZ,
		ei.canvasWidthX, ei.canvasWidthY, ei.canvasWidthZ);

	//qDebug("Done messing with volume viewer");

	volumeWithContours->enableVolumeRendering();

	// this is probably wrong if a border is being used
	// delete this stuff as soon as possible
	//contourManager->setData((unsigned char*)m_ThumbnailBuffer, widthX, widthY, widthZ,
	//	fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ));
	volumeWithContours->getMultiContour()->setData((unsigned char*)buffer->getBuffer(), ei.widthX, ei.widthY, ei.widthZ,
		fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ),
		contourCoordOfSample(minX, ei.startSampleX, ei.endSampleX, sourceManager->getMinX(), sourceManager->getMaxX(), sourceManager->getDimX(), ei.level),
		contourCoordOfSample(minY, ei.startSampleY, ei.endSampleY, sourceManager->getMinY(), sourceManager->getMaxY(), sourceManager->getDimY(), ei.level),
		contourCoordOfSample(minZ, ei.startSampleZ, ei.endSampleZ, sourceManager->getMinZ(), sourceManager->getMaxZ(), sourceManager->getDimZ(), ei.level),
		contourCoordOfSample(maxX, ei.startSampleX, ei.endSampleX, sourceManager->getMinX(), sourceManager->getMaxX(), sourceManager->getDimX(), ei.level),
		contourCoordOfSample(maxY, ei.startSampleY, ei.endSampleY, sourceManager->getMinY(), sourceManager->getMaxY(), sourceManager->getDimY(), ei.level),
		contourCoordOfSample(maxZ, ei.startSampleZ, ei.endSampleZ, sourceManager->getMinZ(), sourceManager->getMaxZ(), sourceManager->getDimZ(), ei.level),
		buffer->getMinX(), buffer->getMinY(), buffer->getMinZ(),
		buffer->getMaxX(), buffer->getMaxY(), buffer->getMaxZ());
}
*/

bool DownLoadManager::testSize(unsigned int x, unsigned int y, unsigned int z) const
{
	return ((x*y*z<=m_BufferX*m_BufferY*m_BufferZ) && x<=512 && y<=512 && z<=512);
}

void DownLoadManager::fillData(SourceManager* sourceManager, VolumeBuffer* buffer, int startSampleX, int startSampleY, int startSampleZ, int endSampleX, int endSampleY, int endSampleZ, unsigned int level, unsigned int variable, unsigned int timeStep) 
{
	// adjust startSample to be >= 0
	if (startSampleX<0) startSampleX=0;
	if (startSampleY<0) startSampleY=0;
	if (startSampleZ<0) startSampleZ=0;

	// adjust endSample to be <= adjustedLast
	// adjustedLast is the last sample for a given level
	int adjustedLastX = ((sourceManager->getDimX()-1)>>level)<<level;
	int adjustedLastY = ((sourceManager->getDimY()-1)>>level)<<level;
	int adjustedLastZ = ((sourceManager->getDimZ()-1)>>level)<<level;
	if (endSampleX>adjustedLastX) endSampleX=adjustedLastX;
	if (endSampleY>adjustedLastY) endSampleY=adjustedLastY;
	if (endSampleZ>adjustedLastZ) endSampleZ=adjustedLastZ;

	// calculate width
	unsigned int widthX = ((endSampleX - startSampleX)/(double)(1<<level) + 1);
	unsigned int widthY = ((endSampleY - startSampleY)/(double)(1<<level) + 1);
	unsigned int widthZ = ((endSampleZ - startSampleZ)/(double)(1<<level) + 1);

	if (!buffer->allocateMemory(widthX, widthY, widthZ)) {
		qDebug("failed to allocate buffer memory");
		return;
	}
	
	/* uncomment for logging and timing 
	FILE* fp = fopen("log.txt", "a+");
	QTime t;
	t.start();
	*/
	// download the data
	sourceManager->getSource()->fillData((char*)buffer->getBuffer(), 
		(unsigned int)startSampleX, (unsigned int)startSampleY, (unsigned int)startSampleZ, 
		(unsigned int)endSampleX, (unsigned int)endSampleY, (unsigned int)endSampleZ, 
		widthX, widthY, widthZ, variable, timeStep);

	/* uncomment for logging and timing
	fprintf(fp, "%8d%8d \n", widthX, t.elapsed());
	fclose(fp);
	*/
}

void DownLoadManager::fillGradientData(SourceManager* sourceManager, VolumeBuffer* buffer, int startSampleX, int startSampleY, int startSampleZ, int endSampleX, int endSampleY, int endSampleZ, unsigned int level, unsigned int variable, unsigned int timeStep) 
{
	// adjust startSample to be >= 0
	if (startSampleX<0) startSampleX=0;
	if (startSampleY<0) startSampleY=0;
	if (startSampleZ<0) startSampleZ=0;

	// adjust endSample to be <= adjustedLast
	// adjustedLast is the last sample for a given level
	int adjustedLastX = ((sourceManager->getDimX()-1)>>level)<<level;
	int adjustedLastY = ((sourceManager->getDimY()-1)>>level)<<level;
	int adjustedLastZ = ((sourceManager->getDimZ()-1)>>level)<<level;
	if (endSampleX>adjustedLastX) endSampleX=adjustedLastX;
	if (endSampleY>adjustedLastY) endSampleY=adjustedLastY;
	if (endSampleZ>adjustedLastZ) endSampleZ=adjustedLastZ;

	// calculate width
	unsigned int widthX = ((endSampleX - startSampleX)/(double)(1<<level) + 1);
	unsigned int widthY = ((endSampleY - startSampleY)/(double)(1<<level) + 1);
	unsigned int widthZ = ((endSampleZ - startSampleZ)/(double)(1<<level) + 1);

	if (!buffer->allocateMemory(widthX, widthY, widthZ, 4)) {
		qDebug("failed to allocate buffer memory");
		return;
	}
	
	/* uncomment for logging and timing 
	FILE* fp = fopen("log.txt", "a+");
	QTime t;
	t.start();
	*/
	// download the data
	sourceManager->getSource()->fillGradientData((char*)buffer->getBuffer(), 
		(unsigned int)startSampleX, (unsigned int)startSampleY, (unsigned int)startSampleZ, 
		(unsigned int)endSampleX, (unsigned int)endSampleY, (unsigned int)endSampleZ, 
		widthX, widthY, widthZ, variable, timeStep);

	/* uncomment for logging and timing
	fprintf(fp, "%8d%8d \n", widthX, t.elapsed());
	fclose(fp);
	*/
}

int DownLoadManager::getStartSample(double windowMin, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const
{
	/* old get start sample  I'll delete it when I'm sure the new one works

	double cellWidth = (dataMax-dataMin)/(double)(dataResolution-1);
	double doublePos = (windowMin-dataMin)/cellWidth;

	// take it down to the nearest integer
	unsigned int flooredPos = (unsigned int)floor(doublePos);

	// make it a multiple of 2^level
	flooredPos = flooredPos >> level;
	flooredPos = flooredPos << level;

	return flooredPos;
	*/
	unsigned int adjustedLast = ((dataResolution-1)>>level);
	double cellWidth = (dataMax-dataMin)/(double)(adjustedLast);
	double doublePos = (windowMin-dataMin)/cellWidth;

	// take it down to the nearest integer
	int flooredPos = (unsigned int)floor(doublePos);

	// divide by 2^level
	//flooredPos = flooredPos >> level;
	//doublePos = doublePos / (double)(1<<level);

	if (doublePos-(double)flooredPos <= 0.5) {
		// one extra
		flooredPos--;
	}

	// when not using texture border
	if (flooredPos<0) {
		flooredPos=0;
	}

	// multiply by 2^level
	flooredPos = flooredPos * (1<<level);


	return flooredPos;
}

int DownLoadManager::getEndSample(double windowMax, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const
{
	unsigned int adjustedLast = ((dataResolution-1)>>level);
	double cellWidth = (dataMax-dataMin)/(double)(adjustedLast);
	double doublePos = (windowMax-dataMin)/cellWidth;

	// take it down to the nearest integer
	int flooredPos = (unsigned int)floor(doublePos);

	// divide by 2^level
	//flooredPos = flooredPos >> level;
	//doublePos = doublePos / (double)(1<<level);


	flooredPos++;

	if ((double)flooredPos-doublePos < 0.5) {
		// one extra
		flooredPos++;
	}

	// when not using texture border
	if (flooredPos>=0 && (unsigned int)flooredPos>adjustedLast) {
		flooredPos=adjustedLast;
	}


	// multiply by s^level
	flooredPos = flooredPos * (1<<level);


	return flooredPos;
}

void DownLoadManager::setDefaults()
{
	m_BufferX = 0;
	m_BufferY = 0;
	m_BufferZ = 0;

	resetError();
}

bool DownLoadManager::allocateBuffer(unsigned int bufferX, unsigned int bufferY, unsigned int bufferZ)
{
	destroyBuffer();

	// free current memory
	/*m_UploadableBuffer = new unsigned char[bufferX*bufferY*bufferZ];
	m_ZoomedBuffer = new unsigned char[bufferX*bufferY*bufferZ];
	m_ThumbnailBuffer = new unsigned char[bufferX*bufferY*bufferZ];

	if (m_UploadableBuffer && m_ZoomedBuffer && m_ThumbnailBuffer) {*/
		m_BufferX = bufferX;
		m_BufferY = bufferY;
		m_BufferZ = bufferZ;
		return true;
	/*}
	else {
		destroyBuffer();
		return false;
	}*/
}

void DownLoadManager::destroyBuffer()
{
	/*delete [] m_UploadableBuffer;
	m_UploadableBuffer = 0;
	delete [] m_ZoomedBuffer;
	m_ZoomedBuffer = 0;
	delete [] m_ThumbnailBuffer;
	m_ThumbnailBuffer = 0;*/
	m_BufferX = 0;
	m_BufferY = 0;
	m_BufferZ = 0;
}

/*
void DownLoadManager::copyToUploadableBuffer(SourceManager* sourceManager, VolumeBuffer* buffer,
											 int startSampleX, int startSampleY, int startSampleZ, 
											 int endSampleX, int endSampleY, int endSampleZ, unsigned int level,
											 unsigned int cwidthX, unsigned int cwidthY, unsigned int cwidthZ)
{
	unsigned int offsetX=0, offsetY=0, offsetZ=0;
	// adjust startSample to be >= 0
	if (startSampleX<0) {startSampleX=0;offsetX=1;}
	if (startSampleY<0) {startSampleY=0;offsetY=1;}
	if (startSampleZ<0) {startSampleZ=0;offsetZ=1;}

	// adjust endSample to be <= adjustedLast
	// adjustedLast is the last sample for a given level
	int adjustedLastX = ((sourceManager->getDimX()-1)>>level)<<level;
	int adjustedLastY = ((sourceManager->getDimY()-1)>>level)<<level;
	int adjustedLastZ = ((sourceManager->getDimZ()-1)>>level)<<level;
	if (endSampleX>adjustedLastX) endSampleX=adjustedLastX;
	if (endSampleY>adjustedLastY) endSampleY=adjustedLastY;
	if (endSampleZ>adjustedLastZ) endSampleZ=adjustedLastZ;

	// calculate width
	unsigned int widthX = ((endSampleX - startSampleX)/(double)(1<<level) + 1);
	unsigned int widthY = ((endSampleY - startSampleY)/(double)(1<<level) + 1);
	unsigned int widthZ = ((endSampleZ - startSampleZ)/(double)(1<<level) + 1);

	if (!m_UploadableBuffer.allocateMemory(cwidthX, cwidthY, cwidthZ)) {
		qDebug("Failed to allocate memory");
		return;
	}

	unsigned int j, k;
	unsigned int targetSlice, targetLine;
	unsigned int sourceSlice, sourceLine;

	for (k=0; k<widthZ; k++) {
		targetSlice = (k+offsetZ)*cwidthX*cwidthY;
		sourceSlice = k*widthX*widthY;
		for (j=0; j<widthY; j++) {
			targetLine = (j+offsetY)*cwidthX+offsetX;
			sourceLine = j*widthX;
			memcpy(m_UploadableBuffer.getBuffer()+targetSlice+targetLine, buffer->getBuffer()+sourceSlice+sourceLine, widthX);
		}
	}
}
*/

/*
void DownLoadManager::determineTextureSubCube(SourceManager* sourceManager, VolumeWithContours* volumeWithContours, 
											  int startSampleX, int startSampleY, int startSampleZ, 
											  int endSampleX, int endSampleY, int endSampleZ, unsigned int level,
											  double minX, double minY, double minZ, double maxX, double maxY, double maxZ,
											  unsigned int cwidthX, unsigned int cwidthY, unsigned int cwidthZ)
{

	// this is the old determin texture subcube  I'll delete it when I'm sure the new one works
	//viewer->setTextureSubCube(
	//	texCoordOfPixel(0, cwidthX),
	//	texCoordOfPixel(0, cwidthY),
	//	texCoordOfPixel(0, cwidthZ),
	//	texCoordOfPixel(widthX-1, cwidthX),
	//	texCoordOfPixel(widthY-1, cwidthY),
	//	texCoordOfPixel(widthZ-1, cwidthZ)
	//	);
	//	
	volumeWithContours->getVolumeRenderer().setTextureSubCube(
		texCoordOfSample(minX, startSampleX, endSampleX, cwidthX, sourceManager->getMinX(), sourceManager->getMaxX(), sourceManager->getDimX(), level),
		texCoordOfSample(minY, startSampleY, endSampleY, cwidthY, sourceManager->getMinY(), sourceManager->getMaxY(), sourceManager->getDimY(), level),
		texCoordOfSample(minZ, startSampleZ, endSampleZ, cwidthZ, sourceManager->getMinZ(), sourceManager->getMaxZ(), sourceManager->getDimZ(), level),
		texCoordOfSample(maxX, startSampleX, endSampleX, cwidthX, sourceManager->getMinX(), sourceManager->getMaxX(), sourceManager->getDimX(), level),
		texCoordOfSample(maxY, startSampleY, endSampleY, cwidthY, sourceManager->getMinY(), sourceManager->getMaxY(), sourceManager->getDimY(), level),
		texCoordOfSample(maxZ, startSampleZ, endSampleZ, cwidthZ, sourceManager->getMinZ(), sourceManager->getMaxZ(), sourceManager->getDimZ(), level)
		);


}

*/
