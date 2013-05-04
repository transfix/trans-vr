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

// DownLoadManager.h: interface for the DownLoadManager class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_DOWNLOADMANAGER_H__F7796471_866E_435B_8B59_994879DD405D__INCLUDED_)
#define AFX_DOWNLOADMANAGER_H__F7796471_866E_435B_8B59_994879DD405D__INCLUDED_

#include <VolumeFileTypes/VolumeSource.h>
#include <math.h>
#include <Filters/Filter.h>
#include <VolumeFileTypes/VolumeBuffer.h>
#include <VolumeFileTypes/SourceManager.h>
#include <qwidget.h>

class ContourManager;

///\struct ExtractionInformation
///\brief Contains info necessary to extract a subvolume from a cache
struct ExtractionInformation {
	int startSampleX, endSampleX;
	int startSampleY, endSampleY;
	int startSampleZ, endSampleZ;
	unsigned int widthX, widthY, widthZ;
	unsigned int canvasWidthX, canvasWidthY, canvasWidthZ;
	int level;
};

///\class DownLoadManager DownLoadManager.h
///\author Anthony Thane
///\author John Wiggins
///\brief The DownLoadManager provides a nice interface with VolumeFileSource
///	for accessing volume data which can be uploaded to a graphics card. It
///	handles all the nasty details of reading data from the right mipmap level.
///	It also aids in the subvolume saving process by converting spatial
///	coordinates to ijk coordinates.
class DownLoadManager  
{
public:
///\fn DownLoadManager(unsigned int bufferX=128, unsigned int bufferY=128, unsigned int bufferZ=128)
///\brief The constructor (note that the buffer is not actually constructed. ever.)
///\param bufferX The size of the download buffer in X
///\param bufferY The size of the download buffer in Y
///\param bufferZ The size of the download buffer in Z
	DownLoadManager(unsigned int bufferX=128, unsigned int bufferY=128, unsigned int bufferZ=128);
	virtual ~DownLoadManager();

///\fn bool getThumbnail(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time)
///\brief Grabs the thumbnail volume from the given SourceManager's VolumeSource.
///\param dest The destination VolumeBuffer
///\param sourceManager A SourceManager instance that has a valid VolumeSource
///\param var A variable index
///\param time A time step index
///\return A bool indicating success or failure
	bool getThumbnail(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time);
///\fn bool getData(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
///\brief Fills a VolumeBuffer with data. The bounding box is given in spatial coordinates and the function determines what 
/// mipmap level the data needs to be fetched from.
///\param dest The destination VolumeBuffer
///\param sourceManager A SourceManager instance that has a valid VolumeSource
///\param var A variable index
///\param time A time step index
///\param minX The extent minimum X coordinate
///\param minY The extent minimum Y coordinate
///\param minZ The extent minimum Z coordinate
///\param maxX The extent maximum X coordinate
///\param maxY The extent maximum Y coordinate
///\param maxZ The extent maximum Z coordinate
///\return A bool indicating success or failure
	bool getData(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time, double minX, double minY, double minZ, double maxX, double maxY, double maxZ);
///\fn bool getGradientData(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
///\brief The analogue of getData for gradient data
///\param dest The destination VolumeBuffer
///\param sourceManager A SourceManager instance that has a valid VolumeSource
///\param var A variable index
///\param time A time step index
///\param minX The extent minimum X coordinate
///\param minY The extent minimum Y coordinate
///\param minZ The extent minimum Z coordinate
///\param maxX The extent maximum X coordinate
///\param maxY The extent maximum Y coordinate
///\param maxZ The extent maximum Z coordinate
///\return A bool indicating success or failure
	bool getGradientData(VolumeBuffer* dest, SourceManager* sourceManager, unsigned int var, unsigned int time, double minX, double minY, double minZ, double maxX, double maxY, double maxZ);

///\fn void applyFilter(SourceManager* sourceManager, QWidget* parent, Filter* filter)
///\deprecated This interface is not used for running Filters. See NewVolumeMainWindow for actual implementation.
	void applyFilter(SourceManager* sourceManager, QWidget* parent, Filter* filter);

///\fn bool getExtentCoords(SourceManager* sourceManager, double minX, double minY, double minZ,	double maxX, double maxY, double maxZ, int& startX, int& startY, int& startZ, int& endX, int& endY, int& endZ)
///\brief Provides data coords for subvolume extraction
///\param sourceManager A SourceManager instance that has a valid VolumeSource
///\param minX The extent minimum X coordinate
///\param minY The extent minimum Y coordinate
///\param minZ The extent minimum Z coordinate
///\param maxX The extent maximum X coordinate
///\param maxY The extent maximum Y coordinate
///\param maxZ The extent maximum Z coordinate
///\param startX The dataset minimum X coordinate (in samples)
///\param startY The dataset minimum Y coordinate (in samples)
///\param startZ The dataset minimum Z coordinate (in samples)
///\param endX The dataset maximum X coordinate (in samples)
///\param endY The dataset maximum Y coordinate (in samples)
///\param endZ The dataset maximum Z coordinate (in samples)
///\return A bool indicating success or failure
	bool getExtentCoords(SourceManager* sourceManager,
												double minX, double minY, double minZ,
												double maxX, double maxY, double maxZ,
												int& startX, int& startY, int& startZ,
												int& endX, int& endY, int& endZ);

///\fn bool error() const
///\brief Reports whether an error has occured
///\return A bool indicating whether or not an error occured
	bool error() const;
///\fn bool errorMustRestart() const
///\brief Reports whether a fatal error has occured
///\return A bool indicating whether or not an error occured
	bool errorMustRestart() const;
///\fn const QString& errorReason() const
///\brief Provides a human-readable description of an error
///\return A QString describing an error
	const QString& errorReason() const;

///\fn void resetError()
///\brief Resets the error state for non-fatal errors
	void resetError();
///\fn bool allocateBuffer(unsigned int bufferX, unsigned int bufferY, unsigned int bufferZ)
///\deprecated Doesn't really do anything. The object does not use its VolumeBuffer.
	bool allocateBuffer(unsigned int bufferX, unsigned int bufferY, unsigned int bufferZ);

protected:
	bool testSize(unsigned int x, unsigned int y, unsigned int z) const;
	void fillData(SourceManager* sourceManager, VolumeBuffer* buffer, int startSampleX, int startSampleY, int startSampleZ, int endSampleX, int endSampleY, int endSampleZ, unsigned int level, unsigned int variable, unsigned int timeStep); 
	void fillGradientData(SourceManager* sourceManager, VolumeBuffer* buffer, int startSampleX, int startSampleY, int startSampleZ, int endSampleX, int endSampleY, int endSampleZ, unsigned int level, unsigned int variable, unsigned int timeStep); 
	void setError(const QString& reason, bool mustRestart = false);
	void setError(VolumeSource* source);
	ExtractionInformation getExtractionInformation(SourceManager* sourceManager, double minX, double minY, double minZ, double maxX, double maxY, double maxZ) const;
	int getStartSample(double windowMin, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const;
	int getEndSample(double windowMax, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const;
	
	inline bool isCloseEnough(double val1, double val2) const;

	void setDefaults();
	void destroyBuffer();
	//void copyToUploadableBuffer(SourceManager* sourceManager, VolumeBuffer* buffer, int startSampleX, int startSampleY, int startSampleZ, int endSampleX, int endSampleY, int endSampleZ, unsigned int level,
	//	unsigned int cwidthX, unsigned int cwidthY, unsigned int cwidthZ);
	//void determineTextureSubCube(SourceManager* sourceManager, VolumeWithContours* volumeWithContours, int startSampleX, int startSampleY, int startSampleZ, int endSampleX, int endSampleY, int endSampleZ, unsigned int level,
	//	double minX, double minY, double minZ, double maxX, double maxY, double maxZ,
	//	unsigned int cwidthX, unsigned int cwidthY, unsigned int cwidthZ);
	inline double texCoordOfSample(double sample, int firstPixel, int lastPixel, int canvasWidth, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const;
	inline double contourCoordOfSample(double sample, int firstPixel, int lastPixel, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const;
	inline unsigned int upToPowerOfTwo(unsigned int value) const;
	
	//void upload(SourceManager* sourceManager, VolumeWithContours* volumeWithContours, VolumeBuffer* buffer, ExtractionInformation ei);

	// data buffers
	VolumeBuffer m_UploadableBuffer;

	// data details
	unsigned int m_BufferX, m_BufferY, m_BufferZ;

	// error details
	bool m_Error;
	bool m_ErrorMustRestart;
	QString m_ErrorReason;
};

inline bool DownLoadManager::isCloseEnough(double val1, double val2) const
{
	return (fabs(val1-val2) < 0.00001);
}

inline double DownLoadManager::texCoordOfSample(double sample, int firstPixel, int lastPixel, int canvasWidth, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const
{
	int dummy = 0;
	dummy = lastPixel;
	unsigned int adjustedLast = ((dataResolution-1)>>level);
	double cellWidth = (dataMax-dataMin)/(double)(adjustedLast);
	double doublePos = (sample-dataMin)/cellWidth;

	double doubleStart = (double)firstPixel/(double)(1<<level);
	//double doubleEnd = (double)lastPixel/(double)(1<<level);


	/* with texture border
	return (doublePos-(doubleStart+0.5))/(double)(canvasWidth-2);
	*/
	// without texture border
	return (doublePos-(doubleStart-0.5))/(double)(canvasWidth);
}

inline double DownLoadManager::contourCoordOfSample(double sample, int firstPixel, int lastPixel, double dataMin, double dataMax, unsigned int dataResolution, unsigned int level) const
{
	unsigned int adjustedLast = ((dataResolution-1)>>level);
	double cellWidth = (dataMax-dataMin)/(double)(adjustedLast);
	double doublePos = (sample-dataMin)/cellWidth;

	double doubleStart = (double)firstPixel/(double)(1<<level);
	double doubleEnd = (double)lastPixel/(double)(1<<level);


	/* with texture border
	return (doublePos-(doubleStart+0.5))/(double)(canvasWidth-2);
	*/
	// without texture border
	return (doublePos-doubleStart)/(doubleEnd-doubleStart);
}

inline unsigned int DownLoadManager::upToPowerOfTwo(unsigned int value) const
{
	unsigned int c = 0;
	unsigned int v = value;

	// round down to nearest power of two
	while (v>1) {
		v = v>>1;
		c++;
	}

	// if that isn't exactly the original value
	if ((v<<c)!=value) {
		// return the next power of two
		return (v<<(c+1));
	}
	else {
		// return this power of two
		return (v<<c);
	}
}


#endif // !defined(AFX_DOWNLOADMANAGER_H__F7796471_866E_435B_8B59_994879DD405D__INCLUDED_)

