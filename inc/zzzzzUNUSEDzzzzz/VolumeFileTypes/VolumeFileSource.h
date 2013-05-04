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

// VolumeFileSource.h: interface for the VolumeFileSource class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMEFILESOURCE_H__92FE6BA3_8715_4988_BC88_482288C894EC__INCLUDED_)
#define AFX_VOLUMEFILESOURCE_H__92FE6BA3_8715_4988_BC88_482288C894EC__INCLUDED_

#include <boost/scoped_ptr.hpp>

// This is a unix only kludge for large file support until we port to Qt 4.0
#ifdef LARGEFILE_KLUDGE
#include <VolumeFileTypes/pfile.h>
#endif

#include <VolumeFileTypes/VolumeSource.h>

#include <VolumeFileTypes/VolumeFile.h>
#include <VolumeFileTypes/VolumeFileFactory.h>


#include <qstring.h>
#include <qfileinfo.h>
#include <qdir.h>
#include <q3progressdialog.h>

// uncomment to turn on threading
//#define VFS_USE_THREADING

#ifdef VFS_USE_THREADING
#include <q3ptrlist.h>
#include "WorkerThread.h"
#endif


class RawIVFileImpl;

///\class VolumeFileSource VolumeFileSource.h
///\author Anthony Thane
///\author John Wiggins
///\brief The class VolumeFileSource provides a large subset of Volume Rover's
///	functionality. It is responsible for all local access to volume data. As
///	part of this responsibility it can read data from any file that has a
///	working VolumeFile implementation. It also handles the building of the
///	cache, so that already opened datasets can be accessed quickly.
class VolumeFileSource : public VolumeSource  
{
public:
///\fn VolumeFileSource(const QString& filename, const QString& cacheDir)
///\brief The constructor
///\param filename A file to open for reading
///\param cacheDir A path to a directory where a cache of this file already
///	exists or can be created.
	VolumeFileSource(const QString& filename, const QString& cacheDir);
	virtual ~VolumeFileSource();

///\fn virtual bool open(QWidget* parent)
///\brief This fuction opens the file specified in the constructor and
///	generates a cache if necessary.
///\param parent A QWidget that can be used to instantiate UI stuff (a QProgressDialog in this case)
///\return A bool indicating success or failure. This function will fail if
///	the filename specified is not valid, or if the cache fails to be built.
	virtual bool open(QWidget* parent);
///\fn virtual void close()
///\brief Closes a previously open()'d file.
	virtual void close();

	virtual void fillThumbnail(char* data, 
		unsigned int xDim, unsigned int yDim, unsigned int zDim, uint variable, uint timeStep);

	virtual void fillData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep);
	virtual bool fillGradientData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep);

	virtual bool readRawData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, uint variable, uint timeStep);
	

	virtual DownLoadFrequency interactiveUpdateHint() const;
	
	virtual unsigned int getVariableType(unsigned int variable) const;
	virtual QString getVariableName(unsigned int variable) const;

	virtual double getFunctionMinimum(unsigned int variable, unsigned int timeStep) const;
	virtual double getFunctionMaximum(unsigned int variable, unsigned int timeStep) const;

	virtual QString getContourSpectrumFileName(unsigned int variable, unsigned int timeStep);
	virtual QString getContourTreeFileName(unsigned int variable, unsigned int timeStep);

	virtual void computeContourSpectrum(QObject *obj, unsigned int variable, unsigned int timeStep);
	virtual void computeContourTree(QObject *obj, unsigned int variable, unsigned int timeStep);
	
#ifdef VFS_USE_THREADING
	virtual void cleanUpWorkerThread(int thid);
#endif

///\fn static QString getFilter()
///\brief Returns a "filter" string that can be used with QFileDialog. The
///	string contains all the file formats accessible to VolumeFileSource.
///\return A QString that can be passed to QFileDialog
	static QString getFilter();

protected:
	enum DataType {DataTypeByte, DataTypeShort, DataTypeFloat};

///\fn void setDefaults()
///\brief sets up default values for member variables, does not do memory
/// allocation or deallocation
	void setDefaults();

	bool allocateBuffer(unsigned int size);
	bool forceAllocateBuffer(unsigned int size);
	void destroyBuffer();
	
	// contour tree/spectrum thread functions
///\fn void callCreateContourSpectrum(void *args)
///\brief This is a stub for running the contour spectrum calculation in its
///	own thread. It is not currently being used.
	void callCreateContourSpectrum(void *args);
///\fn void callCreateContourTree(void *args)
///\brief This is a stub for running the contour tree calculation in its
///	own thread. It is not currently being used.
	void callCreateContourTree(void *args);

///\fn bool readHeader()
///\brief reads in the rawIV header and stores the dimensions and stats
	bool readHeader();

///\fn void extractFromRawIVFile(char* data, uint xMin, uint yMin, uint zMin, uint xMax, uint yMax, uint zMax, unsigned int variable, unsigned int timeStep)
///\brief Despite the misleading name, this function reads data from a
///	VolumeFile (m_VolumeFile to be precise).
	void extractFromRawIVFile(char* data, uint xMin, uint yMin, uint zMin, 
		uint xMax, uint yMax, uint zMax, unsigned int variable, unsigned int timeStep);
///\fn void extractFromCacheFile(char* data, uint xMin, uint yMin, uint zMin, uint xMax, uint yMax, uint zMax, uint newDimX, uint newDimY, uint newDimZ, QFile& file)
///\brief This function reads data from a cache file.
	void extractFromCacheFile(char* data, uint xMin, uint yMin, uint zMin, 
		uint xMax, uint yMax, uint zMax, 
#ifdef LARGEFILE_KLUDGE
		uint newDimX, uint newDimY, uint newDimZ, PFile& file);
#else
		uint newDimX, uint newDimY, uint newDimZ, QFile& file);
#endif
///\fn void extractFromGradientCacheFile(char* data, uint xMin, uint yMin, uint zMin, uint xMax, uint yMax, uint zMax, uint newDimX, uint newDimY, uint newDimZ, QFile& file)
///\brief This function reads gradient data from a cache file.
	void extractFromGradientCacheFile(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, 
#ifdef LARGEFILE_KLUDGE
		uint newDimX, uint newDimY, uint newDimZ, PFile& file);
#else
		uint newDimX, uint newDimY, uint newDimZ, QFile& file);
#endif

///\fn bool createCache(const QString& fileName, const QString& cacheRoot)
///\brief Creates a new cache for the opened file.
///\param fileName The path to the file
///\param cacheRoot The cache directory
///\return A bool indicating success or failure
	bool createCache(const QString& fileName, const QString& cacheRoot);
///\fn void createNewCache(QFileInfo fileName, QDir cacheDir)
///\brief This function does the actual work of creating a new cache
///\param fileName A QFileInfo for the file that the cache is being created for
///\param cacheDir The directory that contains the file's cache. This is a directory inside the main cache directory.
	void createNewCache(QFileInfo fileName, QDir cacheDir);
///\fn void createCacheRecordFile(QFileInfo fileInfo, QDir cacheDir)
///\brief Creates the recod file for a file's cache
///\param fileInfo A QFileInfo for the file that the cache is being created for
///\param cacheDir The directory that contains the file's cache. This is a directory inside the main cache directory.
	void createCacheRecordFile(QFileInfo fileInfo, QDir cacheDir);
///\fn void clearCache(QDir cacheDir)
///\brief Clears a files cache directory of cache files
///\param cacheDir The directory that contains a file's cache
	void clearCache(QDir cacheDir);

///\fn void createMipmapLevels(unsigned int variable, unsigned int timeStep);
///\brief Creates the cache mipmap for a specific variable and time step
	void createMipmapLevels(unsigned int variable, unsigned int timeStep);
///\fn void createContourSpectrum(unsigned int variable, unsigned int timeStep);
///\brief Computes the contour spectrum for a specific variable and time step
	void createContourSpectrum(unsigned int variable, unsigned int timeStep);
///\fn void createContourTree(unsigned int variable, unsigned int timeStep);
///\brief Computes the contour tree for a specific variable and time step
	void createContourTree(unsigned int variable, unsigned int timeStep);

#ifdef LARGEFILE_KLUDGE
	bool createMipmapLevelZeroFromShortRawIV(PFile& targetFile, unsigned int variable, unsigned int timeStep);
	bool createMipmapLevelZeroFromFloatRawIV(PFile& targetFile, unsigned int variable, unsigned int timeStep);

	bool createMipmapLevelFromRawIV(PFile& targetFile, unsigned int variable, unsigned int timeStep);
	bool createMipmapLevelFromCacheFile(PFile& source, PFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ);

	bool createMipmapLevelFromRawIVGaussian(PFile& targetFile, unsigned int variable, unsigned int timeStep);
	bool createMipmapLevelFromCacheFileGaussian(PFile& source, PFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int offset=0);

	void createGradientFromCharRawiv(PFile& targetFile, unsigned int variable, unsigned int timeStep);
	void createGradientFromShortRawiv(PFile& targetFile, unsigned int variable, unsigned int timeStep);
	void createGradientFromFloatRawiv(PFile& targetFile, unsigned int variable, unsigned int timeStep);
	void createGradientFromCacheFile(PFile& source, PFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ);
#else
	bool createMipmapLevelZeroFromShortRawIV(QFile& targetFile, unsigned int variable, unsigned int timeStep);
	bool createMipmapLevelZeroFromFloatRawIV(QFile& targetFile, unsigned int variable, unsigned int timeStep);

	bool createMipmapLevelFromRawIV(QFile& targetFile, unsigned int variable, unsigned int timeStep);
	bool createMipmapLevelFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ);

	bool createMipmapLevelFromRawIVGaussian(QFile& targetFile, unsigned int variable, unsigned int timeStep);
	bool createMipmapLevelFromCacheFileGaussian(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int offset=0);

	void createGradientFromCharRawiv(QFile& targetFile, unsigned int variable, unsigned int timeStep);
	void createGradientFromShortRawiv(QFile& targetFile, unsigned int variable, unsigned int timeStep);
	void createGradientFromFloatRawiv(QFile& targetFile, unsigned int variable, unsigned int timeStep);
	void createGradientFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ);
#endif

	void createContourSpectrumFromRawIV(QFile& target, unsigned int variable, unsigned int timeStep);
	void createContourSpectrumFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ);

	void createContourTreeFromRawIV(QFile& target, unsigned int variable, unsigned int timeStep);
	void createContourTreeFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ);

///\fn bool cacheOutdated(QFileInfo fileName, QDir cacheDir);
///\brief Checks to see if the cache for a file is outdated or not. A cache is
///	outdated if the file has moved or been modified since the cache was built.
///	It is also outdated in the rare situation that the cacheing process is
///	modified in Volume Rover. (see CACHE_RECORD_VERSION_NUMBER in
/// VolumeFileSource.cpp)
	bool cacheOutdated(QFileInfo fileName, QDir cacheDir);
	bool checkMipmapLevel(uint level, unsigned int variable, unsigned int timeStep, uint dimX, uint dimY, uint dimZ);
	bool checkAllMipmapLevels(unsigned int variable, unsigned int timeStep);

	inline QString mipmapLevelFileName(unsigned int level, unsigned int variable, unsigned int timeStep);
	inline QString contourSpectrumFileName(unsigned int variable, unsigned int timeStep);
	inline QString contourTreeFileName(unsigned int variable, unsigned int timeStep);
	inline QString gradientLevelFileName(unsigned int level, unsigned int variable, unsigned int timeStep);

	void incrementProgress(int amount);
	qulonglong determineCacheWorkAmount();
	qulonglong determineWorkForVariable(unsigned int variable);

	// the directory where cache files are stored
	const QDir m_CacheDir;
	// the fileinfo object which has information about the volume file
	const QFileInfo m_FileInfo;
	// the file object representing the rawIV file
        boost::scoped_ptr<VolumeFile> m_VolumeFile;

	// a buffer for file input
	unsigned char* m_Buffer;
	// the current size of the buffer
	unsigned int m_BufferSize;

	// the parent widget of the progress dialog
	QWidget* m_ProgressParent;
	// the progress dialog
	Q3ProgressDialog* m_ProgressDialog;
	// current progress
	qulonglong m_ProgressValue;
	qulonglong m_OnePercentProgress;

	// function min/max values
	double* m_FuncMin;
	double* m_FuncMax;

#ifdef VFS_USE_THREADING
	// a list to hold running WorkerThread instances
	Q3PtrList<WorkerThread> m_WorkerList;
#endif
	
	// boolean indicating whether the file is currently open
	bool m_IsOpen;
	
	///\class UserCancelException
	///\brief An exception that is thrown whenever the user cancels the cache building process.
	class UserCancelException {

	};

};

// these two structs are just containers for passing args to a WorkerThread
///\struct ContourSpectrumArgs
///\brief This is a container for passing function arguments to a WorkerThread
struct ContourSpectrumArgs {
	VolumeFileSource *thisPtr;
	int var;
	int timeStep;
};

///\struct ContourTreeArgs
///\brief This is a container for passing function arguments to a WorkerThread
struct ContourTreeArgs {
	VolumeFileSource *thisPtr;
	int var;
	int timeStep;
};

inline QString VolumeFileSource::mipmapLevelFileName(unsigned int level, unsigned int variable, unsigned int timeStep)
{
	QString levelString;
	levelString.sprintf("v%02dt%02d%01d",variable, timeStep, level);
	return m_CacheDir.absPath() + "/" + m_FileInfo.fileName() + "/" + m_FileInfo.baseName() + levelString + ".cache";
}

inline QString VolumeFileSource::contourSpectrumFileName(unsigned int variable, unsigned int timeStep)
{
	QString levelString;
	levelString.sprintf("v%02dt%02d",variable, timeStep);
	return m_CacheDir.absPath() + "/" + m_FileInfo.fileName() + "/" + m_FileInfo.baseName() + levelString + ".spectrum";
}

inline QString VolumeFileSource::contourTreeFileName(unsigned int variable, unsigned int timeStep)
{
	QString levelString;
	levelString.sprintf("v%02dt%02d",variable, timeStep);
	return m_CacheDir.absPath() + "/" + m_FileInfo.fileName() + "/" + m_FileInfo.baseName() + levelString + ".tree";
}

inline QString VolumeFileSource::gradientLevelFileName(unsigned int level, unsigned int variable, unsigned int timeStep)
{
	QString levelString;
	levelString.sprintf("v%02dt%02d%01d",variable, timeStep, level);
	return m_CacheDir.absPath() + "/" + m_FileInfo.fileName() + "/" + m_FileInfo.baseName() + levelString + ".grad";
}

#endif // !defined(AFX_VOLUMEFILESOURCE_H__92FE6BA3_8715_4988_BC88_482288C894EC__INCLUDED_)
