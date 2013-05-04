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

// RawIVSource.h: interface for the RawIVSource class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAWIVSOURCE_H__92FE6BA3_8715_4988_BC88_482288C894EC__INCLUDED_)
#define AFX_RAWIVSOURCE_H__92FE6BA3_8715_4988_BC88_482288C894EC__INCLUDED_

#include <VolumeFileTypes/VolumeSource.h>

#include <qstring.h>
#include <qfileinfo.h>
#include <qdir.h>
#include <qprogressdialog.h>

class RawIVFileImpl;

///\class RawIVSource RawIVSource.h
///\author Anthony Thane
///\deprecated This class has been superceded by VolumeFileSource.
class RawIVSource : public VolumeSource  
{
public:
	RawIVSource(const QString& filename, const QString& cacheDir);
	virtual ~RawIVSource();

	virtual bool open(QWidget* parent);
	virtual void close();

	virtual void fillThumbnail(char* data, 
		unsigned int xDim, unsigned int yDim, unsigned int zDim, uint variable, uint timeStep);

	virtual void fillData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep);

	virtual DownLoadFrequency interactiveUpdateHint() const;


protected:
	enum DataType {DataTypeByte, DataTypeShort, DataTypeFloat};

	// sets up default values for member variables, does not do memory allocation or deallocation
	void setDefaults();

	bool allocateBuffer(unsigned int size);
	bool forceAllocateBuffer(unsigned int size);
	void destroyBuffer();

	// reads in the rawIV header and stores the dimensions and stats
	bool readHeader();

	void extractFromRawIVFile(char* data, uint xMin, uint yMin, uint zMin, 
		uint xMax, uint yMax, uint zMax);
	void extractFromCacheFile(char* data, uint xMin, uint yMin, uint zMin, 
		uint xMax, uint yMax, uint zMax, 
		uint newDimX, uint newDimY, uint newDimZ, QFile& file);

	bool createCache(const QString& fileName, const QString& cacheRoot);
	void createNewCache(QFileInfo fileName, QDir cacheDir);
	void createCacheRecordFile(QFileInfo fileInfo, QDir cacheDir);
	void clearCache(QDir cacheDir);

	void createMipmapLevels();
	bool createMipmapLevelZeroFromShortRawIV(QFile& targetFile);
	bool createMipmapLevelZeroFromFloatRawIV(QFile& targetFile);

	bool createMipmapLevelFromRawIV(QFile& targetFile);
	bool createMipmapLevelFromCacheFile(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ);

	bool createMipmapLevelFromRawIVGaussian(QFile& targetFile);
	bool createMipmapLevelFromCacheFileGaussian(QFile& source, QFile& target, unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int offset=0);

	bool cacheOutdated(QFileInfo fileName, QDir cacheDir);
	bool checkMipmapLevel(uint level, uint dimX, uint dimY, uint dimZ);

	inline QString mipmapLevelFileName(unsigned int level);

	void incrementProgress(int amount);
	int determineCacheWorkAmount();

	// the directory where cache files are stored
	const QDir m_CacheDir;
	// the fileinfo object which has information about the volume file
	const QFileInfo m_FileInfo;
	// the file object representing the rawIV file
	QFile m_File;
	// boolean indicating whether the file is currently open
	bool m_IsOpen;

	// a buffer for file input
	unsigned char* m_Buffer;
	// the current size of the buffer
	unsigned int m_BufferSize;

	// the parent widget of the progress dialog
	QWidget* m_ProgressParent;
	// the progress dialog
	QProgressDialog* m_ProgressDialog;
	// current progress
	int m_ProgressValue;

	// the rawiv file type
	DataType m_DataType;

	class UserCancelException {

	};

};

inline QString RawIVSource::mipmapLevelFileName(unsigned int level)
{
	QString levelString;
	levelString.sprintf("%0d",level);
	return m_CacheDir.absPath() + "/" + m_FileInfo.fileName() + "/" + m_FileInfo.baseName() + levelString + ".cache";
}


#endif // !defined(AFX_RAWIVSOURCE_H__92FE6BA3_8715_4988_BC88_482288C894EC__INCLUDED_)
