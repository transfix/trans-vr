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

// VolumeFileFactory.h: interface for the VolumeFileFactory class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMEFILEFACTORY_H__1146F195_3CA9_485A_89CB_918C60C1411A__INCLUDED_)
#define AFX_VOLUMEFILEFACTORY_H__1146F195_3CA9_485A_89CB_918C60C1411A__INCLUDED_

#include <vector>
#include <boost/shared_ptr.hpp>

#include <qmap.h>
class VolumeFile;

///\class VolumeFileFactory VolumeFileFactory.h
///\author Anthony Thane
///\brief This class provides an eassy interface for managing all the different VolumeFile instances.
/// You simple pass it a filename extension or a filename, and it returns a VolumeFile that will
/// handle that extension/filename.
class VolumeFileFactory  
{
public:
	static VolumeFileFactory ms_MainFactory;
	virtual ~VolumeFileFactory();

///\fn VolumeFile* getLoader(const QString& fileName)
///\brief Returns a VolumeFile that can handle a given file
///\param fileName A QString containing a path to a file
///\return A pointer to a VolumeFile that can read the file at the specified path or NULL if no VolumeFile can handle the file.
	VolumeFile* getLoader(const QString& fileName);
///\fn VolumeFile* getLoaderForExtension(const QString& extension)
///\brief Returns a VolumeFile that can handle files with a given extension
///\param extension A QString containing a filename extension
///\return A pointer to a VolumeFile that can read files with the specified extension or NULL if no VolumeFile exists to handle that type of file.
	VolumeFile* getLoaderForExtension(const QString& extension);
///\fn QString getFilterString()
///\brief Returns a string containing all the available VolumeFile formats that can be used by a
/// QFileDialog.
///\return A QString that contains a QFileDialog-compatible filter.
	QString getFilterString();

protected:
	VolumeFileFactory();

	QString getAllExtensions();
	VolumeFile* tryAll(const QString& fileName);
	void addVolumeFileType(VolumeFile* type);
	QMap<QString, VolumeFile*> m_ExtensionMap;
	QMap<QString, VolumeFile*> m_FilterMap;
	std::vector<boost::shared_ptr<VolumeFile> > m_VolumeFileFactoryHandles; //responsible for cleanup of file type objects
};

#endif // !defined(AFX_VOLUMEFILEFACTORY_H__1146F195_3CA9_485A_89CB_918C60C1411A__INCLUDED_)
