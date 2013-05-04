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

// VolumeFileFactory.cpp: implementation of the VolumeFileFactory class.
//
//////////////////////////////////////////////////////////////////////

#include <qfileinfo.h>
#include <VolumeFileTypes/VolumeFileFactory.h>
#include <VolumeFileTypes/RawIVFileImpl.h>
#include <VolumeFileTypes/RawVFileImpl.h>
#include <VolumeFileTypes/MrcFileImpl.h>
#include <VolumeFileTypes/PifFileImpl.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeFileFactory VolumeFileFactory::ms_MainFactory;

VolumeFileFactory::VolumeFileFactory()
{
	// add each volume type to the maps
	addVolumeFileType(new RawVFileImpl);
	addVolumeFileType(new RawIVFileImpl);
	addVolumeFileType(new MrcFileImpl);
	addVolumeFileType(new PifFileImpl);
}

VolumeFileFactory::~VolumeFileFactory()
{

}

VolumeFile* VolumeFileFactory::getLoader(const QString& fileName)
{
	QFileInfo fileInfo(fileName);
	QString extension = fileInfo.extension(false);
	if (extension.isEmpty() || !m_ExtensionMap.contains(extension)) {
		// test every file type to find the correct one
		return tryAll(fileName);
	}
	else {
		// try to load the file with the correct file type
		VolumeFile* volumeFile = m_ExtensionMap[extension];
		if (!volumeFile->checkType(fileName)) { // failed, try the other loaders
			return tryAll(fileName);
		}
		else {
			// success
			return volumeFile->getNewVolumeFileLoader();
		}
	}
}

VolumeFile* VolumeFileFactory::getLoaderForExtension(const QString& extension)
{
	if (extension.isEmpty() || !m_ExtensionMap.contains(extension)) {
		// fail without a proper extension
		return NULL;
	}
	else {
		// return a loader for the file type
		VolumeFile* volumeFile = m_ExtensionMap[extension];
		return volumeFile->getNewVolumeFileLoader();
	}
}

QString VolumeFileFactory::getFilterString()
{
	QString string("All Volume Files ");
	string.append(getAllExtensions());
	//bool first = true;
	// iterate through each loader and combine all the filters
	QMap<QString, VolumeFile*>::Iterator it;
	for (it = m_FilterMap.begin(); it!=m_FilterMap.end(); ++it) {
		//if (first) {
		//	first = false;
		//	string = it.key();
		//}
		//else 
			string.append(";;" + it.key());
	}
	string.append(";;All Files (*.*)");
	return string;
}

QString VolumeFileFactory::getAllExtensions()
{
	QString string("(");
	bool first = true;
	// iterate through each loader and combine all the filters
	QMap<QString, VolumeFile*>::Iterator it;
	for (it = m_ExtensionMap.begin(); it!=m_ExtensionMap.end(); ++it) {
		if (first) {
			first = false;
			string.append("*." + it.key());
		}
		else 
			string.append(" *." + it.key());
	}
	string.append(")");

	return string;
}

VolumeFile* VolumeFileFactory::tryAll(const QString& fileName)
{
	// iterate through each loader and call checkType to determine
	// which loader can load the file
	QMap<QString, VolumeFile*>::Iterator it;
	for (it = m_ExtensionMap.begin(); it!=m_ExtensionMap.end(); ++it) {
		if (it.data()->checkType(fileName)) { // found it
			return it.data()->getNewVolumeFileLoader();
		}
	}

	// didnt find the right loader
	return 0;
}

void VolumeFileFactory::addVolumeFileType(VolumeFile* type)
{
	m_ExtensionMap[type->getExtension()] = type;
	m_FilterMap[type->getFilter()] = type;

	m_VolumeFileFactoryHandles.push_back(boost::shared_ptr<VolumeFile>(type));
}

