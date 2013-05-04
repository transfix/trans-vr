/*
  Copyright 2002-2005 The University of Texas at Austin
  
	Authors: John Wiggins 2005 <prok@ices.utexas.edu>
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

//////////////////////////////////////////////////////////////////////
//
// VolumeFileSink.cpp: implementation of the VolumeFileSink class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/VolumeFileSink.h>
#include <qfileinfo.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeFileSink::VolumeFileSink(const QString& fileName, const QString& extension)
{
	QFileInfo fileInfo(fileName);

	//qDebug("FileSink created for: %s", fileInfo.absFilePath().ascii());
	// init other variables to default values
	VolumeFileSink::setDefaults();
	
	// open the file
	//m_VolumeFile = VolumeFileFactory::ms_MainFactory.getLoaderForExtension(fileInfo.absFilePath());
	m_VolumeFile = VolumeFileFactory::ms_MainFactory.getLoaderForExtension(extension);

	if (m_VolumeFile 
		&& m_VolumeFile->attachToFile(fileInfo.absFilePath(), VolumeFile::Write)) {
		// success!
	}
	else { // failure!
		delete m_VolumeFile;
		m_VolumeFile = NULL;
	}
}

VolumeFileSink::~VolumeFileSink()
{
	if (m_VolumeFile)
		delete m_VolumeFile;
}

bool VolumeFileSink::writeHeader()
{
	if (m_VolumeFile) {
		// pass VolumeSink member vars to VolumeFile instance
		// (this is kinda kludgy...)
		m_VolumeFile->setDimX(m_DimX);
		m_VolumeFile->setDimY(m_DimY);
		m_VolumeFile->setDimZ(m_DimZ);
		m_VolumeFile->setMinX(m_MinX);
		m_VolumeFile->setMinY(m_MinY);
		m_VolumeFile->setMinZ(m_MinZ);
		m_VolumeFile->setMinT(m_MinT);
		m_VolumeFile->setMaxX(m_MaxX);
		m_VolumeFile->setMaxY(m_MaxY);
		m_VolumeFile->setMaxZ(m_MaxZ);
		m_VolumeFile->setMaxT(m_MaxT);

		// write it!
		return m_VolumeFile->writeHeader();
	}
	else return false;
}

bool VolumeFileSink::writeRawData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, uint variable, uint timeStep)
{
	uint width = xMax-xMin+1;
	uint height = yMax-yMin+1;
	uint depth = zMax-zMin+1;
	uint j,k;
	bool ret = false;
	
	if (!m_VolumeFile)
		return ret;

	if (width == m_VolumeFile->getDimX()) {
		// this is the fast case for writes.
		// because the width of the volume being written is the same as
		// the file's width, we can write in slice sized increments.
		// (bigger writes are better, for the most part)
		switch (m_VolumeFile->getVariableType(variable))
		{
			case VolumeFile::Char:
			{
				// loop through each slice
				for (k=0; k<depth; k++) {
					// write a slice	
					ret = m_VolumeFile->writeCharData(
									(unsigned char*)data+(k*width*height), 
									(zMin+k)*getDimX()*((Q_ULLONG)getDimY()) + yMin*getDimX(),
									width*height, variable, timeStep);
				}
				break;
			}
			case VolumeFile::Short:
			{
				// loop through each slice
				for (k=0; k<depth; k++) {
					// write a slice	
					ret = m_VolumeFile->writeShortData(
									(unsigned short*)data+(k*width*height), 
									(zMin+k)*getDimX()*((Q_ULLONG)getDimY()) + yMin*getDimX(),
									width*height, variable, timeStep);
				}
				break;
			}
			case VolumeFile::Long:
			{
				// loop through each slice
				for (k=0; k<depth; k++) {
					// write a slice	
					ret = m_VolumeFile->writeLongData(
									(unsigned int*)data+(k*width*height), 
									(zMin+k)*getDimX()*((Q_ULLONG)getDimY()) + yMin*getDimX(),
									width*height, variable, timeStep);
				}
				break;
			}
			case VolumeFile::Float:
			{
				// loop through each slice
				for (k=0; k<depth; k++) {
					// write a slice	
					ret = m_VolumeFile->writeFloatData(
									(float*)data+(k*width*height), 
									(zMin+k)*getDimX()*((Q_ULLONG)getDimY()) + yMin*getDimX(),
									width*height, variable, timeStep);
				}
				break;
			}
			case VolumeFile::Double:
			{
				// loop through each slice
				for (k=0; k<depth; k++) {
					// write a slice	
					ret = m_VolumeFile->writeDoubleData(
									(double*)data+(k*width*height), 
									(zMin+k)*getDimX()*((Q_ULLONG)getDimY()) + yMin*getDimX(),
									width*height, variable, timeStep);
				}
				break;
			}
			default:
				break;
		}
	}
	else {
		// this is the catch all version
		// it works just fine, but it calls writeData() way more than the previous
		// version.
		switch (m_VolumeFile->getVariableType(variable))
		{
			case VolumeFile::Char:
			{
				// loop through each slice
				for (k=0; k<depth; k++) {
					// loop through each line
					for (j=0; j<height; j++) {
						// write a line
						ret = m_VolumeFile->writeCharData(
										(unsigned char*)data+(k*width*height + j*width), 
										(zMin+k)*getDimX()*((Q_ULLONG)getDimY())+((yMin+j)*getDimX())+xMin,
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
						// write a line
						ret = m_VolumeFile->writeShortData(
										(unsigned short*)data+(k*width*height + j*width), 
										(zMin+k)*getDimX()*((Q_ULLONG)getDimY())+((yMin+j)*getDimX())+xMin,
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
						// write a line
						ret = m_VolumeFile->writeLongData(
										(unsigned int*)data+(k*width*height + j*width), 
										(zMin+k)*getDimX()*((Q_ULLONG)getDimY())+((yMin+j)*getDimX())+xMin,
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
						// write a line
						ret = m_VolumeFile->writeFloatData(
										(float*)data+(k*width*height + j*width), 
										(zMin+k)*getDimX()*((Q_ULLONG)getDimY())+((yMin+j)*getDimX())+xMin,
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
						// write a line
						ret = m_VolumeFile->writeDoubleData(
										(double*)data+(k*width*height + j*width), 
										(zMin+k)*getDimX()*((Q_ULLONG)getDimY())+((yMin+j)*getDimX())+xMin,
										width, variable, timeStep);
					}
				}
				break;
			}
			default:
				// shouldn't get here. ret remains unchanged.
				break;
		}
	}

	return ret;
}

void VolumeFileSink::setVariableName(unsigned int variable, QString name)
{
	// pass it to the VolumeFile instance
	if (m_VolumeFile)
		m_VolumeFile->setVariableName(variable, name);
}

void VolumeFileSink::setVariableType(unsigned int variable, VolumeFile::Type type)
{
	// pass it to the VolumeFile instance
	if (m_VolumeFile)
		m_VolumeFile->setVariableType(variable, type);
}

void VolumeFileSink::setNumVariables(unsigned int num)
{
	m_NumVariables = num;

	// pass it to the VolumeFile instance
	if (m_VolumeFile)
		m_VolumeFile->setNumVariables(num);
}

void VolumeFileSink::setNumTimeSteps(unsigned int num)
{
	m_NumTimeSteps = num;
	
	// pass it to the VolumeFile instance
	if (m_VolumeFile)
		m_VolumeFile->setNumTimeSteps(num);
}

void VolumeFileSink::setDefaults()
{
	VolumeSink::setDefaults();

	m_NumVariables = 0; m_NumTimeSteps = 0;
	m_VolumeFile = NULL;
}

