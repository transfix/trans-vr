/*
  Copyright 2002-2004 The University of Texas at Austin
  
	Authors: John Wiggins <prok@ices.utexas.edu>
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

// VolumeTranscriber.cpp: definition of the VolumeTranscriber class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/VolumeTranscriber.h>
#include <VolumeFileTypes/VolumeSource.h>
#include <VolumeFileTypes/VolumeSink.h>
#include <qapplication.h>
#include <q3progressdialog.h>

#include <Filters/OOCBilateralFilter.h>
	
OutOfCoreFilter::Type maptype(unsigned int t)
{
	if (t == VolumeFile::Char)
		return OutOfCoreFilter::U_CHAR;
	else if (t == VolumeFile::Short)
		return OutOfCoreFilter::U_SHORT;
	else if (t == VolumeFile::Long)
		return OutOfCoreFilter::U_INT;
	else if (t == VolumeFile::Float)
		return OutOfCoreFilter::FLOAT;
	else if (t == VolumeFile::Double)
		return OutOfCoreFilter::DOUBLE;
	else // we should not get here
		return OutOfCoreFilter::U_CHAR;
}

VolumeTranscriber::VolumeTranscriber(VolumeSource* source, VolumeSink* sink)
{
	m_Source = source;
	m_Sink = sink;
}

VolumeTranscriber::~VolumeTranscriber()
{
	// nothing to clean up. Caller has ownership of passed pointers.
}

bool VolumeTranscriber::go(QWidget *parent, uint minX, uint minY, uint minZ,
													 uint minT, uint maxX, uint maxY, uint maxZ,
													 uint maxT)
{
	bool ret = false, userCancelled=false;
	char *buffer;
	uint numVars, bigVar=0;
	uint width = maxX-minX+1;
	uint height = maxY-minY+1;
	uint depth = maxZ-minZ+1;
	uint timespan = maxT-minT+1;
	uint v,t,z;
	double spanX,spanY,spanZ,spanT;
	unsigned int sizes[] = {1, 2, 4, 4, 8};
	// see VolumeFile.h if you need an explanation for the line above
	Q3ProgressDialog progressDialog("Performing subvolume extraction.", 
		QObject::tr("Cancel"), 100, parent, "Subvolume Extraction", true);
	progressDialog.setProgress(0);

	// fail if things aren't right
	if (!m_Source || !m_Sink)
		return ret;

	// fill in data for the sink
	m_Sink->setNumVariables(m_Source->getNumVars());
	m_Sink->setNumTimeSteps(timespan);
	//
	for (v=0; v < m_Source->getNumVars(); v++) {
		m_Sink->setVariableName(v, m_Source->getVariableName(v));
		m_Sink->setVariableType(v, (VolumeFile::Type)m_Source->getVariableType(v));
	}
	//
	m_Sink->setDimX(width);
	m_Sink->setDimY(height);
	m_Sink->setDimZ(depth);
	//
	spanX = (m_Source->getMaxX()-m_Source->getMinX())/(m_Source->getDimX()-1);
	spanY = (m_Source->getMaxY()-m_Source->getMinY())/(m_Source->getDimY()-1);
	spanZ = (m_Source->getMaxZ()-m_Source->getMinZ())/(m_Source->getDimZ()-1);
	spanT = (m_Source->getMaxT()-m_Source->getMinT())/m_Source->getNumTimeSteps();
	//
	m_Sink->setMinX(m_Source->getMinX() + minX*spanX);
	m_Sink->setMinY(m_Source->getMinY() + minY*spanY);
	m_Sink->setMinZ(m_Source->getMinZ() + minZ*spanZ);
	m_Sink->setMinT(m_Source->getMinT() + minT*spanT);
	//
	m_Sink->setMaxX(m_Source->getMinX() + maxX*spanX);
	m_Sink->setMaxY(m_Source->getMinY() + maxY*spanY);
	m_Sink->setMaxZ(m_Source->getMinZ() + maxZ*spanZ);
	m_Sink->setMaxT(m_Source->getMinT() + maxT*spanT);
	
	// start out by writing the header
	if (!m_Sink->writeHeader())
		return ret;

	// how many variables?
	numVars = m_Source->getNumVars();
	
	//virtual bool writeRawData(char* data, uint xMin, uint yMin, uint zMin,
	//	uint xMax, uint yMax, uint zMax, uint variable, uint timeStep);
	//virtual bool readRawData(char* data, uint xMin, uint yMin, uint zMin,
	//	uint xMax, uint yMax, uint zMax, uint variable, uint timeStep);
	
	// create a buffer large enough to hold the largest slice
	for (v=0; v < numVars; v++)
		if (sizes[m_Source->getVariableType(v)] > bigVar)
			bigVar = sizes[m_Source->getVariableType(v)];
	
	// allocate
	buffer = new char [width*height*bigVar];

	// set up progress bar
	progressDialog.setTotalSteps(depth*numVars*timespan);

	for (v = 0; v < numVars && !userCancelled; v++) {
		for (t = 0; t < timespan && !userCancelled; t++) {
			for (z = 0; z < depth && !userCancelled; z++) {
				// read and write one slice at a time
				//qDebug("read from %d,%d,%d to %d,%d,%d", minX,minY,minZ+z, maxX,maxY,minZ+z);
				ret = m_Source->readRawData(buffer, minX,minY,minZ+z, maxX,maxY,minZ+z, v,t);
				//if (!ret) qDebug("readRawData() returned false;");

				if (ret) {
					ret = m_Sink->writeRawData(buffer, 0,0,z, width-1, height-1, z, v, t);
					//if (!ret) qDebug("writeRawData() returned false;");
				}

				// update progress
				progressDialog.setProgress(v*timespan*depth + t*depth + z);
				// let the app process gui events
				qApp->processEvents();
				// check for cancellation
				if (progressDialog.wasCancelled()) {
					// make note
					userCancelled = true;
				}
			}
		}
	}

	// the operation did not run to completion
	if (userCancelled) ret = false;

	delete [] buffer;

	return ret;
}

bool VolumeTranscriber::goFiltered(QWidget* parent,
									uint minX, uint minY, uint minZ, uint minT,
									uint maxX, uint maxY, uint maxZ, uint maxT)
{
	bool ret = false, userCancelled = false;
	char **buffers;
	uint numVars, bigVar=0;
	uint width = maxX-minX+1;
	uint height = maxY-minY+1;
	uint depth = maxZ-minZ+1;
	uint timespan = maxT-minT+1;
	uint v,t,z;
	double spanX,spanY,spanZ,spanT;
	unsigned int sizes[] = {1, 2, 4, 4, 8}, currentSlice=0, numSlices=0;
	// see VolumeFile.h if you need an explanation for the line above
	Q3ProgressDialog progressDialog("Performing filtered subvolume extraction.", 
		QObject::tr("Cancel"), 100, parent, "Filtered Subvolume Extraction", true);
	progressDialog.setProgress(0);
	OOCBilateralFilter filter;

	// fail if things aren't right
	if (!m_Source || !m_Sink)
		return ret;

	// we need this many slices for the read ahead
	numSlices = filter.getNumCacheSlices();
	buffers = new char *[numSlices];

	// fill in data for the sink
	m_Sink->setNumVariables(m_Source->getNumVars());
	m_Sink->setNumTimeSteps(timespan);
	//
	for (v=0; v < m_Source->getNumVars(); v++) {
		m_Sink->setVariableName(v, m_Source->getVariableName(v));
		m_Sink->setVariableType(v, (VolumeFile::Type)m_Source->getVariableType(v));
	}
	//
	m_Sink->setDimX(width);
	m_Sink->setDimY(height);
	m_Sink->setDimZ(depth);
	//
	spanX = (m_Source->getMaxX()-m_Source->getMinX())/(m_Source->getDimX()-1);
	spanY = (m_Source->getMaxY()-m_Source->getMinY())/(m_Source->getDimY()-1);
	spanZ = (m_Source->getMaxZ()-m_Source->getMinZ())/(m_Source->getDimZ()-1);
	spanT = (m_Source->getMaxT()-m_Source->getMinT())/m_Source->getNumTimeSteps();
	//
	m_Sink->setMinX(m_Source->getMinX() + minX*spanX);
	m_Sink->setMinY(m_Source->getMinY() + minY*spanY);
	m_Sink->setMinZ(m_Source->getMinZ() + minZ*spanZ);
	m_Sink->setMinT(m_Source->getMinT() + minT*spanT);
	//
	m_Sink->setMaxX(m_Source->getMinX() + maxX*spanX);
	m_Sink->setMaxY(m_Source->getMinY() + maxY*spanY);
	m_Sink->setMaxZ(m_Source->getMinZ() + maxZ*spanZ);
	m_Sink->setMaxT(m_Source->getMinT() + maxT*spanT);
	
	// start out by writing the header
	if (!m_Sink->writeHeader())
		return ret;

	// how many variables?
	numVars = m_Source->getNumVars();
	
	//virtual bool writeRawData(char* data, uint xMin, uint yMin, uint zMin,
	//	uint xMax, uint yMax, uint zMax, uint variable, uint timeStep);
	//virtual bool readRawData(char* data, uint xMin, uint yMin, uint zMin,
	//	uint xMax, uint yMax, uint zMax, uint variable, uint timeStep);
	
	// create a buffer large enough to hold the largest slice
	for (v=0; v < numVars; v++)
		if (sizes[m_Source->getVariableType(v)] > bigVar)
			bigVar = sizes[m_Source->getVariableType(v)];
	
	// allocate
	for (v=0; v < numSlices; v++)
		buffers[v] = new char [width*height*bigVar];

	// set up progress bar
	progressDialog.setTotalSteps(depth*numVars*timespan);

	for (v = 0; v < numVars && !userCancelled; v++) {
		// set the data type for this slice
		filter.setDataType(maptype(m_Source->getVariableType(v)));
		for (t = 0; t < timespan && !userCancelled; t++) {
			// reset the filter
			filter.initFilter(width, height, depth, m_Source->getFunctionMinimum(v,t), m_Source->getFunctionMaximum(v,t));

			// init
			//oocFilter.initFilter(dim[0],dim[1],dim[2]);
			// pre-init the cache
			//for (i=0; i < oocFilter.getNumCacheSlices(); i++)
			//	oocFilter.addSlice((void *)(bPtr + i*dim[0]*dim[1]));
			// filter
			//for (i=0; i < dim[2]; i++) {
			//	oocFilter.filterSlice((void *)(bPtr + i*dim[0]*dim[1]));
			//	if (i+oocFilter.getNumCacheSlices() < dim[2])
			//		oocFilter.addSlice((void *)(bPtr + (i+oocFilter.getNumCacheSlices())*dim[0]*dim[1]));
			//}

			// pre-cache slices
			for (z = 0; z < numSlices; z++) {
				m_Source->readRawData(buffers[z], minX,minY,minZ+z, maxX,maxY,minZ+z, v,t);
				filter.addSlice(buffers[z]);
			}
			// filter and write slices
			for (z = 0; z < depth && !userCancelled; z++) {
				// filter the current slice
				ret = filter.filterSlice(buffers[z%numSlices]);
				
				// write the filtered slice
				if (ret) {
					ret = m_Sink->writeRawData(buffers[z%numSlices], 0,0,z, width-1, height-1, z, v, t);
				}

				// maybe read a slice
				if (z+numSlices < depth) {
					ret = m_Source->readRawData(buffers[z%numSlices], minX,minY,minZ+z+numSlices, maxX,maxY,minZ+z+numSlices, v,t);
					// add the slice to the cache
					filter.addSlice(buffers[z%numSlices]);
				}

				// update progress
				progressDialog.setProgress(v*timespan*depth + t*depth + z);
				// let the app process gui events
				qApp->processEvents();
				if (progressDialog.wasCancelled()) {
					// make note
					userCancelled = true;
				}
			}
		}
	}

	for (v=0; v < numSlices; v++)
		delete [] buffers[v];
	delete [] buffers;

	// the operation did not run to completion
	if (userCancelled) ret = false;

	return ret;
}

