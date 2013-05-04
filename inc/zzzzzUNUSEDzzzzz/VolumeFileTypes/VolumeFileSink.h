/*
  Copyright 2002-2005 The University of Texas at Austin
  
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

// VolumeFileSink.h: declaration of the VolumeFileSink class.
//
//////////////////////////////////////////////////////////////////////

#ifndef VOLUME_FILE_SINK_H
#define VOLUME_FILE_SINK_H

#include <VolumeFileTypes/VolumeSink.h>
#include <VolumeFileTypes/VolumeFile.h>
#include <VolumeFileTypes/VolumeFileFactory.h>

#include <qstring.h>

///\class VolumeFileSink VolumeFileSink.h
///\brief The VolumeFileSink class is a VolumeSink instance that uses
/// a VolumeFile object to write volume data to disk.
///\author John Wiggins
class VolumeFileSink : public VolumeSink
{
public:

	VolumeFileSink(const QString& fileName, const QString& extension);
	virtual ~VolumeFileSink();

	virtual bool writeHeader();
	virtual bool writeRawData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax, uint variable, uint timeStep);

	virtual void setNumVariables(unsigned int num);
	virtual void setNumTimeSteps(unsigned int num);

	virtual void setVariableName(unsigned int variable, QString name);
	virtual void setVariableType(unsigned int variable, VolumeFile::Type type);

protected:
///\fn virtual void setDefaults()
///\brief This function intializes the member variables.
	virtual void setDefaults();
	unsigned int m_NumVariables, m_NumTimeSteps;

	// the volume file we are writing to
	VolumeFile* m_VolumeFile;
};

#endif

