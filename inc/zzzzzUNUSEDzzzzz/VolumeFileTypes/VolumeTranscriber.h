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

// VolumeTranscriber.h: declaration of the VolumeTranscriber class.
//
//////////////////////////////////////////////////////////////////////

#ifndef VOLUME_TRANSCRIBER_H
#define VOLUME_TRANSCRIBER_H

class VolumeSource;
class VolumeSink;
class QWidget;

typedef unsigned int uint;

///\class VolumeTranscriber VolumeTranscriber.h
///\author John Wiggins
///\brief VolumeTranscriber reads from a VolumeSource and writes to a
///	VolumeSink. It is currently the basis of the "Save Subvolume" feature in
///	Volume Rover. It was designed to be usable for more exciting things than
///	than writing out files, but at the very least it does that.
class VolumeTranscriber
{
public:

///\fn VolumeTranscriber(VolumeSource* source, VolumeSink* sink)
///\brief The constructor
///\param source A VolumeSource to read data from
///\param sink A VolumeSink to write data to
	VolumeTranscriber(VolumeSource* source, VolumeSink* sink);
	~VolumeTranscriber();

///\fn bool go(QWidget* parent, uint minX, uint minY, uint minZ, uint minT, uint maxX, uint maxY, uint maxZ, uint maxT)
///\brief This is the main workhorse of the class. Basically, you pass it a 4D
///	bounding box and it reads from that bounding in the source volume. All units
///	are samples/voxels.
///\warning Watch out for off-by-one errors when calling this function. I'm
///	not sure that it doesn't have a bug. The Save Subvolume feature works, but
///	that doesn't mean writing a new VolumeSink instance won't be headache free.
///\param parent This is a QWidget which is needed to construct a QProgressDialog.
///\param minX The minimum X coord of the bounding box.
///\param minY The minimum Y coord of the bounding box.
///\param minZ The minimum Z coord of the bounding box.
///\param minT The minimum T coord of the bounding box.
///\param maxX The maximum X coord of the bounding box.
///\param maxY The maximum Y coord of the bounding box.
///\param maxZ The maximum Z coord of the bounding box.
///\param maxT The maximum T coord of the bounding box.
///\return A boolean indicating sucess or failure
	bool go(QWidget* parent, uint minX, uint minY, uint minZ, uint minT,
									uint maxX, uint maxY, uint maxZ, uint maxT);

///\fn bool goFiltered(QWidget* parent, uint minX, uint minY, uint minZ, uint minT, uint maxX, uint maxY, uint maxZ, uint maxT)
///\brief This function is identical to go() except that it processes the data
///	it reads with an OOCBilateralFilter before writing.
	bool goFiltered(QWidget* parent, uint minX, uint minY, uint minZ, uint minT,
									uint maxX, uint maxY, uint maxZ, uint maxT);

private:

	VolumeSource* m_Source;
	VolumeSink* m_Sink;

};

#endif

