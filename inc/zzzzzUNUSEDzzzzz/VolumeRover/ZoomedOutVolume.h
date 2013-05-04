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

// ZoomedOutVolume.h: interface for the ZoomedOutVolume class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ZOOMEDOUTVOLUME_H__BE6441C5_A261_4A5B_BA51_B4AEFCC8756B__INCLUDED_)
#define AFX_ZOOMEDOUTVOLUME_H__BE6441C5_A261_4A5B_BA51_B4AEFCC8756B__INCLUDED_

#include <VolumeRover/RoverRenderable.h>
#include <VolumeRover/Rover3DWidget.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/Vector.h>

class MouseHandler;
#ifdef VOLUMEGRIDROVER
class VolumeGridRover;
#endif
///\class ZoomedOutVolume ZoomedOutVolume.h
///\brief ZoomedOutVolume is the RoverRenderable instance that controls the
///	right ("Explorer") view of Volume Rover.
///\author Anthony Thane
class ZoomedOutVolume : public RoverRenderable  
{
public:
	ZoomedOutVolume(Extents* extent, RenderableArray* geometryArray);
	virtual ~ZoomedOutVolume();

	void addToSimpleOpenGLWidget(SimpleOpenGLWidget& simpleOpenGLWidget, QObject* receiver, const char* member);

	virtual void setAspectRatio(double x, double y, double z);

	virtual void connectRoverSignals(QObject* receiver, const char* exploring, const char* released);

	void toggleWireCubeDrawing(bool state);
	void setRover3DWidgetColor(float r, float g, float b);

	Extents getSubVolume() const;
	Extents getBoundary() const;

	Vector getPreviewOrigin() const;
	
	virtual bool render();
	
 protected:
	Rover3DWidget m_Rover3DWidget;
};

#endif // !defined(AFX_ZOOMEDOUTVOLUME_H__BE6441C5_A261_4A5B_BA51_B4AEFCC8756B__INCLUDED_)
