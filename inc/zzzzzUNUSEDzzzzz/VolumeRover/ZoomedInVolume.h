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

// ZoomedInVolume.h: interface for the ZoomedInVolume class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ZOOMEDINVOLUME_H__999F0280_D856_48CD_A4B5_3A0D93FFE9DC__INCLUDED_)
#define AFX_ZOOMEDINVOLUME_H__999F0280_D856_48CD_A4B5_3A0D93FFE9DC__INCLUDED_

#include <VolumeRover/RoverRenderable.h>
#include <VolumeWidget/WireCubeRenderable.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>

///\class ZoomedInVolume ZoomedInVolume.h
///\brief ZoomedInVolume is the RoverRenderable instance that controls the
///	left (sub-volume) view of Volume Rover.
///\author Anthony Thane
class ZoomedInVolume : public RoverRenderable  
{
public:
	ZoomedInVolume(Extents* extent, RenderableArray* geometryArray);
	virtual ~ZoomedInVolume();

	void addToSimpleOpenGLWidget(SimpleOpenGLWidget& simpleOpenGLWidget, QObject* receiver, const char* member);
	
	void toggleWireCubeDrawing(bool state);
	void setWireCubeColor(float r, float g, float b);

	virtual void setAspectRatio(double x, double y, double z);

protected:
	WireCubeRenderable m_WireCube;
};

#endif // !defined(AFX_ZOOMEDINVOLUME_H__999F0280_D856_48CD_A4B5_3A0D93FFE9DC__INCLUDED_)
