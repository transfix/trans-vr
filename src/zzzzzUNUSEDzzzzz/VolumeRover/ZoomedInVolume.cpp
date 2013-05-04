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

/* $Id: ZoomedInVolume.cpp 1528 2010-03-12 22:28:08Z transfix $ */

// ZoomedInVolume.cpp: implementation of the ZoomedInVolume class.
//
//////////////////////////////////////////////////////////////////////

#include <boost/scoped_ptr.hpp>
#include <VolumeRover/ZoomedInVolume.h>
#include <VolumeWidget/PanInteractor.h>
#include <VolumeWidget/ZoomInteractor.h>
#include <VolumeWidget/TrackballRotateInteractor.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ZoomedInVolume::ZoomedInVolume(Extents* extent, RenderableArray* geometryArray) :
RoverRenderable(extent, geometryArray)
{
	setOpaqueRenderable(&m_WireCube);
}

ZoomedInVolume::~ZoomedInVolume()
{

}

void ZoomedInVolume::addToSimpleOpenGLWidget(SimpleOpenGLWidget& simpleOpenGLWidget, QObject* receiver, const char* member)
{
  boost::scoped_ptr<PanInteractor> leftButtonHandler(new PanInteractor);
  boost::scoped_ptr<ZoomInteractor> middleButtonHandler(new ZoomInteractor);
  boost::scoped_ptr<ZoomInteractor> wheelHandler(new ZoomInteractor);
  boost::scoped_ptr<TrackballRotateInteractor> rightButtonHandler(new TrackballRotateInteractor);

	if(simpleOpenGLWidget.initForContext(this) == false)
		qDebug("Could not initialize a renderer for this OpenGL context!");
	simpleOpenGLWidget.setMainRenderable(this);
	simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, leftButtonHandler.get());

	MouseHandler* handler;
	simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::MiddleButtonHandler, middleButtonHandler.get());
	simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::WheelHandler, wheelHandler.get());
	handler = simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::RightButtonHandler, rightButtonHandler.get());
	QObject::connect(handler, SIGNAL(ViewChanged()), receiver, member);
}

void ZoomedInVolume::toggleWireCubeDrawing(bool state)
{
	if (state) {
		setOpaqueRenderable(&m_WireCube);
	}
	else {
		setOpaqueRenderable(NULL);
	}
}

void ZoomedInVolume::setWireCubeColor(float r, float g, float b)
{
	m_WireCube.setColor(r,g,b);
}

void ZoomedInVolume::setAspectRatio(double x, double y, double z)
{
	m_VolumeRenderer->setAspectRatio(x,y,z);
	m_WireCube.setAspectRatio(x,y,z);
}

