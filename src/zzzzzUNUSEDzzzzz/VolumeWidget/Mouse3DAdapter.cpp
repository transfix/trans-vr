/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
             Vinay Siddavanahalli <skvinay@cs.utexas.edu>
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

// Mouse3DAdapter.cpp: implementation of the Mouse3DAdapter class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/Mouse3DAdapter.h>
#include <VolumeWidget/MouseEvent3DPrivate.h>
#include <VolumeWidget/Mouse3DHandler.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Mouse3DAdapter::Mouse3DAdapter(Mouse3DHandler* mouse3DHandler) :
m_Mouse3DHandler(mouse3DHandler)
{

}

Mouse3DAdapter::~Mouse3DAdapter()
{

}

MouseHandler* Mouse3DAdapter::clone() const
{
	return new Mouse3DAdapter(*this);
}

bool Mouse3DAdapter::mousePressEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	MouseEvent3DPrivate event(simpleOpenGLWidget, *e);
	return m_Mouse3DHandler->mousePress3DEvent(simpleOpenGLWidget,&event);
}

bool Mouse3DAdapter::mouseReleaseEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	MouseEvent3DPrivate event(simpleOpenGLWidget, *e);
	return m_Mouse3DHandler->mouseRelease3DEvent(simpleOpenGLWidget,&event);
}

bool Mouse3DAdapter::mouseDoubleClickEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	MouseEvent3DPrivate event(simpleOpenGLWidget, *e);
	return m_Mouse3DHandler->mouseDoubleClick3DEvent(simpleOpenGLWidget,&event);
}

bool Mouse3DAdapter::mouseMoveEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	MouseEvent3DPrivate event(simpleOpenGLWidget, *e);
	return m_Mouse3DHandler->mouseMove3DEvent(simpleOpenGLWidget,&event);
}

bool Mouse3DAdapter::wheelEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QWheelEvent* e)
{
	// do nothing
	return false;
}

