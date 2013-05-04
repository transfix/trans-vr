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

// ViewInteractor.cpp: implementation of the ViewInteractor class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/ViewInteractor.h>
#include <VolumeWidget/View.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ViewInteractor::ViewInteractor()
{
	m_ButtonDown= false;
}

ViewInteractor::~ViewInteractor()
{

}

bool ViewInteractor::mousePressEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	startDrag(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool ViewInteractor::mouseReleaseEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	endDrag(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool ViewInteractor::mouseDoubleClickEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	// do nothing for now
	return false;
}

bool ViewInteractor::mouseMoveEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	mouseMove(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool ViewInteractor::wheelEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QWheelEvent* e)
{
	// simulate a drag in y
	doInteraction(simpleOpenGLWidget, 0, 0, 0, e->delta()/3);
	simpleOpenGLWidget->updateGL();
	return true;
}

void ViewInteractor::startDrag(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
{
	// y window coordinates are opposite of opengl coordinates
	// invert
	y = invertY(y, simpleOpenGLWidget->getHeight());

	m_ButtonDown = true;
	m_OldX = x;
	m_OldY = y;
}

void ViewInteractor::mouseMove(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
{
	// y window coordinates are opposite of opengl coordinates
	// invert
	y = invertY(y, simpleOpenGLWidget->getHeight());

	if (m_ButtonDown) {
		doInteraction(simpleOpenGLWidget, m_OldX, m_OldY, x, y);
	}
	m_OldX = x;
	m_OldY = y;
}

void ViewInteractor::endDrag(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
{
	if (m_ButtonDown) {
		emit ViewChanged();
	}
	m_ButtonDown = false;
}

void ViewInteractor::setDefaults()
{
	m_ButtonDown = false;
	m_OldX = 0;
	m_OldY = 0;
}

int ViewInteractor::invertY(int y, int height) const
{
	return height-y-1;
}


