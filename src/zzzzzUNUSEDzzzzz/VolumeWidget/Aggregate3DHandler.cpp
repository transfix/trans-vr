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

// Aggregate3DHandler.cpp: implementation of the Aggregate3DHandler class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/Aggregate3DHandler.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Aggregate3DHandler::Aggregate3DHandler() :
m_CurrentCapture(0)
{

}

Aggregate3DHandler::Aggregate3DHandler(const Aggregate3DHandler& copy) :
m_Mouse3DHandlers(copy.m_Mouse3DHandlers), m_CurrentCapture(0)
{

}

Aggregate3DHandler::~Aggregate3DHandler()
{

}

Mouse3DHandler* Aggregate3DHandler::clone() const
{
	return new Aggregate3DHandler(*this);
}

int Aggregate3DHandler::add(Mouse3DHandler* handler)
{
	return m_Mouse3DHandlers.add(handler);
}

Mouse3DHandler* Aggregate3DHandler::remove( unsigned int index )
{
	return m_Mouse3DHandlers.remove(index);
}

bool Aggregate3DHandler::mousePress3DEvent(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	float closest;
	Mouse3DHandler* closestHandler;
	bool closestInitialized = false;
	unsigned int i;
	for (i=0; i<m_Mouse3DHandlers.getNumberOfObjects(); i++) {
		float current = m_Mouse3DHandlers.getIth(i)->getNearestClickDistance(simpleOpenGLWidget, e);
		if (current > 0.0 && (!closestInitialized || current<closest)) {
			closestHandler = m_Mouse3DHandlers.getIth(i);
			closest = current;
			closestInitialized = true;
		}
	}

	if (closestInitialized) {
		bool ret = closestHandler->mousePress3DEvent(simpleOpenGLWidget, e);
		if (ret) {
			m_CurrentCapture = closestHandler;
		}
		return ret;
	}
	else {
		return false;
	}
}

bool Aggregate3DHandler::mouseRelease3DEvent(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	if (m_CurrentCapture) {
		bool ret = m_CurrentCapture->mouseRelease3DEvent(simpleOpenGLWidget, e);
		m_CurrentCapture = 0;
		return ret;
	}
	else {
		return false;
	}
}

bool Aggregate3DHandler::mouseDoubleClick3DEvent(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	float closest;
	Mouse3DHandler* closestHandler;
	bool closestInitialized = false;
	unsigned int i;
	for (i=0; i<m_Mouse3DHandlers.getNumberOfObjects(); i++) {
		float current = m_Mouse3DHandlers.getIth(i)->getNearestClickDistance(simpleOpenGLWidget, e);
		if (current > 0.0 && (!closestInitialized || current<closest)) {
			closestHandler = m_Mouse3DHandlers.getIth(i);
			closest = current;
			closestInitialized = true;
		}
	}

	if (closestInitialized) {
		bool ret = closestHandler->mouseDoubleClick3DEvent(simpleOpenGLWidget, e);
		if (ret) {
			m_CurrentCapture = closestHandler;
		}
		return ret;
	}
	else {
		return false;
	}
}

bool Aggregate3DHandler::mouseMove3DEvent(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	bool ret;
	if (m_CurrentCapture) {
		ret = m_CurrentCapture->mouseMove3DEvent(simpleOpenGLWidget, e);
	}
	else {
		ret = false;
	}
	return ret;
}

float Aggregate3DHandler::getNearestClickDistance(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	float closest;
	bool closestInitialized = false;
	unsigned int i;
	for (i=0; i<m_Mouse3DHandlers.getNumberOfObjects(); i++) {
		float current = m_Mouse3DHandlers.getIth(i)->getNearestClickDistance(simpleOpenGLWidget, e);
		if (current > 0.0 && (!closestInitialized || current<closest)) {
			closest = current;
			closestInitialized = true;
		}
	}
	if (closestInitialized) {
		return closest;
	}
	else {
		return -1.0;
	}
}

