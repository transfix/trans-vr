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

// MouseHandler.h: interface for the MouseHandler class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MOUSEHANDLER_H__756501C7_4A0C_4CAD_B69C_92A0C80B3EBF__INCLUDED_)
#define AFX_MOUSEHANDLER_H__756501C7_4A0C_4CAD_B69C_92A0C80B3EBF__INCLUDED_

#include <qobject.h>
class QMouseEvent;
class QWheelEvent;
class SimpleOpenGLWidget;

///\class MouseHandler MouseHandler.h
///\author Anthony Thane
///\brief MouseHandler is an abstract base class for objects that handle mouse
/// events for SimpleOpenGLWidget (and others, possibly).
class MouseHandler : public QObject
{
public:
	MouseHandler();
	virtual ~MouseHandler();
	MouseHandler(const MouseHandler& copy);

///\fn virtual MouseHandler* clone() const
///\brief Creates a clone of the object
///\return A pointer to a MouseHandler instance
	virtual MouseHandler* clone() const = 0;

///\fn virtual bool mousePressEvent(SimpleOpenGLWidget* widget, QMouseEvent* e)
///\brief Handles mouse button press events
///\param widget The SimpleOpenGLWidget that received the mouse event
///\param e The QMouseEvent received by widget
	virtual bool mousePressEvent(SimpleOpenGLWidget* widget, QMouseEvent* e) = 0;
///\fn virtual bool mouseReleaseEvent(SimpleOpenGLWidget* widget, QMouseEvent* e)
///\brief Handles mouse button release events
///\param widget The SimpleOpenGLWidget that received the mouse event
///\param e The QMouseEvent received by widget
	virtual bool mouseReleaseEvent(SimpleOpenGLWidget* widget, QMouseEvent* e) = 0;
///\fn virtual bool mouseDoubleClickEvent(SimpleOpenGLWidget* widget, QMouseEvent* e)
///\brief Handles mouse button double click events
///\param widget The SimpleOpenGLWidget that received the mouse event
///\param e The QMouseEvent received by widget
	virtual bool mouseDoubleClickEvent(SimpleOpenGLWidget* widget, QMouseEvent* e) = 0;
///\fn virtual bool mouseMoveEvent(SimpleOpenGLWidget* widget, QMouseEvent* e) 
///\brief Handles mouse move events
///\param widget The SimpleOpenGLWidget that received the mouse event
///\param e The QMouseEvent received by widget
	virtual bool mouseMoveEvent(SimpleOpenGLWidget* widget, QMouseEvent* e) = 0;
///\fn virtual bool wheelEvent(SimpleOpenGLWidget* widget, QWheelEvent* e) 
///\brief Handles mouse wheel events
///\param widget The SimpleOpenGLWidget that received the mouse event
///\param e The QMouseEvent received by widget
	virtual bool wheelEvent(SimpleOpenGLWidget* widget, QWheelEvent* e) = 0;

};

#endif // !defined(AFX_MOUSEHANDLER_H__756501C7_4A0C_4CAD_B69C_92A0C80B3EBF__INCLUDED_)
