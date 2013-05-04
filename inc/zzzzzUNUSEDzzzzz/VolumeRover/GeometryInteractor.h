/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
             Vinay Siddavanahalli <skvinay@cs.utexas.edu>
             John Wiggins <prok@ices.utexas.edu>
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

// GeometryInteractor.h: interface for the GeometryInteractor class.
//
//////////////////////////////////////////////////////////////////////

#ifndef GEOMETRY_INTERACTOR_H
#define GEOMETRY_INTERACTOR_H

#include <VolumeWidget/MouseHandler.h>

#include <Qt>
#include <QMouseEvent>
//Added by qt3to4:
#include <QWheelEvent>

///\class GeometryInteractor GeometryInteractor.h
///\brief This MouseHandler instance is used to manipulate the
///	GeometryRenderer instance of the ZoomedOutVolume in NewVolumeMainWindow.
///	When activated, it overrides all other MouseHandler objects for the
///	SimpleOpenGLWidget that it is attached to.
///\author John Wiggins
class GeometryRenderer;

class GeometryInteractor : public MouseHandler
{
	Q_OBJECT
public:
///\fn GeometryInteractor::GeometryInteractor(GeometryRenderer *geom)
///\brief The class contstructor
///\param geom The GeometryRenderer instance that will be manipulated by this object.
	GeometryInteractor(GeometryRenderer *geom);
	virtual ~GeometryInteractor();
	
	virtual MouseHandler* clone() const;

	virtual bool mousePressEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool mouseReleaseEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool mouseDoubleClickEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool mouseMoveEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool wheelEvent(SimpleOpenGLWidget* widget, QWheelEvent* e);

signals:
///\fn void ViewChanged( )
///\brief Signals the end of a mouse interaction (ie- the button is released) 
	void ViewChanged( );

private:
///\fn void startDrag(SimpleOpenGLWidget* widget, int x, int y, ButtonState button)
///\brief Called at the beginning of a mouse interaction
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param x The X coordinate of the current mouse position
///\param y The Y coordinate of the current mouse position
///\param button The mouse button that is pressed
	void startDrag(SimpleOpenGLWidget* widget, int x, int y, Qt::ButtonState button);
///\fn void mouseMove(SimpleOpenGLWidget* widget, int x, int y)
///\brief Called between startDrag and endDrag whenever the mouse is moved
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param x The X coordinate of the current mouse position
///\param y The Y coordinate of the current mouse position
	void mouseMove(SimpleOpenGLWidget* widget, int x, int y);
///\fn void endDrag(SimpleOpenGLWidget* widget, int x, int y)
///\brief Called at the end of a mouse interaction
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param x The X coordinate of the current mouse position
///\param y The Y coordinate of the current mouse position
	void endDrag(SimpleOpenGLWidget* widget, int x, int y);

///\fn void doTranslation(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY)
///\brief Handles translation events
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param startX The starting X coordinate
///\param startY The starting Y coordinate
///\param endX The ending X coordinate
///\param endY The ending Y coordinate
	void doTranslation(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY);
///\fn void doScaling(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY)
///\brief Handles scaling events
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param startX The starting X coordinate
///\param startY The starting Y coordinate
///\param endX The ending X coordinate
///\param endY The ending Y coordinate
	void doScaling(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY);
///\fn void doRotation(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY)
///\brief Handles rotation events
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param startX The starting X coordinate
///\param startY The starting Y coordinate
///\param endX The ending X coordinate
///\param endY The ending Y coordinate
	void doRotation(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY);

	void setDefaults();
///\fn inline int invertY(int y, int height) const
///\brief Inverts a Y coordinate
///\param y The coordinate to be inverted
///\param height The height of the widget
	inline int invertY(int y, int height) const;

	GeometryRenderer *m_Geometry;
	Qt::ButtonState m_Button;
	int m_OldX, m_OldY;
	bool m_ButtonDown;

};

#endif

