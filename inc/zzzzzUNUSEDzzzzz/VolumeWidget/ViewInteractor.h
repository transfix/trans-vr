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

// ViewInteractor.h: interface for the ViewInteractor class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VIEWINTERACTOR_H__4B4D940D_FC32_447F_A45B_F4E53FBF55CD__INCLUDED_)
#define AFX_VIEWINTERACTOR_H__4B4D940D_FC32_447F_A45B_F4E53FBF55CD__INCLUDED_

#include <VolumeWidget/MouseHandler.h>
class View;

///\class ViewInteractor ViewInteractor.h
///\author Anthony Thane
///\brief ViewInteractor is an abstract base class derived from MouseHandler
/// that can modify a SimpleOpenGLWidget's View (the camera transformation)
class ViewInteractor : public MouseHandler
{
	Q_OBJECT
public:
	ViewInteractor();
	virtual ~ViewInteractor();

	virtual bool mousePressEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool mouseReleaseEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool mouseDoubleClickEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool mouseMoveEvent(SimpleOpenGLWidget* widget, QMouseEvent* e);
	virtual bool wheelEvent(SimpleOpenGLWidget* widget, QWheelEvent* e);

signals:
///\fn void ViewChanged( )
///\brief Signals when the mouse interaction is finished
	void ViewChanged( );

protected:
///\fn void startDrag(SimpleOpenGLWidget* widget, int x, int y, ButtonState button)
///\brief Called at the beginning of a mouse interaction
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param x The X coordinate of the current mouse position
///\param y The Y coordinate of the current mouse position
///\param button The mouse button that is pressed
	virtual void startDrag(SimpleOpenGLWidget* widget, int x, int y);
///\fn void mouseMove(SimpleOpenGLWidget* widget, int x, int y)
///\brief Called between startDrag and endDrag whenever the mouse is moved
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param x The X coordinate of the current mouse position
///\param y The Y coordinate of the current mouse position
	virtual void mouseMove(SimpleOpenGLWidget* widget, int x, int y);
///\fn void endDrag(SimpleOpenGLWidget* widget, int x, int y)
///\brief Called at the end of a mouse interaction
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param x The X coordinate of the current mouse position
///\param y The Y coordinate of the current mouse position
	virtual void endDrag(SimpleOpenGLWidget* widget, int x, int y);
///\fn void doInteraction(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY)
///\brief Handles interaction events
///\param widget The SimpleOpenGLWidget that the handler is attached to
///\param startX The starting X coordinate
///\param startY The starting Y coordinate
///\param endX The ending X coordinate
///\param endY The ending Y coordinate
	virtual void doInteraction(SimpleOpenGLWidget* widget, int startX, int startY, int endX, int endY) = 0;

	void setDefaults();
///\fn inline int invertY(int y, int height) const
///\brief Inverts a Y coordinate
///\param y The coordinate to be inverted
///\param height The height of the widget
	inline int invertY(int y, int height) const;

	bool m_ButtonDown;
	int m_OldX, m_OldY;

};

#endif // !defined(AFX_VIEWINTERACTOR_H__4B4D940D_FC32_447F_A45B_F4E53FBF55CD__INCLUDED_)
