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

// SimpleOpenGLWidget.h: interface for the SimpleOpenGLWidget class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_SIMPLEOPENGLWIDGET_H__F3E1C04E_84BE_43BF_93E8_38A672AB157E__INCLUDED_)
#define AFX_SIMPLEOPENGLWIDGET_H__F3E1C04E_84BE_43BF_93E8_38A672AB157E__INCLUDED_

#include <glew/glew.h>
#include <qgl.h>

#include <VolumeWidget/MouseEvent3D.h>
class View;
class MouseHandler;
class Renderable;

///\class SimpleOpenGLWidget SimpleOpenGLWidget.h
///\brief The SimpleOpenGLWidget class is a QGLWidget that provides the OpenGL
///	rendering context for Volume Rover. It handles mouse events via objects
///	derived from the abstract base class MouseHandler and uses a single
///	Renderable instance to draw its contents.
///\author Anthony Thane
///\author John Wiggins

///\enum SimpleOpenGLWidget::MouseHandlerPosition
///\brief These values are used to describe the various positions of
///	MouseHandler objects. A position can either be the mouse button handled by
///	the MouseHandler, or it can be more general as in the case of
///	SpecialHandler.

///\var SimpleOpenGLWidget::MouseHandlerPosition SimpleOpenGLWidget::LeftButtonHandler
/// The MouseHandler for the left mouse button

///\var SimpleOpenGLWidget::MouseHandlerPosition SimpleOpenGLWidget::MiddleButtonHandler
/// The MouseHandler for the middle mouse button

///\var SimpleOpenGLWidget::MouseHandlerPosition SimpleOpenGLWidget::RightButtonHandler
/// The MouseHandler for the right mouse button

///\var SimpleOpenGLWidget::MouseHandlerPosition SimpleOpenGLWidget::WheelHandler
/// The MouseHandler for the mouse wheel

///\var SimpleOpenGLWidget::MouseHandlerPosition SimpleOpenGLWidget::SpecialHandler
/// This MouseHandler is used when oject manipulation mode is enabled. It
///	handles events from all mouse buttons.

class SimpleOpenGLWidget : public QGLWidget 
{
	Q_OBJECT
public:
	SimpleOpenGLWidget( QWidget *parent = 0, const char *name = 0 );
	virtual ~SimpleOpenGLWidget();

///\fn View& getView()
///\brief This function returns a View object which describes the projection
///	and modelview transformations. This function exists so that MouseHandler
///	objects can modify the view transformation.
///\return A reference to a View object
	View& getView();
///\fn ViewInformation getViewInformation() const
///\brief This fuction returns a ViewInformation object which contains similar
///information as a View object, but cannot be modified.
///\return A ViewInformation instance
	ViewInformation getViewInformation() const;

	enum MouseHandlerPosition {LeftButtonHandler, MiddleButtonHandler, RightButtonHandler, WheelHandler, SpecialHandler};

///\fn MouseHandler* setMouseHandler(MouseHandlerPosition position, MouseHandler* handler)
///\brief This function assigns a MouseHandler to one of the
///	MouseHandlerPositions.
///\param position The MouseHandlerPosition to be assigned to
///\param handler A pointer to a MouseHandler
///\return A pointer to a copy of the MouseHandler that was passed to the
///	function or NULL if position is not valid
	MouseHandler* setMouseHandler(MouseHandlerPosition position, MouseHandler* handler);
///\fn void clearMouseHandler(MouseHandlerPosition position)
///\brief This function clears a specified MouseHandler
///\param position The MouseHandlerPosition of the MouseHandler which is to be cleared
	void clearMouseHandler(MouseHandlerPosition position);
///\fn MouseHandler* getMouseHandler(MouseHandlerPosition position)
///\brief This function returns a pointer to a specified MouseHandler.
///\param position the MouseHandlerPosition of the requested MouseHandler
///\return A pointer to a MouseHandler or NULL
	MouseHandler* getMouseHandler(MouseHandlerPosition position);

///\fn int getWidth() const
///\brief This function returns the width of the view.
///\return The integer width of the view
	int getWidth() const;
///\fn int getHeight() const
///\brief This function returns the height of the view.
///\return The integer height of the view
	int getHeight() const;

///\fn void setMainRenderable(Renderable* renderable)
///\brief This function assigns a Renderable object to draw when PaintGL() is
///	called.
///\param renderable A pointer to a Renderable object
	void setMainRenderable(Renderable* renderable);
///\fn void unsetMainRenderable()
///\brief This function clears any references to previously assigned
///	Renderable object.
	void unsetMainRenderable();
///\fn bool initForContext(Renderable* renderable)
///\brief This function initializes a Renderable to render using the object's
///	OpenGL context.
///\param renderable A pointer to a Renderable object
///\return A bool indicating success or failure.
	bool initForContext(Renderable* renderable);
///\fn bool deinitForContext(Renderable* renderable)
///\brief This function undoes the work of initForContext for a given
///	Renderable.
///\param renderable A pointer to a Renderable object
///\return A bool indicating success or failure
	bool deinitForContext(Renderable* renderable);

///\fn void setBackgroundColor(const QColor& backgroundColor)
///\brief This function sets the color used by OpenGL to clear the color
///	buffer.
///\param backgroundColor A QColor object describing an RGB color
	void setBackgroundColor(const QColor& backgroundColor);
///\fn const QColor& getBackgroundColor()
///\brief This function returns the current background color.
///\return A QColor object describing an RGB color
	const QColor& getBackgroundColor();

	// Object Manipulation Mode
	// true == MouseHandlers control individual objects
	// false == MouseHandlers control entire scene (default)
///\fn void setObjectManipulationMode(bool state)
///\brief This function enables/disables object manipulation mode. Object
///	manipulation mode describes the state when mouse events affect the loaded
///	geometry rather than the camera (see GeometryRenderer).
///\param state If true, OMM is enabled. If false, OMM is disabled.
	void setObjectManipulationMode(bool state);
	

protected:
	virtual QSizePolicy sizePolicy() const;
	virtual QSize sizeHint() const;
///\fn virtual void paintGL()
///\brief This function draws the contents of the widget.
	virtual void paintGL();
	virtual void resizeGL( int w, int h );
	virtual void initializeGL();
	virtual void initView();
	virtual void initLighting();
	virtual void mousePressEvent(QMouseEvent* e);
	virtual void mouseReleaseEvent(QMouseEvent* e);
	virtual void mouseDoubleClickEvent(QMouseEvent* e);
	virtual void mouseMoveEvent(QMouseEvent* e);
	virtual void wheelEvent(QWheelEvent* e);

	void prepareTransformation();
	void drawManipulationIndicator();

	QColor m_BackgroundColor;
	MouseHandler* m_LeftHandler;
	MouseHandler* m_MiddleHandler;
	MouseHandler* m_RightHandler;
	MouseHandler* m_WheelHandler;
	MouseHandler* m_SpecialHandler;

	View* m_View;

	Renderable* m_MainRenderable;

	int m_Width;
	int m_Height;

	bool m_ManipulateObjects;

	bool m_SelectMode;
};

#endif // !defined(AFX_SIMPLEOPENGLWIDGET_H__F3E1C04E_84BE_43BF_93E8_38A672AB157E__INCLUDED_)
