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

// SimpleOpenGLWidget.cpp: implementation of the SimpleOpenGLWidget class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/SimpleOpenGLWidget.h>

//#include <qobjectlist.h>
#include <QObject>
#include <qpainter.h>
#include <VolumeWidget/MouseHandler.h>
#include <VolumeWidget/View.h>
#include <VolumeWidget/PerspectiveView.h>
#include <VolumeWidget/OrthographicView.h>
#include <VolumeWidget/Renderable.h>


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SimpleOpenGLWidget::SimpleOpenGLWidget( QWidget *parent, const char *name )
: QGLWidget( parent, name ), m_BackgroundColor(0,0,0)
{
#if 0
	if (parent) {
		QObjectList *l = parent->queryList( "SimpleOpenGLWidget" );
		QObjectListIt it( *l ); // iterate over the buttons
		QObject *obj;

		while ( (obj = it.current()) != 0 ) {
			// for each found object...
			++it;
			if (this!=obj) {
				((QGLContext*)context())->reset();
				((QGLContext*)context())->create( ((SimpleOpenGLWidget*)obj)->context());

				break;

			}
		}
		delete l; // delete the list, not the objects
	}
#endif

	m_LeftHandler = 0;
	m_MiddleHandler = 0;
	m_RightHandler = 0;
	m_WheelHandler = 0;
	m_SpecialHandler = 0;
	m_View = 0;
	m_MainRenderable = 0;
	m_ManipulateObjects = false;
	m_SelectMode = false;
	initView();
	
}

SimpleOpenGLWidget::~SimpleOpenGLWidget()
{
	delete m_LeftHandler;
	delete m_MiddleHandler;
	delete m_RightHandler;
	delete m_WheelHandler;
	delete m_SpecialHandler;
	delete m_View;
}

View& SimpleOpenGLWidget::getView()
{
	//if (!m_ManipulateObjects)
	return *m_View;
	//else
	//	return m_MainRenderable->getView();
}

ViewInformation SimpleOpenGLWidget::getViewInformation() const
{
	return m_View->getViewInformation();
}

MouseHandler* SimpleOpenGLWidget::setMouseHandler(MouseHandlerPosition position, MouseHandler* handler)
{
	MouseHandler* ret = 0;
	if(handler == NULL) return ret; //for some reason this is called with a null handler on occasion
	switch (position) {
	case LeftButtonHandler:
		delete m_LeftHandler;
		m_LeftHandler = handler->clone();
		ret = m_LeftHandler;
		break;
	case MiddleButtonHandler:
		delete m_MiddleHandler;
		m_MiddleHandler = handler->clone();
		ret = m_MiddleHandler;
		break;
	case RightButtonHandler:
		delete m_RightHandler;
		m_RightHandler = handler->clone();
		ret = m_RightHandler;
		break;
	case WheelHandler:
		delete m_WheelHandler;
		m_WheelHandler = handler->clone();
		ret = m_WheelHandler;
		break;
	case SpecialHandler:
		delete m_SpecialHandler;
		m_SpecialHandler = handler->clone();
		ret = m_SpecialHandler;
		break;
	}
	return ret;
}

void SimpleOpenGLWidget::clearMouseHandler(MouseHandlerPosition position)
{
	switch (position) {
	case LeftButtonHandler:
		delete m_LeftHandler;
		break;
	case MiddleButtonHandler:
		delete m_MiddleHandler;
		break;
	case RightButtonHandler:
		delete m_RightHandler;
		break;
	case WheelHandler:
		delete m_WheelHandler;
		break;
	case SpecialHandler:
		delete m_SpecialHandler;
		break;
	}
}

MouseHandler * SimpleOpenGLWidget::getMouseHandler(MouseHandlerPosition position)
{
	MouseHandler *ret = NULL;
	
	switch (position) {
	case LeftButtonHandler:
		ret = m_LeftHandler;
		break;
	case MiddleButtonHandler:
		ret = m_MiddleHandler;
		break;
	case RightButtonHandler:
		ret = m_RightHandler;
		break;
	case WheelHandler:
		ret = m_WheelHandler;
		break;
	case SpecialHandler:
		ret = m_SpecialHandler;
		break;
	}

	return ret;
}

int SimpleOpenGLWidget::getWidth() const
{
	return m_Width;
}

int SimpleOpenGLWidget::getHeight() const
{
	return m_Height;
}

void SimpleOpenGLWidget::setMainRenderable(Renderable* renderable)
{
	m_MainRenderable = renderable;
}

void SimpleOpenGLWidget::unsetMainRenderable()
{
	m_MainRenderable = 0;
}

bool SimpleOpenGLWidget::initForContext(Renderable* renderable)
{
	makeCurrent();

	return renderable->initForContext();
}

bool SimpleOpenGLWidget::deinitForContext(Renderable* renderable)
{
	makeCurrent();

	return renderable->initForContext();
}

void SimpleOpenGLWidget::setBackgroundColor(const QColor& backgroundColor)
{
	m_BackgroundColor = backgroundColor;
}

const QColor& SimpleOpenGLWidget::getBackgroundColor()
{
	return m_BackgroundColor;
}

void SimpleOpenGLWidget::setObjectManipulationMode(bool state)
{
	m_ManipulateObjects = state;
}

QSizePolicy SimpleOpenGLWidget::sizePolicy() const
{
	return QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
}

QSize SimpleOpenGLWidget::sizeHint() const
{
	return QSize(320,320);
}

void SimpleOpenGLWidget::paintGL()
{
	qglClearColor(m_BackgroundColor);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	//if (m_ManipulateObjects) drawManipulationIndicator();
	glInitNames();

	prepareTransformation();
	
	if (m_MainRenderable) m_MainRenderable->render();
}

void SimpleOpenGLWidget::resizeGL( int w, int h )
{
	glViewport( 0, 0, (GLint)w, (GLint)h );
	m_Width = w;
	m_Height = h;
	if (m_View) m_View->resizeWindow(w,h);
}

void SimpleOpenGLWidget::initializeGL()
{
	GLenum err = glewInit();
	if(GLEW_OK != err)
	{
		fprintf(stderr,"Error: %s\n", glewGetErrorString(err));
	}
	fprintf(stdout,"Status: Using GLEW %s\n",glewGetString(GLEW_VERSION));

	setAutoBufferSwap( true );
	initLighting();
}

void SimpleOpenGLWidget::initView()
{
	PerspectiveView* view = new PerspectiveView(4.0);
	//OrthographicView* view = new OrthographicView(4.0);
	view->setFieldOfView(30.0f*3.14159f/180.0f);
	m_View = view;
}

void SimpleOpenGLWidget::initLighting()
{
	GLfloat position0[] = {70.0f, 50.0f, 100.0f, 0.0f};
	GLfloat position1[] = {-100.0f, 200.0f, 100.0f, 0.0f};
	GLfloat lightColor0[] = {0.75f, 0.75f, 0.75f, 1.0f};
	GLfloat lightColor1[] = {0.25f, 0.25f, 0.25f, 1.0f};
	
	glLightfv(GL_LIGHT0, GL_POSITION, position0);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor0);
	glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor0);
	glLightfv(GL_LIGHT0, GL_AMBIENT, lightColor0);

	glLightfv(GL_LIGHT1, GL_POSITION, position1);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor1);
	glLightfv(GL_LIGHT1, GL_SPECULAR, lightColor1);
	glLightfv(GL_LIGHT1, GL_AMBIENT, lightColor1);

	glEnable( GL_NORMALIZE );
	glEnable( GL_LIGHTING );
	glEnable( GL_LIGHT0 );
	glEnable( GL_LIGHT1 );
	
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
	glDisable( GL_CULL_FACE );
}

void SimpleOpenGLWidget::mousePressEvent(QMouseEvent* e)
{
  
  if(m_SelectMode)
    {
      

      m_SelectMode = false;
    }

	if (m_ManipulateObjects && m_SpecialHandler) {
		m_SpecialHandler->mousePressEvent(this, e);
	}
	else if (e->button() == Qt::LeftButton && m_LeftHandler) {
		m_LeftHandler->mousePressEvent(this, e);
	}
	else if (e->button() == Qt::MidButton && m_MiddleHandler) {
		m_MiddleHandler->mousePressEvent(this, e);
	}
	else if (e->button() == Qt::RightButton && m_RightHandler) {
		m_RightHandler->mousePressEvent(this, e);
	}
}

void SimpleOpenGLWidget::mouseReleaseEvent(QMouseEvent* e)
{
	if (m_ManipulateObjects && m_SpecialHandler) {
		m_SpecialHandler->mouseReleaseEvent(this, e);
	}
	else if (e->button() == Qt::LeftButton && m_LeftHandler) {
		m_LeftHandler->mouseReleaseEvent(this, e);
	}
	else if (e->button() == Qt::MidButton && m_MiddleHandler) {
		m_MiddleHandler->mouseReleaseEvent(this, e);
	}
	else if (e->button() == Qt::RightButton && m_RightHandler) {
		m_RightHandler->mouseReleaseEvent(this, e);
	}
}

void SimpleOpenGLWidget::mouseDoubleClickEvent(QMouseEvent* e)
{
	if (m_ManipulateObjects && m_SpecialHandler) {
		m_SpecialHandler->mouseDoubleClickEvent(this, e);
	}
	else if (e->button() == Qt::LeftButton && m_LeftHandler) {
		m_LeftHandler->mouseDoubleClickEvent(this, e);
	}
	else if (e->button() == Qt::MidButton && m_MiddleHandler) {
		m_MiddleHandler->mouseDoubleClickEvent(this, e);
	}
	else if (e->button() == Qt::RightButton && m_RightHandler) {
		m_RightHandler->mouseDoubleClickEvent(this, e);
	}
}

void SimpleOpenGLWidget::mouseMoveEvent(QMouseEvent* e)
{
	// pass the event to each handler
	if (m_ManipulateObjects && m_SpecialHandler) {
		m_SpecialHandler->mouseMoveEvent(this, e);
	}
	if (e->state() & Qt::LeftButton && m_LeftHandler) {
		m_LeftHandler->mouseMoveEvent(this, e);
	}
	if (e->state() & Qt::MidButton && m_MiddleHandler) {
		m_MiddleHandler->mouseMoveEvent(this, e);
	}
	if (e->state() & Qt::RightButton && m_RightHandler) {
		m_RightHandler->mouseMoveEvent(this, e);
	}
}

void SimpleOpenGLWidget::wheelEvent(QWheelEvent* e)
{
	if (m_WheelHandler) {
		m_WheelHandler->wheelEvent(this, e);
	}
}

void SimpleOpenGLWidget::prepareTransformation()
{
	m_View->SetView();
}

void SimpleOpenGLWidget::drawManipulationIndicator()
{
	// no transformation
	glMatrixMode(GL_MODELVIEW_MATRIX);
  glLoadIdentity();

	initLighting();
	// no depth writes
	//glDepthMask(GL_FALSE);

	// draw a red frame around the scene
	glBegin(GL_QUADS);
		glColor3d(1.0,1.0,1.0);
		glVertex3f(-1.0,-1.0,-1.0);
		glVertex3f(-1.0,1.0,-1.0);
		glVertex3f(1.0,1.0,-1.0);
		glVertex3f(1.0,-1.0,-1.0);
	glEnd();

	//qglColor(m_BackgroundColor);
	//glBegin(GL_QUADS);
	//	glVertex3f(-1.0,-1.0,1.0);
	//	glVertex3f(-1.0,1.0,1.0);
	//	glVertex3f(1.0,1.0,1.0);
	//	glVertex3f(1.0,-1.0,1.0);
	//glEnd();

	
	// restore depth writes
	//glDepthMask(GL_TRUE);
}

