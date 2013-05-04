// MouseSliderHandler.cpp: implementation of the MouseSliderHandler class.
//
//////////////////////////////////////////////////////////////////////

#include <AnimationMaker/MouseSliderHandler.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MouseSliderHandler::MouseSliderHandler(double* target)
: m_Target(target)
{

}

MouseSliderHandler::~MouseSliderHandler()
{

}

MouseHandler* MouseSliderHandler::clone() const
{
	return new MouseSliderHandler(*this);
}

bool MouseSliderHandler::mousePressEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	startDrag(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool MouseSliderHandler::mouseReleaseEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	endDrag(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool MouseSliderHandler::mouseDoubleClickEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	// do nothing for now
	return false;
}

bool MouseSliderHandler::mouseMoveEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	mouseMove(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool MouseSliderHandler::wheelEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QWheelEvent* e)
{
	// simulate a drag in y
	doInteraction(simpleOpenGLWidget, 0, 0, 0, e->delta()/3);
	simpleOpenGLWidget->updateGL();
	return true;
}

void MouseSliderHandler::startDrag(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
{
	// y window coordinates are opposite of opengl coordinates
	// invert
	y = invertY(y, simpleOpenGLWidget->getHeight());

	m_ButtonDown = true;
	m_OldX = x;
	m_OldY = y;
}

void MouseSliderHandler::mouseMove(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
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

void MouseSliderHandler::endDrag(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
{
	if (m_ButtonDown) {
		//emit ViewChanged();
	}
	m_ButtonDown = false;
}

void MouseSliderHandler::setDefaults()
{
	m_ButtonDown = false;
	m_OldX = 0;
	m_OldY = 0;
}

int MouseSliderHandler::invertY(int y, int height) const
{
	return height-y-1;
}

void MouseSliderHandler::doInteraction(SimpleOpenGLWidget* simpleOpenGLWidget, int startX, int startY, int endX, int endY)
{
	(*m_Target) += (double)(endY-startY)/300.0;
	(*m_Target) = ((*m_Target) > 1.0?1.0:(*m_Target));
	(*m_Target) = ((*m_Target) < 0.0?0.0:(*m_Target));

	simpleOpenGLWidget->updateGL();
}

