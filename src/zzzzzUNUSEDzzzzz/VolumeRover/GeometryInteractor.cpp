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
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// GeometryInteractor.cpp: implementation of the GeometryInteractor class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRover/GeometryInteractor.h>
#include <VolumeRover/GeometryRenderer.h>

#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/View.h>
#include <VolumeWidget/Quaternion.h>
#include <VolumeWidget/Vector.h>
#include <math.h>

#include <Qt>
//Added by qt3to4:
#include <QWheelEvent>
#include <QMouseEvent>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

static float projectToBall(float ballSize, float x, float y)
{
	float d, t, z;
	
	d = (float)sqrt(x * x + y * y);
	if (d < ballSize * 0.70710678118654752440) {  /* Inside sphere. */
		z = (float)sqrt(ballSize * ballSize - d * d);
	} else {              /* On hyperbola. */
		t = (float)(ballSize / 1.41421356237309504880);
		z = t * t / d;
	}
	return z;
}

GeometryInteractor::GeometryInteractor(GeometryRenderer *geom)
{
	setDefaults();
	m_Geometry = geom;
}

GeometryInteractor::~GeometryInteractor()
{
}

MouseHandler* GeometryInteractor::clone() const
{
	return new GeometryInteractor(*this);
}

bool GeometryInteractor::mousePressEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	startDrag(simpleOpenGLWidget, e->x(), e->y(), e->button());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool GeometryInteractor::mouseReleaseEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	endDrag(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool GeometryInteractor::mouseDoubleClickEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	// do nothing for now
	return false;
}

bool GeometryInteractor::mouseMoveEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QMouseEvent* e)
{
	mouseMove(simpleOpenGLWidget, e->x(), e->y());
	simpleOpenGLWidget->updateGL();
	return true;
}

bool GeometryInteractor::wheelEvent(SimpleOpenGLWidget* simpleOpenGLWidget, QWheelEvent* e)
{
	// scale
	simpleOpenGLWidget->updateGL();
	return true;
}

void GeometryInteractor::startDrag(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y, Qt::ButtonState button)
{
	// y window coordinates are opposite of opengl coordinates
	// invert
	y = invertY(y, simpleOpenGLWidget->getHeight());

	m_ButtonDown = true;
	m_Button = button;
	m_OldX = x;
	m_OldY = y;
}

void GeometryInteractor::mouseMove(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
{
	// y window coordinates are opposite of opengl coordinates
	// invert
	y = invertY(y, simpleOpenGLWidget->getHeight());

	if (m_ButtonDown) {
		switch (m_Button)
		{
		case Qt::LeftButton:
				doTranslation(simpleOpenGLWidget, m_OldX, m_OldY, x, y);
				break;
		case Qt::MidButton:
				doScaling(simpleOpenGLWidget, m_OldX, m_OldY, x, y);
				break;
		case Qt::RightButton:
				doRotation(simpleOpenGLWidget, m_OldX, m_OldY, x, y);
				break;
			default:
				break;
		}
	}

	m_OldX = x;
	m_OldY = y;
}

void GeometryInteractor::endDrag(SimpleOpenGLWidget* simpleOpenGLWidget, int x, int y)
{
	int dummy = 0;
	dummy = x;
	dummy = y;
	
	if (m_ButtonDown) {
		emit ViewChanged();
	}
	m_ButtonDown = false;
	m_Button = Qt::NoButton;
}

//////////////////////
// The next three functions were cribbed from various ViewInteractor instances
//
void GeometryInteractor::doTranslation(SimpleOpenGLWidget* simpleOpenGLWidget,
													int startX, int startY, int endX, int endY)
{
	Quaternion orientation = simpleOpenGLWidget->getView().getOrientation();
	Vector target = simpleOpenGLWidget->getView().getTarget();
	float windowSize = simpleOpenGLWidget->getView().GetWindowSize();
	int width = simpleOpenGLWidget->getWidth();
	int height = simpleOpenGLWidget->getHeight();

	float dx = (float)(endX-startX);
	float dy = (float)(endY-startY);

	Vector up(orientation.applyRotation(Vector(0.0f, 1.0f, 0.0f, 0.0f)));
	Vector right(orientation.applyRotation(Vector(1.0f, 0.0f, 0.0f, 0.0f)));

	float objectDx = dx * windowSize / (height<width?height: width);
	float objectDy = dy * windowSize / (height<width?height: width);

	Vector displacement(right*objectDx+up*objectDy);
	//target-=displacement;

	// update the target on the view
	//simpleOpenGLWidget->getView().setTarget(target);
	m_Geometry->translateBy(displacement[0]*10.0,
													displacement[1]*10.0,
													displacement[2]*10.0);
}

void GeometryInteractor::doScaling(SimpleOpenGLWidget* simpleOpenGLWidget,
													int startX, int startY, int endX, int endY)
{
	int dummy = 0;
	dummy = startX;
	dummy = endX;
	
	float dz = (float)(startY-endY);
	float incamount = 0.007f/10.0f;

	//simpleOpenGLWidget->getView().zoom(incamount*10.0*dz);
	m_Geometry->scaleBy(incamount*10.0*dz);
}

void GeometryInteractor::doRotation(SimpleOpenGLWidget* simpleOpenGLWidget,
													int startX, int startY, int endX, int endY)
{
	Quaternion orientation = simpleOpenGLWidget->getView().getOrientation();
	float windowSize = simpleOpenGLWidget->getView().GetWindowSize();
	int width = simpleOpenGLWidget->getWidth();
	int height = simpleOpenGLWidget->getHeight();

	float ballSize = windowSize/2.0f;

	// find the eye coordinates of the mouse position
	float objectX1 = (float)(startX-width/2) * windowSize / (float)(height<width?height: width);
	float objectY1 = (float)(startY-height/2) * windowSize / (float)(height<width?height: width);
	float objectX2 = (float)(endX-width/2) * windowSize / (float)(height<width?height: width);
	float objectY2 = (float)(endY-height/2) * windowSize / (float)(height<width?height: width);
	float objectZ1 = projectToBall(ballSize, objectX1, objectY1);
	float objectZ2 = projectToBall(ballSize, objectX2, objectY2);

	Vector v1(objectX1, objectY1, objectZ1, 0.0f);
	Vector v2(objectX2, objectY2, objectZ2, 0.0f);
	Vector axis = v1.cross(v2);

	Vector distanceDirection = v2-v1;
	float distanceScalar = distanceDirection.norm() / (2.0f * ballSize);

	// adjust so its not out of range
	if (distanceScalar > 1.0f) distanceScalar=1.0f;
	if (distanceScalar < -1.0f) distanceScalar=-1.0f;
	float angle = 2.0f * (float)asin(distanceScalar);

	
	// transform the axis to object space
	axis = orientation.applyRotation(axis);

	// rotate the geometry
	m_Geometry->rotateBy(angle, axis[0], axis[1], axis[2]);

	//orientation.postMultiply(Quaternion::rotation(-angle, axis));
	// update the orientation on the view
	//simpleOpenGLWidget->getView().SetOrientation(orientation);
}

void GeometryInteractor::setDefaults()
{
	m_ButtonDown = false;
	m_Button = Qt::NoButton;
	m_OldX = 0;
	m_OldY = 0;
}

int GeometryInteractor::invertY(int y, int height) const
{
	return height-y-1;
}


