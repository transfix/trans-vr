/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
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

// Axis3DHandler.cpp: implementation of the Axis3DHandler class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRover/Axis3DHandler.h>
#include <VolumeRover/Rover3DWidget.h>
#include <math.h>
#include <VolumeWidget/Ray.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/MouseEvent3DPrivate.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Axis3DHandler::Axis3DHandler(Rover3DWidget* rover3DWidget, Axis3DHandler::Axis axis) :
m_Rover3DWidget(rover3DWidget), m_Axis(axis)
{
	m_MouseDown = false;
}

Axis3DHandler::~Axis3DHandler()
{

}

Mouse3DHandler* Axis3DHandler::clone() const
{
	return new Axis3DHandler(*this);
}


bool Axis3DHandler::mousePress3DEvent(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	Vector origin = m_Rover3DWidget->getSubVolume().getOrigin();

	float distance = 0.f;
	distance = nearestTOnAxis(e);
	Vector point = nearestPointOnAxis(e);
	bool isValid = m_Rover3DWidget->getSubVolume().withinCube(point) && closeToMouse(simpleOpenGLWidget, e, point);
	if (closeToMouse(simpleOpenGLWidget, e, getPNob())) {
		m_MouseDown = true;
		m_Section = PScaleNob;
		m_PositionOnAxis = point - origin;
		m_Rover3DWidget->roverDown(m_Axis);
		simpleOpenGLWidget->updateGL();
		return true;
	}
	else if (closeToMouse(simpleOpenGLWidget, e, getNNob())) {
		m_MouseDown = true;
		m_Section = NScaleNob;
		m_PositionOnAxis = point - origin;
		m_Rover3DWidget->roverDown(m_Axis);
		simpleOpenGLWidget->updateGL();
		return true;
	}
	else if (isValid) {
		m_MouseDown = true;
		m_Section = Middle;
		m_PositionOnAxis = point - origin;
		m_Rover3DWidget->roverDown(m_Axis);
		simpleOpenGLWidget->updateGL();
		return true;
	}
	else {
		m_MouseDown = false;
		return false;
	}
}

bool Axis3DHandler::mouseRelease3DEvent(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	m_MouseDown = false;
	m_Rover3DWidget->roverReleased(m_Axis);
	simpleOpenGLWidget->updateGL();
	return true;
}

bool Axis3DHandler::mouseDoubleClick3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e)
{
	// ignore this for now
	return false;
}

bool Axis3DHandler::mouseMove3DEvent(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	if (m_MouseDown) {
		Vector point = nearestPointOnAxis(e);
		if (m_Section==PScaleNob) {
			setPScaleNob(point);
			m_Rover3DWidget->roverMoving(m_Axis);
			simpleOpenGLWidget->updateGL();
			return true;
		}
		else if (m_Section==NScaleNob) {
			setNScaleNob(point);
			m_Rover3DWidget->roverMoving(m_Axis);
			simpleOpenGLWidget->updateGL();
			return true;
		}
		else { // middle
			m_Rover3DWidget->setOrigin(point-m_PositionOnAxis);
			m_Rover3DWidget->roverMoving(m_Axis);
			simpleOpenGLWidget->updateGL();
			return true;
		}
	}
	else {
		return false;
	}
}


float Axis3DHandler::getNearestClickDistance(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e)
{
	float distance = nearestTOnAxis(e);
	Vector point = nearestPointOnAxis(e);
	bool isValid = m_Rover3DWidget->getSubVolume().withinCube(point) && closeToMouse(simpleOpenGLWidget, e, point);
	if (closeToMouse(simpleOpenGLWidget, e, getPNob())) {
		return distance;
	}
	else if (closeToMouse(simpleOpenGLWidget, e, getNNob())) {
		return distance;
	}
	else if (isValid) {
		return distance;
	}
	else {
		return -1.0;
	}
}

Axis3DHandler::Axis Axis3DHandler::getAxis() const
{
	return m_Axis;
}

bool Axis3DHandler::closeToMouse(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e, const Vector& vector)
{
	Vector screenPoint = simpleOpenGLWidget->getView().GetScreenPoint(vector);
	double d1 = fabs((float)(screenPoint[0]-(float)e->x()));
	double d2 = fabs((float)(screenPoint[1]-(float)e->y()));
	return  d1 < 5.0 &&
		 d2 < 5.0;
}

float Axis3DHandler::nearestTOnAxis(MouseEvent3DPrivate* e) const
{

	Vector origin = m_Rover3DWidget->getSubVolume().getOrigin();
	Ray pickRay = e->ray();
	if (m_Axis==XAxis) {
		float xDistance = pickRay.nearestTOnXAxis(origin);
		return xDistance;
	}
	else if (m_Axis==YAxis) {
		float yDistance = pickRay.nearestTOnYAxis(origin);
		return yDistance;
	}
	else {
		float zDistance = pickRay.nearestTOnZAxis(origin);
		return zDistance;
	}
}

Vector Axis3DHandler::nearestPointOnAxis(MouseEvent3DPrivate* e) const
{
	Vector origin = m_Rover3DWidget->getSubVolume().getOrigin();
	Ray pickRay = e->ray();
	if (m_Axis==XAxis) {
		Vector xPoint = pickRay.nearestPointOnXAxis(origin);
		return xPoint;
	}
	else if (m_Axis==YAxis) {
		Vector yPoint = pickRay.nearestPointOnYAxis(origin);
		return yPoint;
	}
	else {
		Vector zPoint = pickRay.nearestPointOnZAxis(origin);
		return zPoint;
	}
}

void Axis3DHandler::setPScaleNob(const Vector& value) const
{
	if (m_Axis==XAxis) {
		m_Rover3DWidget->setXScaleNob(value);
	}
	else if (m_Axis==YAxis) {
		m_Rover3DWidget->setYScaleNob(value);
	}
	else {
		m_Rover3DWidget->setZScaleNob(value);
	}
}

void Axis3DHandler::setNScaleNob(const Vector& value) const
{
	if (m_Axis==XAxis) {
		m_Rover3DWidget->setXScaleNob(value);
	}
	else if (m_Axis==YAxis) {
		m_Rover3DWidget->setYScaleNob(value);
	}
	else {
		m_Rover3DWidget->setZScaleNob(value);
	}
}

Vector Axis3DHandler::getPNob()
{
	if (m_Axis==XAxis) {
		return Vector(
			(float)m_Rover3DWidget->getSubVolume().getXMax(),
			(float)((m_Rover3DWidget->getSubVolume().getYMin() + m_Rover3DWidget->getSubVolume().getYMax())/2.0),
			(float)((m_Rover3DWidget->getSubVolume().getZMin() + m_Rover3DWidget->getSubVolume().getZMax())/2.0),
			1.0f
			);
	}
	else if (m_Axis==YAxis) {
		return Vector(
			(float)((m_Rover3DWidget->getSubVolume().getXMin() + m_Rover3DWidget->getSubVolume().getXMax())/2.0),
			(float)m_Rover3DWidget->getSubVolume().getYMax(),
			(float)((m_Rover3DWidget->getSubVolume().getZMin() + m_Rover3DWidget->getSubVolume().getZMax())/2.0),
			1.0f
			);
	}
	else {
		return Vector(
			(float)((m_Rover3DWidget->getSubVolume().getXMin() + m_Rover3DWidget->getSubVolume().getXMax())/2.0),
			(float)((m_Rover3DWidget->getSubVolume().getYMin() + m_Rover3DWidget->getSubVolume().getYMax())/2.0),
			(float)m_Rover3DWidget->getSubVolume().getZMax(),
			1.0f
			);
	}
}

Vector Axis3DHandler::getNNob()
{
	if (m_Axis==XAxis) {
		return Vector(
			(float)m_Rover3DWidget->getSubVolume().getXMin(),
			(float)((m_Rover3DWidget->getSubVolume().getYMin() + m_Rover3DWidget->getSubVolume().getYMax())/2.0),
			(float)((m_Rover3DWidget->getSubVolume().getZMin() + m_Rover3DWidget->getSubVolume().getZMax())/2.0),
			1.0f
			);
	}
	else if (m_Axis==YAxis) {
		return Vector(
			(float)((m_Rover3DWidget->getSubVolume().getXMin() + m_Rover3DWidget->getSubVolume().getXMax())/2.0),
			(float)m_Rover3DWidget->getSubVolume().getYMin(),
			(float)((m_Rover3DWidget->getSubVolume().getZMin() + m_Rover3DWidget->getSubVolume().getZMax())/2.0),
			1.0f
			);
	}
	else {
		return Vector(
			(float)((m_Rover3DWidget->getSubVolume().getXMin() + m_Rover3DWidget->getSubVolume().getXMax())/2.0),
			(float)((m_Rover3DWidget->getSubVolume().getYMin() + m_Rover3DWidget->getSubVolume().getYMax())/2.0),
			(float)m_Rover3DWidget->getSubVolume().getZMin(),
			1.0f
			);
	}
}

