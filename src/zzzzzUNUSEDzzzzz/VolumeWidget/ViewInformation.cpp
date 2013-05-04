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

// ViewInformation.cpp: implementation of the ViewInformation class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/ViewInformation.h>
#include <math.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ViewInformation::ViewInformation(const Vector& target, const Quaternion& orientation, float windowSize, float fov) :
m_Target(target), m_Orientation(orientation), m_WindowSize(windowSize), m_Fov(fov)
{
	m_ClipPlane = 0.0;
}

ViewInformation::~ViewInformation()
{

}

Vector ViewInformation::getTarget() const
{
	return m_Target;
}

Quaternion ViewInformation::getOrientation() const
{
	return m_Orientation;
}

void ViewInformation::setClipPlane(float clipPlane)
{
	m_ClipPlane = clipPlane;
}

float ViewInformation::getClipPlane() const
{
	return m_ClipPlane;
}

float ViewInformation::getFov() const
{
	return m_Fov;
}

float ViewInformation::getWindowSize() const
{
	return m_WindowSize;
}

bool ViewInformation::isPerspective() const
{
	return m_Fov!=0.0;
}


// the eye point for an orthographic view will be at infinity (x,y,z,0)
// the eye point for a perspective view will not be at infinity (x,y,z,1)
Vector ViewInformation::getEyePoint() const
{
	if (isPerspective()) {
		return getEyePointPerspective();
	}
	else {
		return getEyePointOrthographic();
	}
}

// returns a unit length up vector
Vector ViewInformation::getUpVector() const
{
	Vector up(m_Orientation.applyRotation(Vector(0.0f, 1.0f, 0.0f, 0.0f)));
	return up;
}

// returns a unit lenght right vector
Vector ViewInformation::getRightVector() const
{
	Vector right(m_Orientation.applyRotation(Vector(1.0f, 0.0f, 0.0f, 0.0f)));
	return right;
}

// returns a unit length view vector
Vector ViewInformation::getViewVector() const
{
	Vector view(m_Orientation.applyRotation(Vector(0.0f, 0.0f, -1.0f, 0.0f)));
	return view;
}

// the eye point for an orthographic view will be at infinity (x,y,z,0)
Vector ViewInformation::getEyePointOrthographic() const
{
	Vector eyePoint(m_Orientation.applyRotation(Vector(0.0f, 0.0f, 1.0f, 0.0f)));
	return eyePoint;
}

// the eye point for a perspective view will not be at infinity (x,y,z,1)
Vector ViewInformation::getEyePointPerspective() const
{
	float distance;
	// the distance of the eye from the target based on the fov and 
	// windowsize
	distance = m_WindowSize / 2.0f / (float)tan( m_Fov/2.0f );
	Vector eyePoint(m_Orientation.applyRotation(Vector(0.0f, 0.0f, distance, 1.0f)));
	return eyePoint;
}

