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

// View.cpp: implementation of the View class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/View.h>
#include <math.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

View::View(float windowsize) : m_Target(0.0f, 0.0f, 0.0f, 1.0f),
m_Orientation()
{
	m_WindowSize = windowsize;
	m_bDragStarted = false;
	m_Width = 200;
	m_Height = 200;
}

View::View()
{
	m_WindowSize = 100;
	m_bDragStarted = false;
	m_Width = 200;
	m_Height = 200;
}

View::~View()
{

}

void View::SetOrientation(const Quaternion& orientation)
{
	m_Orientation = orientation;
}

void View::SetOrientation(const View& view)
{
	m_Orientation = view.m_Orientation;
}

void View::setTarget(const Vector& target)
{
	m_Target = target;
}

Quaternion View::getOrientation()
{
	return m_Orientation;
}

Vector View::getTarget()
{
	return m_Target;
}

void View::resizeWindow( int w, int h )
{
	m_Width = w;
	m_Height = h;
}

void View::startDrag( int x, int y )
{
	y = InvertY(y);
	m_bDragStarted = true;
	savePosition(x,y);
}

void View::mousePan( int xNew, int yNew )
{
	yNew = InvertY(yNew);
	pan((float)(xNew - m_xOld), (float)(yNew - m_yOld));
	savePosition(xNew, yNew);
}

void View::mouseRotate( int xNew, int yNew )
{
	//rotate((xNew - m_xOld)/100.0f, (yNew - m_yOld)/100.0f);
	yNew = InvertY(yNew);
	rotate((float)(xNew - m_xOld)/100.0f, (float)(yNew - m_yOld)/100.0f);
	savePosition(xNew, yNew);
}

void View::mouseWorldAxisRotate( int xNew, int yNew )
{
	//rotate((xNew - m_xOld)/100.0f, (yNew - m_yOld)/100.0f);
	yNew = InvertY(yNew);
	rotateWorldAxis((float)(xNew - m_xOld)/100.0f, (float)(yNew - m_yOld)/100.0f);
	savePosition(xNew, yNew);
}

void View::mouseTrackBallRotate( int xNew, int yNew )
{
	//rotate((xNew - m_xOld)/100.0f, (yNew - m_yOld)/100.0f);
	yNew = InvertY(yNew);
	rotateTrackBall((float)m_xOld, (float)m_yOld, (float)xNew, (float)yNew);
	savePosition(xNew, yNew);
}

void View::mouseZoom( int xNew, int yNew )
{
	yNew = InvertY(yNew);
	zoom(-(float)(yNew - m_yOld));
	savePosition(xNew, yNew);
}

void View::savePosition(int x, int y)
{
	m_xOld = x;
	m_yOld = y;
}

void View::defaultTransformation( int xNew, int yNew )
{
	mousePan(xNew, yNew);
}

void View::pan( float dx, float dy ) {
	Vector up(m_Orientation.applyRotation(Vector(0.0f, 1.0f, 0.0f, 0.0f)));
	Vector right(m_Orientation.applyRotation(Vector(1.0f, 0.0f, 0.0f, 0.0f)));

	float objectDx = dx * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);
	float objectDy = dy * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);

	Vector displacement(right*objectDx+up*objectDy);
	m_Target-=displacement;

/*	dx = -dx;
	dy = -dy;
	float objectDx = dx * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);
	float objectDy = dy * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);

	float right[4];
	float lookAt[4];

	sub( lookAt, m_Target, m_Eye );
	normalize( lookAt );
	cross( right, lookAt, m_Up );
	normalize( right );

	float vpan[4];
	linear( vpan, right, m_Up, objectDx, objectDy );
	add( m_Eye, vpan );
	add( m_Target, vpan );*/
}

void View::rotate( float dx, float dy ) {

	m_Orientation.postMultiply(Quaternion::rotation(-dx, 0.0f, 1.0f, 0.0f));
	m_Orientation.postMultiply(Quaternion::rotation(dy, 1.0f, 0.0f, 0.0f));
}

void View::rotateWorldAxis( float dx, float dy )
{
	m_Orientation.preMultiply(Quaternion::rotation(-dx, 0.0f, 0.0f, 1.0f));
	m_Orientation.postMultiply(Quaternion::rotation(dy, 1.0f, 0.0f, 0.0f));
}

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

void View::rotateTrackBall( float x1, float y1, float x2, float y2 )
{
	float ballSize = m_WindowSize/2.0f;

	// find the eye coordinates of the mouse position
	float objectX1 = (float)(x1-m_Width/2) * m_WindowSize / (float)(m_Height<m_Width?m_Height: m_Width);
	float objectY1 = (float)(y1-m_Height/2) * m_WindowSize / (float)(m_Height<m_Width?m_Height: m_Width);
	float objectX2 = (float)(x2-m_Width/2) * m_WindowSize / (float)(m_Height<m_Width?m_Height: m_Width);
	float objectY2 = (float)(y2-m_Height/2) * m_WindowSize / (float)(m_Height<m_Width?m_Height: m_Width);
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


	m_Orientation.postMultiply(Quaternion::rotation(-angle, axis));
}

void View::zoom( float dz ) {
	float incamount = 0.007f/10.0f;
	unsigned int c;
	for (c=0; c<10; c++) {
		m_WindowSize += m_WindowSize*incamount*(dz);
	}
}

float View::GetWindowSize() {
	return m_WindowSize;
}


void View::SetWindowSize(float size) {
	m_WindowSize = size;
}

int View::InvertY(int y) const
{
	return m_Height-y-1;
}

