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

// PerspectiveView.cpp: implementation of the PerspectiveView class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/PerspectiveView.h>
#include <VolumeWidget/Matrix.h>
#include <VolumeWidget/Ray.h>
#include <glew/glew.h>
#include <math.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

const float PerspectiveView::defaultFieldOfView = 60.0f;

PerspectiveView::PerspectiveView(float windowsize) : View(windowsize)
{
	m_FieldOfView = defaultFieldOfView/180.0f*3.1415926f;
	m_Orientation.rotate(3.1415926f/4.0f, 1.0f, 0.0f, 0.0f);
	m_Orientation.rotate(3.1415926f/4.0f, 0.0f, 0.0f, 1.0f);
}

PerspectiveView::PerspectiveView(const ViewInformation& viewInformation)
{
	matchViewInformation(viewInformation);
}

PerspectiveView::~PerspectiveView()
{

}

View* PerspectiveView::clone() const
{
	return new PerspectiveView(*this);
}

Ray PerspectiveView::GetPickRay(int x, int y) const
{
	y = InvertY(y);
	float fx = (float)x-(float)m_Width/2.0f;
	float fy = (float)y-(float)m_Height/2.0f;
	float objectDx = fx * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);
	float objectDy = fy * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);

	float distance;

	distance = m_WindowSize / 2.0f / (float)tan( m_FieldOfView/2.0f );

	Matrix matrix = Matrix::translation(0.0f, 0.0f, distance);
	matrix.preMultiplication(m_Orientation.buildMatrix());
	matrix.preMultiplication(Matrix::translation(m_Target));

	return Ray(matrix*Vector(0.0f,0.0f,0.0f,1.0f), matrix*Vector(objectDx,objectDy,-distance,0.0f));
}

Vector PerspectiveView::GetScreenPoint(const Vector& p) const
{
	float distance = m_WindowSize / 2.0f / (float)tan( m_FieldOfView/2.0f );

	Matrix matrix = Matrix::translation(-m_Target);
	matrix.preMultiplication(m_Orientation.inverse().buildMatrix());
	matrix.preMultiplication(Matrix::translation(0.0f, 0.0f, -distance));

	Vector result = matrix*p;

	result[0] = result[0] * distance / -result[2];
	result[1] = result[1] * distance / -result[2];
	result[2] = 0.0;
	result[3] = 1.0;

	result[0] = result[0] / (m_WindowSize / (m_Height<m_Width?m_Height: m_Width));
	result[1] = result[1] / (m_WindowSize / (m_Height<m_Width?m_Height: m_Width));
	result[0] = (float)result[0]+(float)m_Width/2.0f;
	result[1] = (float)result[1]+(float)m_Height/2.0f;
	result[1] = InvertY(result[1]);


	return result;

}

void PerspectiveView::SetView()
{
	float distance;
	double nearPlane = m_WindowSize*0.1; //0.1;
	double farPlane = 1400.0; //50.0;

	distance = m_WindowSize / 2.0f / (float)tan( m_FieldOfView/2.0f );

	Matrix matrix = Matrix::translation(-m_Target);
	matrix.preMultiplication(m_Orientation.conjugate().buildMatrix());
	matrix.preMultiplication(Matrix::translation(0.0f, 0.0f, -distance));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (m_Height>m_Width) {
		float angle = 2.0f*atan((double)m_Height / (double)m_Width * tan( m_FieldOfView / 2.0 ) );
		//gluPerspective( angle*180.0/(3.1415925), (double)m_Width / (double)m_Height, 0.1, 1400.0 );
		gluPerspective( angle*180.0/(3.1415925), (double)m_Width / (double)m_Height, nearPlane, farPlane /*1400.0*/ );
	}
	else {
		//gluPerspective( m_FieldOfView *180.0/(3.1415925), (double)m_Width / (double)m_Height, 0.1, 1400.0 );
	        gluPerspective( m_FieldOfView *180.0/(3.1415925), (double)m_Width / (double)m_Height, nearPlane, farPlane /*1400.0*/ );
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(matrix.getMatrix());
}

void PerspectiveView::defaultTransformation( int xNew, int yNew )
{
	//mouseWorldAxisRotate( xNew, yNew );
	mouseTrackBallRotate( xNew, yNew );
}

void PerspectiveView::setFieldOfView(float angle)
{
	// angle is in radians
	m_FieldOfView = angle;
}

float PerspectiveView::getFieldOfView()
{
	return m_FieldOfView;
}

ViewInformation PerspectiveView::getViewInformation() const
{
	return ViewInformation(m_Target, m_Orientation, m_WindowSize, m_FieldOfView);
}

void PerspectiveView::matchViewInformation(const ViewInformation& viewInformation)
{
	m_Target = viewInformation.getTarget();
	m_Orientation = viewInformation.getOrientation();
	m_WindowSize = viewInformation.getWindowSize();
	m_FieldOfView = viewInformation.getFov();
}


Matrix PerspectiveView::getModelViewMatrix() const
{
	float distance;

	distance = m_WindowSize / 2.0f / (float)tan( m_FieldOfView/2.0f );

	Matrix matrix = Matrix::translation(-m_Target);
	matrix.preMultiplication(m_Orientation.conjugate().buildMatrix());
	matrix.preMultiplication(Matrix::translation(0.0f, 0.0f, -distance));
	return matrix;
}

float PerspectiveView::getDistanceToEye() const
{
	return m_WindowSize / 2.0f / (float)tan( m_FieldOfView/2.0f );
}

