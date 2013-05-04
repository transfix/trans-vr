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

// PerspectiveView.cpp: implementation of the PerspectiveView class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/RenderableView.h>
#include <VolumeWidget/Matrix.h>
#include <VolumeWidget/Ray.h>
#include <glew/glew.h>
#include <math.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

const float RenderableView::defaultFieldOfView = 60.0f;

RenderableView::RenderableView(float windowsize) : View(windowsize)
{
	m_FieldOfView = defaultFieldOfView/180.0f*3.1415926f;
	m_ZoomFactor = 1.0;
}

RenderableView::RenderableView(const ViewInformation& viewInformation)
{
	matchViewInformation(viewInformation);
}

RenderableView::~RenderableView()
{
}

View* RenderableView::clone() const
{
	return new RenderableView(*this);
}

Ray RenderableView::GetPickRay(int x, int y) const
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

Vector RenderableView::GetScreenPoint(const Vector& p) const
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

// XXX: This happens after the base transformation has been applied
void RenderableView::SetView()
{
	Matrix matrix = Matrix::scale(m_ZoomFactor, m_ZoomFactor, m_ZoomFactor);
	//Matrix matrix = Matrix::translation(-m_Target);
	matrix.postMultiplication(Matrix::translation(-m_Target));
	matrix.postMultiplication(m_Orientation.conjugate().buildMatrix());
	//matrix.postMultiplication(Matrix::scale(m_ZoomFactor, m_ZoomFactor, m_ZoomFactor));

	glMatrixMode(GL_MODELVIEW);
	//glLoadMatrixf(matrix.getMatrix());
	glMultMatrixf(matrix.getMatrix());
}

void RenderableView::zoom(float dz)
{
	m_ZoomFactor += dz;
}

void RenderableView::defaultTransformation( int xNew, int yNew )
{
	//mouseWorldAxisRotate( xNew, yNew );
	mouseTrackBallRotate( xNew, yNew );
}

void RenderableView::setFieldOfView(float angle)
{
	// angle is in radians
	m_FieldOfView = angle;
}

float RenderableView::getFieldOfView()
{
	return m_FieldOfView;
}

ViewInformation RenderableView::getViewInformation() const
{
	return ViewInformation(m_Target, m_Orientation, m_WindowSize, m_FieldOfView);
}

void RenderableView::matchViewInformation(const ViewInformation& viewInformation)
{
	//m_Target = viewInformation.getTarget();
	//m_Orientation = viewInformation.getOrientation();
	m_WindowSize = viewInformation.getWindowSize();
	m_FieldOfView = viewInformation.getFov();
}


Matrix RenderableView::getModelViewMatrix() const
{
	float distance;

	distance = m_WindowSize / 2.0f / (float)tan( m_FieldOfView/2.0f );

	Matrix matrix = Matrix::translation(-m_Target);
	matrix.preMultiplication(m_Orientation.conjugate().buildMatrix());
	matrix.preMultiplication(Matrix::translation(0.0f, 0.0f, -distance));
	return matrix;
}

float RenderableView::getDistanceToEye() const
{
	return m_WindowSize / 2.0f / (float)tan( m_FieldOfView/2.0f );
}

