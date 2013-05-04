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

// TransformRenderable.cpp: implementation of the TransformRenderable class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/TransformRenderable.h>
#include <VolumeWidget/Quaternion.h>
#include <glew/glew.h>
#include <qgl.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

TransformRenderable::TransformRenderable(Renderable* renderable) :
m_Renderable(renderable)
{
}

TransformRenderable::~TransformRenderable()
{

}

bool TransformRenderable::initForContext()
{
	return (m_Renderable?m_Renderable->initForContext():0);
}

bool TransformRenderable::deinitForContext()
{
	return (m_Renderable?m_Renderable->deinitForContext():0);
}

bool TransformRenderable::render()
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(m_Transformation.getMatrix());
	bool ret = (m_Renderable?m_Renderable->render():0);
	glPopMatrix();
	return ret;
}

void TransformRenderable::translate( float tx, float ty, float tz )
{
	m_Transformation.preMultiplication(Matrix::translation(tx,ty,tz));
}

void TransformRenderable::rotation( float angle, float x, float y, float z )
{
	m_Transformation.preMultiplication(Quaternion::rotation(angle, x,y,z).buildMatrix());
}

void TransformRenderable::scale( float s )
{
	m_Transformation.preMultiplication(Matrix::scale(s,s,s));
}

void TransformRenderable::reset()
{
	m_Transformation.reset();
}

void TransformRenderable::setTransformation( const Matrix& transformation)
{
	m_Transformation = transformation;
}

Matrix& TransformRenderable::getTransformation()
{
	return m_Transformation;
}

const Matrix& TransformRenderable::getTransformation() const
{
	return m_Transformation;
}

