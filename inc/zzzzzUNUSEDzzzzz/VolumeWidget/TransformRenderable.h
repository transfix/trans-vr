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

// TransformRenderable.h: interface for the TransformRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_TRANSFORMRENDERABLE_H__23190966_0855_4B8E_8110_93B3A6643F51__INCLUDED_)
#define AFX_TRANSFORMRENDERABLE_H__23190966_0855_4B8E_8110_93B3A6643F51__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeWidget/Renderable.h>
#include <VolumeWidget/Matrix.h>

///\class TransformRenderable TransformRenderable.h
///\brief This Renderable instance applies it's own viewing transformation
///	before rendering another Renderable instance.
///\author Anthony Thane
class TransformRenderable : public Renderable  
{
public:
///\fn TransformRenderable(Renderable* renderable)
///\param renderable A pointer to a Renderable that should be transformed before rendering
	TransformRenderable(Renderable* renderable);
	virtual ~TransformRenderable();

	virtual bool initForContext();
	virtual bool deinitForContext();
	virtual bool render();

///\fn void translate( float tx, float ty, float tz )
///\brief This function changes the position of the child Renderable.
///\param tx The translation amount in the X direction
///\param ty The translation amount in the Y direction
///\param tz The translation amount in the Z direction
	void translate( float tx, float ty, float tz );
///\fn void rotation( float angle, float x, float y, float z )
///\brief This function rotates the child Renderable some angle about a given
///	axis.
///\param angle The number of radians to rotate
///\param x The X component of the rotation axis
///\param y The Y component of the rotation axis
///\param z The Z component of the rotation axis
	void rotation( float angle, float x, float y, float z );
///\fn void scale( float s )
///\brief This function scales the child Renderable by some amount
///\param s The scaling factor. 1.0 has no effect, > 1.0 makes the object
///	larger, and < 1.0 makes the object smaller.
	void scale( float s );
///\fn void reset()
///\brief This function undoes any previous transformations.
	void reset();
	
///\fn void setTransformation( const Matrix& transformation)
///\brief This function allows a complete transformation in matrix form to be
///	assigned.
///\param transformation A Matrix representing a transformation
	void setTransformation( const Matrix& transformation);
///\fn Matrix& getTransformation()
///\brief This function returns a reference to the current transformation.
///\return A reference to a Matrix
	Matrix& getTransformation();
///\fn const Matrix& getTransformation() const
///\brief This function returns a constant referece to the current
///	transformation.
///\return A constant reference to a Matrix
	const Matrix& getTransformation() const;


protected:
	Renderable* const m_Renderable;
	Matrix m_Transformation;
};

#endif // !defined(AFX_TRANSFORMRENDERABLE_H__23190966_0855_4B8E_8110_93B3A6643F51__INCLUDED_)
