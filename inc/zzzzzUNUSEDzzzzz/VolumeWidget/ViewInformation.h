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

// ViewInformation.h: interface for the ViewInformation class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VIEWINFORMATION_H__79949BE3_1946_4B4C_BABA_A4279FB464A9__INCLUDED_)
#define AFX_VIEWINFORMATION_H__79949BE3_1946_4B4C_BABA_A4279FB464A9__INCLUDED_

#include <VolumeWidget/Vector.h>
#include <VolumeWidget/Quaternion.h>


/* Saves information about the current view.  The class View can return ViewInformation
and can use ViewInformation to set itself */
class ViewInformation  
{
public:
	ViewInformation(const Vector& target, const Quaternion& orientation, float windowSize, float fov);
	virtual ~ViewInformation();

	// returns the coresponding member variable
	Vector getTarget() const;
	Quaternion getOrientation() const;
	float getFov() const;
	// returns the size of the window in world coordinates at the target
	// for an orthographic view, this is simply the width of the view window
	float getWindowSize() const;
	bool isPerspective() const;
	
	// special addition for render servers
	// having a set...() function here is really out of place, BUT...
	// there is no way to get the clipping plane value in the View subclasses
	// that normally construct this object. Therefore, it's here so that it can
	// be set by a higher level object
	void setClipPlane(float clipPlane);
	float getClipPlane() const;

	// these are calculated based on the member variables
	
	// the eye point for an orthographic view will be at infinity (x,y,z,0)
	// the eye point for a perspective view will not be at infinity (x,y,z,1)
	Vector getEyePoint() const;

	// returns a unit length up vector
	Vector getUpVector() const;
	// returns a unit lenght right vector
	Vector getRightVector() const;
	// returns a unit length view vector
	Vector getViewVector() const;


protected:
	// the eye point for an orthographic view will be at infinity (x,y,z,0)
	Vector getEyePointOrthographic() const;
	// the eye point for a perspective view will not be at infinity (x,y,z,1)
	Vector getEyePointPerspective() const;


	Vector m_Target;
	Quaternion m_Orientation;
	float m_WindowSize;
	float m_Fov;
	float m_ClipPlane;

};

#endif // !defined(AFX_VIEWINFORMATION_H__79949BE3_1946_4B4C_BABA_A4279FB464A9__INCLUDED_)
