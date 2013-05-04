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

// View.h: interface for the View class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VIEW_H__B3C3BBFF_964D_4AE0_AB50_AC1D94E2B32D__INCLUDED_)
#define AFX_VIEW_H__B3C3BBFF_964D_4AE0_AB50_AC1D94E2B32D__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeWidget/Vector.h>
#include <VolumeWidget/Quaternion.h>
#include <VolumeWidget/ViewInformation.h>

class View  
{
public:
	View(float windowsize);
	virtual ~View();
	virtual View* clone() const = 0;
	virtual void SetView() = 0;
	virtual Ray GetPickRay(int x, int y) const = 0;
	virtual Vector GetScreenPoint(const Vector& p) const = 0;

	virtual void SetOrientation(const Quaternion& orientation);
	virtual void SetOrientation(const View& view);
	virtual void setTarget(const Vector& target);

	virtual Quaternion getOrientation();
	virtual Vector getTarget();

	virtual void resizeWindow( int w, int h );

	virtual void startDrag( int x, int y );
	virtual void mousePan( int xNew, int yNew );
	virtual void mouseRotate( int xNew, int yNew );
	virtual void mouseWorldAxisRotate( int xNew, int yNew );
	virtual void mouseTrackBallRotate( int xNew, int yNew );
	virtual void mouseZoom( int xNew, int yNew );
	virtual void defaultTransformation( int xNew, int yNew );

	virtual void pan( float dx, float dy );
	virtual void rotate( float dx, float dy );
	virtual void rotateWorldAxis( float dx, float dy );
	virtual void rotateTrackBall( float x1, float y1, float x2, float y2 );
	virtual void zoom( float dz );

	virtual float GetWindowSize();
	virtual void SetWindowSize(float size);

	virtual int InvertY(int y) const;

	virtual ViewInformation getViewInformation() const = 0;
	virtual void matchViewInformation(const ViewInformation& viewInformation) = 0;

	virtual Matrix getModelViewMatrix() const = 0;

	int m_Width, m_Height;
protected:
	View();
	virtual void savePosition( int x, int y );
	int m_xOld, m_yOld;

	bool m_bDragStarted;

	Vector m_Target;
	Quaternion m_Orientation;
	float m_WindowSize;
};

#endif // !defined(AFX_VIEW_H__B3C3BBFF_964D_4AE0_AB50_AC1D94E2B32D__INCLUDED_)
