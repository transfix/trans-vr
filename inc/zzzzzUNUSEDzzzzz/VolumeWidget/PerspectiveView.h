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

// PerspectiveView.h: interface for the PerspectiveView class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_PERSPECTIVEVIEW_H__7F523854_7177_40BC_8ACC_0E48DC611DC6__INCLUDED_)
#define AFX_PERSPECTIVEVIEW_H__7F523854_7177_40BC_8ACC_0E48DC611DC6__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeWidget/View.h>

class PerspectiveView : public View  
{
public:
	PerspectiveView(float windowsize);
	PerspectiveView(const ViewInformation& viewInformation);
	virtual ~PerspectiveView();
	virtual View* clone() const;

	virtual void SetView();
	virtual Ray GetPickRay(int x, int y)const;
	virtual Vector GetScreenPoint(const Vector& p) const;

	virtual void defaultTransformation( int xNew, int yNew );

	void setFieldOfView(float angle);
	float getFieldOfView();

	virtual ViewInformation getViewInformation() const;
	virtual void matchViewInformation(const ViewInformation& viewInformation);

	virtual Matrix getModelViewMatrix() const;

	float getDistanceToEye() const;

	static const float defaultFieldOfView;
protected:
	float m_FieldOfView;
};

#endif // !defined(AFX_PERSPECTIVEVIEW_H__7F523854_7177_40BC_8ACC_0E48DC611DC6__INCLUDED_)
