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

// RenderableView.h: interface for the RenderableView class.
//
//////////////////////////////////////////////////////////////////////

#ifndef RENDERABLE_VIEW_H
#define RENDERABLE_VIEW_H

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeWidget/View.h>

class RenderableView : public View  
{
public:
	RenderableView (float windowsize);
	RenderableView (const ViewInformation& viewInformation);
	virtual ~RenderableView ();
	virtual View* clone() const;

	virtual void SetView();
	virtual Ray GetPickRay(int x, int y)const;
	virtual Vector GetScreenPoint(const Vector& p) const;

	virtual void zoom(float dz);

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
	float m_ZoomFactor;
};

#endif 

