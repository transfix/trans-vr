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

// OrthographicView.h: interface for the OrthographicView class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ORTHOGRAPHICVIEW_H__4003C434_82A1_4E4E_9154_10785DB00338__INCLUDED_)
#define AFX_ORTHOGRAPHICVIEW_H__4003C434_82A1_4E4E_9154_10785DB00338__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeWidget/View.h>
#include <VolumeWidget/Vector.h>
#include <VolumeWidget/Quaternion.h>

class OrthographicView : public View  
{
public:
	OrthographicView(float windowsize);
	OrthographicView(const ViewInformation& viewInformation);
	virtual ~OrthographicView();
	virtual View* clone() const;

	static OrthographicView* Top(float windowsize);
	static OrthographicView* Right(float windowsize);
	static OrthographicView* Left(float windowsize);
	static OrthographicView* Bottom(float windowsize);
	static OrthographicView* Front(float windowsize);
	static OrthographicView* Back(float windowsize	);

	virtual void SetView();
	virtual Ray GetPickRay(int x, int y) const;
	virtual Vector GetScreenPoint(const Vector& p) const;

	virtual void defaultTransformation( int xNew, int yNew );

	virtual ViewInformation getViewInformation() const;
	virtual void matchViewInformation(const ViewInformation& viewInformation);

	virtual Matrix getModelViewMatrix() const;

};

#endif // !defined(AFX_ORTHOGRAPHICVIEW_H__4003C434_82A1_4E4E_9154_10785DB00338__INCLUDED_)
