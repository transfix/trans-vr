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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// ZoomInteractor.h: interface for the ZoomInteractor class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ZOOMINTERACTOR_H__D198A8A7_FB0C_45E4_A503_F8FDED0E11EC__INCLUDED_)
#define AFX_ZOOMINTERACTOR_H__D198A8A7_FB0C_45E4_A503_F8FDED0E11EC__INCLUDED_

#include <VolumeWidget/ViewInteractor.h>

class ZoomInteractor : public ViewInteractor  
{
protected:
	enum Axis { XAxis, YAxis };

public:
	ZoomInteractor();
	virtual ~ZoomInteractor();

	virtual MouseHandler* clone() const;

	virtual void doInteraction(SimpleOpenGLWidget*, int startX, int startY, int endX, int endY);

	void setAxis(Axis axis);
	Axis getAxis() const;

protected:
	Axis m_Axis;

};

#endif // !defined(AFX_ZOOMINTERACTOR_H__D198A8A7_FB0C_45E4_A503_F8FDED0E11EC__INCLUDED_)
