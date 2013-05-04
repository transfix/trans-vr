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

// TrackballRotateInteractor.h: interface for the TrackballRotateInteractor class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_TRACKBALLROTATEINTERACTOR_H__426D70AB_3F52_4DD1_B86F_FA67C2534866__INCLUDED_)
#define AFX_TRACKBALLROTATEINTERACTOR_H__426D70AB_3F52_4DD1_B86F_FA67C2534866__INCLUDED_

#include <VolumeWidget/ViewInteractor.h>

class TrackballRotateInteractor : public ViewInteractor  
{
public:
	TrackballRotateInteractor();
	virtual ~TrackballRotateInteractor();

	virtual MouseHandler* clone() const;

	virtual void doInteraction(SimpleOpenGLWidget*, int startX, int startY, int endX, int endY);

};

#endif // !defined(AFX_TRACKBALLROTATEINTERACTOR_H__426D70AB_3F52_4DD1_B86F_FA67C2534866__INCLUDED_)
