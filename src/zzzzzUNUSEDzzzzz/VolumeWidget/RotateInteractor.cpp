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

// RotateInteractor.cpp: implementation of the RotateInteractor class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/RotateInteractor.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/View.h>
#include <VolumeWidget/Quaternion.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RotateInteractor::RotateInteractor()
{

}

RotateInteractor::~RotateInteractor()
{

}

MouseHandler* RotateInteractor::clone() const
{
	return new RotateInteractor(*this);
}

void RotateInteractor::doInteraction(SimpleOpenGLWidget* simpleOpenGLWidget, int startX, int startY, int endX, int endY)
{
	Quaternion orientation = simpleOpenGLWidget->getView().getOrientation();

	float dx = (float)(endX-startX)/100.0f;
	float dy = (float)(endY-startY)/100.0f;

	orientation.postMultiply(Quaternion::rotation(-dx, 0.0f, 1.0f, 0.0f));
	orientation.postMultiply(Quaternion::rotation(dy, 1.0f, 0.0f, 0.0f));

	// update the orientation on the view
	simpleOpenGLWidget->getView().SetOrientation(orientation);
}

