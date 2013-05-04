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

// PanInteractor.cpp: implementation of the PanInteractor class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/PanInteractor.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/View.h>
#include <VolumeWidget/Quaternion.h>
#include <VolumeWidget/Vector.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

PanInteractor::PanInteractor()
{

}

PanInteractor::~PanInteractor()
{

}

MouseHandler* PanInteractor::clone() const
{
	return new PanInteractor(*this);
}

void PanInteractor::doInteraction(SimpleOpenGLWidget* simpleOpenGLWidget, int startX, int startY, int endX, int endY)
{
	Quaternion orientation = simpleOpenGLWidget->getView().getOrientation();
	Vector target = simpleOpenGLWidget->getView().getTarget();
	float windowSize = simpleOpenGLWidget->getView().GetWindowSize();
	int width = simpleOpenGLWidget->getWidth();
	int height = simpleOpenGLWidget->getHeight();

	float dx = (float)(endX-startX);
	float dy = (float)(endY-startY);

	Vector up(orientation.applyRotation(Vector(0.0f, 1.0f, 0.0f, 0.0f)));
	Vector right(orientation.applyRotation(Vector(1.0f, 0.0f, 0.0f, 0.0f)));

	float objectDx = dx * windowSize / (height<width?height: width);
	float objectDy = dy * windowSize / (height<width?height: width);

	Vector displacement(right*objectDx+up*objectDy);
	target-=displacement;

	// update the target on the view
	simpleOpenGLWidget->getView().setTarget(target);
}

