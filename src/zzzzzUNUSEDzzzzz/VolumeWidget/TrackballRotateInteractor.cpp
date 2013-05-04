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

// TrackballRotateInteractor.cpp: implementation of the TrackballRotateInteractor class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/TrackballRotateInteractor.h>
#include <math.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/View.h>
#include <VolumeWidget/Quaternion.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

TrackballRotateInteractor::TrackballRotateInteractor()
{

}

TrackballRotateInteractor::~TrackballRotateInteractor()
{

}

MouseHandler* TrackballRotateInteractor::clone() const
{
	return new TrackballRotateInteractor(*this);
}

static float projectToBall(float ballSize, float x, float y)
{
	float d, t, z;
	
	d = (float)sqrt(x * x + y * y);
	if (d < ballSize * 0.70710678118654752440) {  /* Inside sphere. */
		z = (float)sqrt(ballSize * ballSize - d * d);
	} else {              /* On hyperbola. */
		t = (float)(ballSize / 1.41421356237309504880);
		z = t * t / d;
	}
	return z;
}

void TrackballRotateInteractor::doInteraction(SimpleOpenGLWidget* simpleOpenGLWidget, int startX, int startY, int endX, int endY)
{
	Quaternion orientation = simpleOpenGLWidget->getView().getOrientation();
	float windowSize = simpleOpenGLWidget->getView().GetWindowSize();
	int width = simpleOpenGLWidget->getWidth();
	int height = simpleOpenGLWidget->getHeight();

	float ballSize = windowSize/2.0f;

	// find the eye coordinates of the mouse position
	float objectX1 = (float)(startX-width/2) * windowSize / (float)(height<width?height: width);
	float objectY1 = (float)(startY-height/2) * windowSize / (float)(height<width?height: width);
	float objectX2 = (float)(endX-width/2) * windowSize / (float)(height<width?height: width);
	float objectY2 = (float)(endY-height/2) * windowSize / (float)(height<width?height: width);
	float objectZ1 = projectToBall(ballSize, objectX1, objectY1);
	float objectZ2 = projectToBall(ballSize, objectX2, objectY2);

	Vector v1(objectX1, objectY1, objectZ1, 0.0f);
	Vector v2(objectX2, objectY2, objectZ2, 0.0f);
	Vector axis = v1.cross(v2);

	Vector distanceDirection = v2-v1;
	float distanceScalar = distanceDirection.norm() / (2.0f * ballSize);

	// adjust so its not out of range
	if (distanceScalar > 1.0f) distanceScalar=1.0f;
	if (distanceScalar < -1.0f) distanceScalar=-1.0f;
	float angle = 2.0f * (float)asin(distanceScalar);




	orientation.postMultiply(Quaternion::rotation(-angle, axis));
	// update the orientation on the view
	simpleOpenGLWidget->getView().SetOrientation(orientation);
}

