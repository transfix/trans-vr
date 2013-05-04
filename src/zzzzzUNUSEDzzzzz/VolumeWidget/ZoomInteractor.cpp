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

// ZoomInteractor.cpp: implementation of the ZoomInteractor class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/ZoomInteractor.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/View.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ZoomInteractor::ZoomInteractor()
{
	m_Axis = YAxis;
}

ZoomInteractor::~ZoomInteractor()
{

}

MouseHandler* ZoomInteractor::clone() const
{
	return new ZoomInteractor(*this);
}

void ZoomInteractor::doInteraction(SimpleOpenGLWidget* simpleOpenGLWidget, int startX, int startY, int endX, int endY)
{
	float windowSize = simpleOpenGLWidget->getView().GetWindowSize();

	float dz;
	if (m_Axis == XAxis) {
		dz = (float)(endX-startX);
	}
	else { // YAxis
		dz = (float)(startY-endY);
	}

	float incamount = 0.007f/10.0f;
	unsigned int c;
	for (c=0; c<10; c++) {
		windowSize += windowSize*incamount*(dz);
	}

	// update the windowSize on the view
	simpleOpenGLWidget->getView().SetWindowSize(windowSize);
}

void ZoomInteractor::setAxis(Axis axis)
{
	m_Axis = axis;
}

ZoomInteractor::Axis ZoomInteractor::getAxis() const
{
	return m_Axis;
}

