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

// WireCubeRenderable.cpp: implementation of the WireCubeRenderable class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/WireCubeRenderable.h>
#include <glew/glew.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

WireCubeRenderable::WireCubeRenderable() :
GeometryRenderable(&m_WireCubeGeometry, false), m_Boundary(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
{
	m_GeometriesAllocated = false;
	m_R = 1.0;
	m_G = 1.0;
	m_B = 1.0;

	prepareGeometry();
}

WireCubeRenderable::~WireCubeRenderable()
{

}

void WireCubeRenderable::setAspectRatio(double x, double y, double z)
{
	double maxratio, ratioX=x, ratioY=y, ratioZ=z;

	// find the maximum ratio
	maxratio = ( ratioX > ratioY ? ratioX : ratioY );
	maxratio = ( maxratio > ratioZ ? maxratio : ratioZ );

	// normalize so the max ratio is 1.0
	ratioX /= maxratio;
	ratioY /= maxratio;
	ratioZ /= maxratio;
	m_Boundary.setExtents(-0.5*ratioX, 0.5*ratioX, -0.5*ratioY, 0.5*ratioY, -0.5*ratioZ, 0.5*ratioZ);

	prepareGeometry();
}

void WireCubeRenderable::setColor(float r, float g, float b)
{
	m_R = r;
	m_G = g;
	m_B = b;

	prepareGeometry();
}

void WireCubeRenderable::prepareGeometry()
{
	unsigned int numberoflines = 12;
	unsigned int numberoflineverts = 8;

	if (!m_GeometriesAllocated) {
		m_WireCubeGeometry.AllocateLines(numberoflineverts, numberoflines);
		m_WireCubeGeometry.AllocateLineColors();

		m_GeometriesAllocated = true;
	}

	// the boundary wirecube
	// make 8 points  zyx
	m_WireCubeGeometry.m_LineVerts[0*3+0] = (float) m_Boundary.getXMin();	// point 0
	m_WireCubeGeometry.m_LineVerts[0*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[0*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[0*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[0*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[0*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[1*3+0] = (float) m_Boundary.getXMax();	// point 1
	m_WireCubeGeometry.m_LineVerts[1*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[1*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[1*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[1*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[1*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[2*3+0] = (float) m_Boundary.getXMin();	// point 2
	m_WireCubeGeometry.m_LineVerts[2*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[2*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[2*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[2*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[2*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[3*3+0] = (float) m_Boundary.getXMax();	// point 3
	m_WireCubeGeometry.m_LineVerts[3*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[3*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[3*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[3*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[3*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[4*3+0] = (float) m_Boundary.getXMin();	// point 4
	m_WireCubeGeometry.m_LineVerts[4*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[4*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[4*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[4*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[4*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[5*3+0] = (float) m_Boundary.getXMax();	// point 5
	m_WireCubeGeometry.m_LineVerts[5*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[5*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[5*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[5*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[5*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[6*3+0] = (float) m_Boundary.getXMin();	// point 6
	m_WireCubeGeometry.m_LineVerts[6*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[6*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[6*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[6*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[6*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[7*3+0] = (float) m_Boundary.getXMax();	// point 7
	m_WireCubeGeometry.m_LineVerts[7*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[7*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[7*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[7*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[7*3+2] = m_B;

	// make 12 lines
	m_WireCubeGeometry.m_Lines[0*2+0] = 0; // line 0
	m_WireCubeGeometry.m_Lines[0*2+1] = 2;
	m_WireCubeGeometry.m_Lines[1*2+0] = 2; // line 1
	m_WireCubeGeometry.m_Lines[1*2+1] = 3;
	m_WireCubeGeometry.m_Lines[2*2+0] = 3; // line 2
	m_WireCubeGeometry.m_Lines[2*2+1] = 1;
	m_WireCubeGeometry.m_Lines[3*2+0] = 1; // line 3
	m_WireCubeGeometry.m_Lines[3*2+1] = 0;

	m_WireCubeGeometry.m_Lines[4*2+0] = 4; // line 4
	m_WireCubeGeometry.m_Lines[4*2+1] = 6;
	m_WireCubeGeometry.m_Lines[5*2+0] = 6; // line 5
	m_WireCubeGeometry.m_Lines[5*2+1] = 7;
	m_WireCubeGeometry.m_Lines[6*2+0] = 7; // line 6
	m_WireCubeGeometry.m_Lines[6*2+1] = 5;
	m_WireCubeGeometry.m_Lines[7*2+0] = 5; // line 7
	m_WireCubeGeometry.m_Lines[7*2+1] = 4;

	m_WireCubeGeometry.m_Lines[8*2+0] = 0; // line 8
	m_WireCubeGeometry.m_Lines[8*2+1] = 4;
	m_WireCubeGeometry.m_Lines[9*2+0] = 2; // line 9
	m_WireCubeGeometry.m_Lines[9*2+1] = 6;
	m_WireCubeGeometry.m_Lines[10*2+0] = 3; // line 10
	m_WireCubeGeometry.m_Lines[10*2+1] = 7;
	m_WireCubeGeometry.m_Lines[11*2+0] = 1; // line 11
	m_WireCubeGeometry.m_Lines[11*2+1] = 5;

}

