/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
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

// Rover3DWidget.cpp: implementation of the Rover3DWidget class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRover/Rover3DWidget.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Rover3DWidget::Rover3DWidget() :
m_Boundary(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), m_SubVolume(-0.25, 0.25, -0.25, 0.25, -0.25, 0.25),
m_WireCubes(&m_WireCubeGeometry, false), m_Axes(&m_AxesGeometry, false), m_Handler(), m_Adapter(&m_Handler)

{
	m_CurrentHighlight = NoAxis;
	m_GeometriesAllocated = false;
	m_R = m_G = m_B = 1.0;
	prepareGeometry();
	prepareHandler();
}

Rover3DWidget::~Rover3DWidget()
{

}

const Extents& Rover3DWidget::getBoundary() const
{
	return m_Boundary;
}

const Extents& Rover3DWidget::getSubVolume() const
{
	return m_SubVolume;
}

void Rover3DWidget::setOrigin(Vector vector)
{
	m_SubVolume.setOrigin(vector, m_Boundary);
}
	
void Rover3DWidget::setXScaleNob(Vector value, bool symetric)
{
	Vector origin = m_SubVolume.getOrigin();
	if (value[0]<origin[0]) {
		setNXScaleNob(value);

		if (symetric) {
			value[0] = origin[0] + (origin[0]-value[0]);
			setPXScaleNob(value);
		}
	}
	else if (value[0]>origin[0]) {
		setPXScaleNob(value);

		if (symetric) {
			value[0] = origin[0] - (value[0]-origin[0]);
			setNXScaleNob(value);
		}

	}
}

void Rover3DWidget::setYScaleNob(Vector value, bool symetric)
{
	Vector origin = m_SubVolume.getOrigin();
	if (value[1]<origin[1]) {
		setNYScaleNob(value);
		
		if (symetric) {
			value[1] = origin[1] + (origin[1]-value[1]);
			setPYScaleNob(value);
		}
	}
	else if (value[1]>origin[1]) {
		setPYScaleNob(value);

		if (symetric) {
			value[1] = origin[1] - (value[1]-origin[1]);
			setNYScaleNob(value);
		}
	}
}

void Rover3DWidget::setZScaleNob(Vector value, bool symetric)
{
	Vector origin = m_SubVolume.getOrigin();
	if (value[2]<origin[2]) {
		setNZScaleNob(value);
		
		if (symetric) {
			value[2] = origin[2] + (origin[2]-value[2]);
			setPZScaleNob(value);
		}
	}
	else if (value[2]>origin[2]) {
		setPZScaleNob(value);

		if (symetric) {
			value[2] = origin[2] - (value[2]-origin[2]);
			setNZScaleNob(value);
		}
	}
}

void Rover3DWidget::setPXScaleNob(const Vector& value)
{
	m_SubVolume.setXMax(value[0] < m_Boundary.getXMax() ? value[0] : m_Boundary.getXMax());
}

void Rover3DWidget::setPYScaleNob(const Vector& value)
{
	m_SubVolume.setYMax(value[1] < m_Boundary.getYMax() ? value[1] : m_Boundary.getYMax());
}

void Rover3DWidget::setPZScaleNob(const Vector& value)
{
	m_SubVolume.setZMax(value[2] < m_Boundary.getZMax() ? value[2] : m_Boundary.getZMax());
}

void Rover3DWidget::setNXScaleNob(const Vector& value)
{
	m_SubVolume.setXMin(value[0] > m_Boundary.getXMin() ? value[0] : m_Boundary.getXMin());
}

void Rover3DWidget::setNYScaleNob(const Vector& value)
{
	m_SubVolume.setYMin(value[1] > m_Boundary.getYMin() ? value[1] : m_Boundary.getYMin());
}

void Rover3DWidget::setNZScaleNob(const Vector& value)
{
	m_SubVolume.setZMin(value[2] > m_Boundary.getZMin() ? value[2] : m_Boundary.getZMin());
}

void Rover3DWidget::setAspectRatio(double x, double y, double z)
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
	m_SubVolume.clampTo(m_Boundary);

	prepareGeometry();
}

void Rover3DWidget::setColor(float r, float g, float b)
{
	m_R = r;
	m_G = g;
	m_B = b;

	prepareWireCubes();
}

void Rover3DWidget::roverDown(Axis3DHandler::Axis axis)
{
	setCurrentHighlight(axis);
	prepareGeometry();
}

void Rover3DWidget::roverMoving(Axis3DHandler::Axis axis)
{
	setCurrentHighlight(axis);
	prepareGeometry();
	emit RoverExploring();
}

void Rover3DWidget::roverReleased(Axis3DHandler::Axis axis)
{
	m_CurrentHighlight = NoAxis;
	prepareGeometry();
	emit RoverReleased();
}

GeometryRenderable* Rover3DWidget::getWireCubes()
{
	return &m_WireCubes;
}

GeometryRenderable* Rover3DWidget::getAxes()
{
	return &m_Axes;
}

Mouse3DHandler* Rover3DWidget::get3DHandler()
{
	return &m_Handler;
}

MouseHandler* Rover3DWidget::getHandler()
{
	return &m_Adapter;
}

void Rover3DWidget::setCurrentHighlight(Axis3DHandler::Axis axis)
{
	if (axis==Axis3DHandler::XAxis)
		m_CurrentHighlight = XAxis;
	else if (axis==Axis3DHandler::YAxis)
		m_CurrentHighlight = YAxis;
	else 
		m_CurrentHighlight = ZAxis;
}

void Rover3DWidget::prepareGeometry()
{
	prepareAxes();
	prepareWireCubes();
	m_GeometriesAllocated = true;
}

void Rover3DWidget::prepareAxes()
{
	if (!m_GeometriesAllocated) {
		m_AxesGeometry.AllocateLines(10, 6);
		m_AxesGeometry.AllocateLineColors();
		//m_AxesGeometry.AllocatePoints(6); //Since the geometry class now makes m_Points and m_LineVerts the same, this call was breaking things (reducing num points to 6 from 10)
		//TODO: FIX THIS!!!!!!!!!
	}
	m_AxesGeometry.m_LineWidth = 2;

	double CenterX = (m_SubVolume.getXMax() + m_SubVolume.getXMin()) / 2.0;
	double CenterY = (m_SubVolume.getYMax() + m_SubVolume.getYMin()) / 2.0;
	double CenterZ = (m_SubVolume.getZMax() + m_SubVolume.getZMin()) / 2.0;

	// size nobds first
	m_AxesGeometry.m_Points[0*3+0] = (float) m_SubVolume.getXMin();	// point 0 x axis
	m_AxesGeometry.m_Points[0*3+1] = (float) CenterY;
	m_AxesGeometry.m_Points[0*3+2] = (float) CenterZ;

	m_AxesGeometry.m_Points[1*3+0] = (float) m_SubVolume.getXMax();	// point 1 x axis
	m_AxesGeometry.m_Points[1*3+1] = (float) CenterY;
	m_AxesGeometry.m_Points[1*3+2] = (float) CenterZ;

	m_AxesGeometry.m_Points[2*3+0] = (float) CenterX;	// point 2 y axis
	m_AxesGeometry.m_Points[2*3+1] = (float) m_SubVolume.getYMin();
	m_AxesGeometry.m_Points[2*3+2] = (float) CenterZ;

	m_AxesGeometry.m_Points[3*3+0] = (float) CenterX;	// point 3 y axis
	m_AxesGeometry.m_Points[3*3+1] = (float) m_SubVolume.getYMax();
	m_AxesGeometry.m_Points[3*3+2] = (float) CenterZ;

	m_AxesGeometry.m_Points[4*3+0] = (float) CenterX;	// point 4 z axis
	m_AxesGeometry.m_Points[4*3+1] = (float) CenterY;
	m_AxesGeometry.m_Points[4*3+2] = (float) m_SubVolume.getZMin();

	m_AxesGeometry.m_Points[5*3+0] = (float) CenterX;	// point 4 z axis
	m_AxesGeometry.m_Points[5*3+1] = (float) CenterY;
	m_AxesGeometry.m_Points[5*3+2] = (float) m_SubVolume.getZMax();



	m_AxesGeometry.m_LineVerts[0*3+0] = (float) CenterX;	// point 0 x axis
	m_AxesGeometry.m_LineVerts[0*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[0*3+2] = (float) CenterZ;
	if (m_CurrentHighlight==XAxis) {
		m_AxesGeometry.m_LineColors[0*3+0] = 0.0;
		m_AxesGeometry.m_LineColors[0*3+1] = 1.0;
		m_AxesGeometry.m_LineColors[0*3+2] = 1.0;
	}
	else {
		m_AxesGeometry.m_LineColors[0*3+0] = 1.0;
		m_AxesGeometry.m_LineColors[0*3+1] = 0.0;
		m_AxesGeometry.m_LineColors[0*3+2] = 0.0;
	}

	m_AxesGeometry.m_LineVerts[1*3+0] = (float) m_SubVolume.getXMax();	// point 1 x axis
	m_AxesGeometry.m_LineVerts[1*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[1*3+2] = (float) CenterZ;
	if (m_CurrentHighlight==XAxis) {
		m_AxesGeometry.m_LineColors[1*3+0] = 0.0;
		m_AxesGeometry.m_LineColors[1*3+1] = 1.0;
		m_AxesGeometry.m_LineColors[1*3+2] = 1.0;
	}
	else {
		m_AxesGeometry.m_LineColors[1*3+0] = 1.0;
		m_AxesGeometry.m_LineColors[1*3+1] = 0.0;
		m_AxesGeometry.m_LineColors[1*3+2] = 0.0;
	}

	m_AxesGeometry.m_LineVerts[2*3+0] = (float) CenterX;	// point 2 y axis
	m_AxesGeometry.m_LineVerts[2*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[2*3+2] = (float) CenterZ;
	if (m_CurrentHighlight==YAxis) {
		m_AxesGeometry.m_LineColors[2*3+0] = 1.0;
		m_AxesGeometry.m_LineColors[2*3+1] = 0.0;
		m_AxesGeometry.m_LineColors[2*3+2] = 1.0;
	}
	else {
		m_AxesGeometry.m_LineColors[2*3+0] = 0.0;
		m_AxesGeometry.m_LineColors[2*3+1] = 1.0;
		m_AxesGeometry.m_LineColors[2*3+2] = 0.0;
	}

	m_AxesGeometry.m_LineVerts[3*3+0] = (float) CenterX;	// point 3 y axis
	m_AxesGeometry.m_LineVerts[3*3+1] = (float) m_SubVolume.getYMax();
	m_AxesGeometry.m_LineVerts[3*3+2] = (float) CenterZ;
	if (m_CurrentHighlight==YAxis) {
		m_AxesGeometry.m_LineColors[3*3+0] = 1.0;
		m_AxesGeometry.m_LineColors[3*3+1] = 0.0;
		m_AxesGeometry.m_LineColors[3*3+2] = 1.0;
	}
	else {
		m_AxesGeometry.m_LineColors[3*3+0] = 0.0;
		m_AxesGeometry.m_LineColors[3*3+1] = 1.0;
		m_AxesGeometry.m_LineColors[3*3+2] = 0.0;
	}

	m_AxesGeometry.m_LineVerts[4*3+0] = (float) CenterX;	// point 4 z axis
	m_AxesGeometry.m_LineVerts[4*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[4*3+2] = (float) CenterZ;
	if (m_CurrentHighlight==ZAxis) {
		m_AxesGeometry.m_LineColors[4*3+0] = 1.0;
		m_AxesGeometry.m_LineColors[4*3+1] = 1.0;
		m_AxesGeometry.m_LineColors[4*3+2] = 0.0;
	}
	else {
		m_AxesGeometry.m_LineColors[4*3+0] = 0.0;
		m_AxesGeometry.m_LineColors[4*3+1] = 0.0;
		m_AxesGeometry.m_LineColors[4*3+2] = 1.0;
	}

	m_AxesGeometry.m_LineVerts[5*3+0] = (float) CenterX;	// point 5 z axis
	m_AxesGeometry.m_LineVerts[5*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[5*3+2] = (float) m_SubVolume.getZMax();
	if (m_CurrentHighlight==ZAxis) {
		m_AxesGeometry.m_LineColors[5*3+0] = 1.0;
		m_AxesGeometry.m_LineColors[5*3+1] = 1.0;
		m_AxesGeometry.m_LineColors[5*3+2] = 0.0;
	}
	else {
		m_AxesGeometry.m_LineColors[5*3+0] = 0.0;
		m_AxesGeometry.m_LineColors[5*3+1] = 0.0;
		m_AxesGeometry.m_LineColors[5*3+2] = 1.0;
	}

	// grey part of axes
	m_AxesGeometry.m_LineVerts[6*3+0] = (float) CenterX;	// point 6 center
	m_AxesGeometry.m_LineVerts[6*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[6*3+2] = (float) CenterZ;

	m_AxesGeometry.m_LineColors[6*3+0] = 0.5;
	m_AxesGeometry.m_LineColors[6*3+1] = 0.5;
	m_AxesGeometry.m_LineColors[6*3+2] = 0.5;

	m_AxesGeometry.m_LineVerts[7*3+0] = (float) m_SubVolume.getXMin();	// point 7 x axis
	m_AxesGeometry.m_LineVerts[7*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[7*3+2] = (float) CenterZ;

	m_AxesGeometry.m_LineColors[7*3+0] = 0.5;
	m_AxesGeometry.m_LineColors[7*3+1] = 0.5;
	m_AxesGeometry.m_LineColors[7*3+2] = 0.5;

	m_AxesGeometry.m_LineVerts[8*3+0] = (float) CenterX;	// point 8 y axis
	m_AxesGeometry.m_LineVerts[8*3+1] = (float) m_SubVolume.getYMin();
	m_AxesGeometry.m_LineVerts[8*3+2] = (float) CenterZ;

	m_AxesGeometry.m_LineColors[8*3+0] = 0.5;
	m_AxesGeometry.m_LineColors[8*3+1] = 0.5;
	m_AxesGeometry.m_LineColors[8*3+2] = 0.5;

	m_AxesGeometry.m_LineVerts[9*3+0] = (float) CenterX;	// point 9 z axis
	m_AxesGeometry.m_LineVerts[9*3+1] = (float) CenterY;
	m_AxesGeometry.m_LineVerts[9*3+2] = (float) m_SubVolume.getZMin();

	m_AxesGeometry.m_LineColors[9*3+0] = 0.5;
	m_AxesGeometry.m_LineColors[9*3+1] = 0.5;
	m_AxesGeometry.m_LineColors[9*3+2] = 0.5;

	m_AxesGeometry.m_Lines[0*2+0] = 0; // line 0
	m_AxesGeometry.m_Lines[0*2+1] = 1;
	m_AxesGeometry.m_Lines[1*2+0] = 2; // line 1
	m_AxesGeometry.m_Lines[1*2+1] = 3;
	m_AxesGeometry.m_Lines[2*2+0] = 4; // line 2
	m_AxesGeometry.m_Lines[2*2+1] = 5;
	m_AxesGeometry.m_Lines[3*2+0] = 6; // line 3
	m_AxesGeometry.m_Lines[3*2+1] = 7;
	m_AxesGeometry.m_Lines[4*2+0] = 6; // line 4
	m_AxesGeometry.m_Lines[4*2+1] = 8;
	m_AxesGeometry.m_Lines[5*2+0] = 6; // line 5
	m_AxesGeometry.m_Lines[5*2+1] = 9;

}

void Rover3DWidget::prepareWireCubes()
{
	unsigned int numberoflines = 12*2;
	unsigned int numberoflineverts = 8*2;

	if (!m_GeometriesAllocated) {
		m_WireCubeGeometry.AllocateLines(numberoflineverts, numberoflines);
		m_WireCubeGeometry.AllocateLineColors();
	}

	// the subvolume wirecube
	// make 8 points  zyx
	m_WireCubeGeometry.m_LineVerts[0*3+0] = (float) m_SubVolume.getXMin();	// point 0
	m_WireCubeGeometry.m_LineVerts[0*3+1] = (float) m_SubVolume.getYMin();
	m_WireCubeGeometry.m_LineVerts[0*3+2] = (float) m_SubVolume.getZMin();
	m_WireCubeGeometry.m_LineColors[0*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[0*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[0*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[1*3+0] = (float) m_SubVolume.getXMax();	// point 1
	m_WireCubeGeometry.m_LineVerts[1*3+1] = (float) m_SubVolume.getYMin();
	m_WireCubeGeometry.m_LineVerts[1*3+2] = (float) m_SubVolume.getZMin();
	m_WireCubeGeometry.m_LineColors[1*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[1*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[1*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[2*3+0] = (float) m_SubVolume.getXMin();	// point 2
	m_WireCubeGeometry.m_LineVerts[2*3+1] = (float) m_SubVolume.getYMax();
	m_WireCubeGeometry.m_LineVerts[2*3+2] = (float) m_SubVolume.getZMin();
	m_WireCubeGeometry.m_LineColors[2*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[2*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[2*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[3*3+0] = (float) m_SubVolume.getXMax();	// point 3
	m_WireCubeGeometry.m_LineVerts[3*3+1] = (float) m_SubVolume.getYMax();
	m_WireCubeGeometry.m_LineVerts[3*3+2] = (float) m_SubVolume.getZMin();
	m_WireCubeGeometry.m_LineColors[3*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[3*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[3*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[4*3+0] = (float) m_SubVolume.getXMin();	// point 4
	m_WireCubeGeometry.m_LineVerts[4*3+1] = (float) m_SubVolume.getYMin();
	m_WireCubeGeometry.m_LineVerts[4*3+2] = (float) m_SubVolume.getZMax();
	m_WireCubeGeometry.m_LineColors[4*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[4*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[4*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[5*3+0] = (float) m_SubVolume.getXMax();	// point 5
	m_WireCubeGeometry.m_LineVerts[5*3+1] = (float) m_SubVolume.getYMin();
	m_WireCubeGeometry.m_LineVerts[5*3+2] = (float) m_SubVolume.getZMax();
	m_WireCubeGeometry.m_LineColors[5*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[5*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[5*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[6*3+0] = (float) m_SubVolume.getXMin();	// point 6
	m_WireCubeGeometry.m_LineVerts[6*3+1] = (float) m_SubVolume.getYMax();
	m_WireCubeGeometry.m_LineVerts[6*3+2] = (float) m_SubVolume.getZMax();
	m_WireCubeGeometry.m_LineColors[6*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[6*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[6*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[7*3+0] = (float) m_SubVolume.getXMax();	// point 7
	m_WireCubeGeometry.m_LineVerts[7*3+1] = (float) m_SubVolume.getYMax();
	m_WireCubeGeometry.m_LineVerts[7*3+2] = (float) m_SubVolume.getZMax();
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

	// the boundary wirecube
	// make 8 points  zyx
	m_WireCubeGeometry.m_LineVerts[8*3+0] = (float) m_Boundary.getXMin();	// point 0
	m_WireCubeGeometry.m_LineVerts[8*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[8*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[8*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[8*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[8*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[9*3+0] = (float) m_Boundary.getXMax();	// point 1
	m_WireCubeGeometry.m_LineVerts[9*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[9*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[9*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[9*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[9*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[10*3+0] = (float) m_Boundary.getXMin();	// point 2
	m_WireCubeGeometry.m_LineVerts[10*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[10*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[10*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[10*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[10*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[11*3+0] = (float) m_Boundary.getXMax();	// point 3
	m_WireCubeGeometry.m_LineVerts[11*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[11*3+2] = (float) m_Boundary.getZMin();
	m_WireCubeGeometry.m_LineColors[11*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[11*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[11*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[12*3+0] = (float) m_Boundary.getXMin();	// point 4
	m_WireCubeGeometry.m_LineVerts[12*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[12*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[12*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[12*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[12*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[13*3+0] = (float) m_Boundary.getXMax();	// point 5
	m_WireCubeGeometry.m_LineVerts[13*3+1] = (float) m_Boundary.getYMin();
	m_WireCubeGeometry.m_LineVerts[13*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[13*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[13*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[13*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[14*3+0] = (float) m_Boundary.getXMin();	// point 6
	m_WireCubeGeometry.m_LineVerts[14*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[14*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[14*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[14*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[14*3+2] = m_B;

	m_WireCubeGeometry.m_LineVerts[15*3+0] = (float) m_Boundary.getXMax();	// point 7
	m_WireCubeGeometry.m_LineVerts[15*3+1] = (float) m_Boundary.getYMax();
	m_WireCubeGeometry.m_LineVerts[15*3+2] = (float) m_Boundary.getZMax();
	m_WireCubeGeometry.m_LineColors[15*3+0] = m_R;
	m_WireCubeGeometry.m_LineColors[15*3+1] = m_G;
	m_WireCubeGeometry.m_LineColors[15*3+2] = m_B;

	// make 12 lines
	m_WireCubeGeometry.m_Lines[12*2+0] = 0+8; // line 0
	m_WireCubeGeometry.m_Lines[12*2+1] = 2+8;
	m_WireCubeGeometry.m_Lines[13*2+0] = 2+8; // line 1
	m_WireCubeGeometry.m_Lines[13*2+1] = 3+8;
	m_WireCubeGeometry.m_Lines[14*2+0] = 3+8; // line 2
	m_WireCubeGeometry.m_Lines[14*2+1] = 1+8;
	m_WireCubeGeometry.m_Lines[15*2+0] = 1+8; // line 3
	m_WireCubeGeometry.m_Lines[15*2+1] = 0+8;

	m_WireCubeGeometry.m_Lines[16*2+0] = 4+8; // line 4
	m_WireCubeGeometry.m_Lines[16*2+1] = 6+8;
	m_WireCubeGeometry.m_Lines[17*2+0] = 6+8; // line 5
	m_WireCubeGeometry.m_Lines[17*2+1] = 7+8;
	m_WireCubeGeometry.m_Lines[18*2+0] = 7+8; // line 6
	m_WireCubeGeometry.m_Lines[18*2+1] = 5+8;
	m_WireCubeGeometry.m_Lines[19*2+0] = 5+8; // line 7
	m_WireCubeGeometry.m_Lines[19*2+1] = 4+8;

	m_WireCubeGeometry.m_Lines[20*2+0] = 0+8; // line 8
	m_WireCubeGeometry.m_Lines[20*2+1] = 4+8;
	m_WireCubeGeometry.m_Lines[21*2+0] = 2+8; // line 9
	m_WireCubeGeometry.m_Lines[21*2+1] = 6+8;
	m_WireCubeGeometry.m_Lines[22*2+0] = 3+8; // line 10
	m_WireCubeGeometry.m_Lines[22*2+1] = 7+8;
	m_WireCubeGeometry.m_Lines[23*2+0] = 1+8; // line 11
	m_WireCubeGeometry.m_Lines[23*2+1] = 5+8;
}

void Rover3DWidget::prepareHandler()
{
	Axis3DHandler xAxis(this, Axis3DHandler::XAxis);
	Axis3DHandler yAxis(this, Axis3DHandler::YAxis);
	Axis3DHandler zAxis(this, Axis3DHandler::ZAxis);
	m_Handler.add(&xAxis);
	m_Handler.add(&yAxis);
	m_Handler.add(&zAxis);
}

