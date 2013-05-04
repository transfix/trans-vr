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

// WireCube.cpp: implementation of the WireCube class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/WireCube.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

WireCube::WireCube()
{
	setExtents(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5);
	m_bShowAxes = false;
	m_WhichHilight = NoAxis;
}

WireCube::WireCube(
		double xMin, double xMax,
		double yMin, double yMax,
		double zMin, double zMax
		)
{
	setExtents(
		xMin, xMax,
		yMin, yMax,
		zMin, zMax);
	m_bShowAxes = false;
	m_WhichHilight = NoAxis;
}

WireCube::~WireCube()
{
}

void WireCube::setExtents(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
{
	m_bOptionsChanged = true;
	m_XMin = xMin; m_XMax = xMax;
	m_YMin = yMin; m_YMax = yMax;
	m_ZMin = zMin; m_ZMax = zMax;
}

Vector WireCube::getOrigin() const
{
	return Vector(
		(float)((m_XMin + m_XMax)/2.0),
		(float)((m_YMin + m_YMax)/2.0),
		(float)((m_ZMin + m_ZMax)/2.0),
		1.0f
		);
}

void WireCube::setOrigin(Vector vector, const WireCube& boundary)
{
	double width = (m_XMax-m_XMin)/2.0;
	double height = (m_YMax-m_YMin)/2.0;
	double depth = (m_ZMax-m_ZMin)/2.0;
	vector[3] = 1.0;
	//Vector dir = vector-getOrigin();
	if (vector[0]+width > boundary.m_XMax) {
		vector[0] = (float)(boundary.m_XMax-width);
	}
	else if (vector[0]-width < boundary.m_XMin) {
		vector[0] = (float)(boundary.m_XMin+width);
	}

	if (vector[1]+height > boundary.m_YMax) {
		vector[1] = (float)(boundary.m_YMax-height);
	}
	else if (vector[1]-height < boundary.m_YMin) {
		vector[1] = (float)(boundary.m_YMin+height);
	}

	if (vector[2]+depth > boundary.m_ZMax) {
		vector[2] = (float)(boundary.m_ZMax-depth);
	}
	else if (vector[2]-depth < boundary.m_ZMin) {
		vector[2] = (float)(boundary.m_ZMin+depth);
	}

	m_XMin = vector[0]-width;
	m_YMin = vector[1]-height;
	m_ZMin = vector[2]-depth;
	m_XMax = vector[0]+width;
	m_YMax = vector[1]+height;
	m_ZMax = vector[2]+depth;
	m_bOptionsChanged = true;
}

void WireCube::move(const Vector& vector)
{
	m_XMin += vector[0];
	m_YMin += vector[1];
	m_ZMin += vector[2];
	m_XMax += vector[0];
	m_YMax += vector[1];
	m_ZMax += vector[2];
	m_bOptionsChanged = true;
}

void WireCube::showAxes()
{
	m_bOptionsChanged = true;
	m_bShowAxes = true;
}

void WireCube::hideAxes()
{
	m_bOptionsChanged = true;
	m_bShowAxes = false;
}

void WireCube::toggleAxes()
{
	m_bOptionsChanged = true;
	if (m_bShowAxes) {
		m_bShowAxes = false;
	}
	else {
		m_bShowAxes = true;
	}
}

bool WireCube::AxesVisible()
{
	return m_bShowAxes;
}

Geometry* WireCube::getGeometry()
{
	if (m_bOptionsChanged) {
		prepareGeometry();
	}

	return &m_Geometry;
}

Geometry* WireCube::getAxes()
{
	if (m_bOptionsChanged) {
		prepareGeometry();
	}

	return &m_Axes;
}

bool WireCube::withinCube(const Vector& vector)
{
	return 
		vector[0]>=m_XMin && vector[0]<=m_XMax &&
		vector[1]>=m_YMin && vector[1]<=m_YMax &&
		vector[2]>=m_ZMin && vector[2]<=m_ZMax;
		
}

void WireCube::clampTo(const WireCube& boundaryCube)
{
	/* decided to go the easy route
	m_XMax = (m_XMax<boundaryCube.m_XMax?m_XMax:boundaryCube.m_XMax);
	m_YMax = (m_YMax<boundaryCube.m_YMax?m_YMax:boundaryCube.m_YMax);
	m_ZMax = (m_ZMax<boundaryCube.m_ZMax?m_ZMax:boundaryCube.m_ZMax);
	m_XMin = (m_XMin<boundaryCube.m_XMax?m_XMin:boundaryCube.m_XMin);
	m_YMin = (m_YMin<boundaryCube.m_YMax?m_YMin:boundaryCube.m_YMin);
	m_ZMin = (m_ZMin<boundaryCube.m_ZMax?m_ZMin:boundaryCube.m_ZMin);

	m_XMin = (
	*/

	m_XMax = boundaryCube.m_XMax;
	m_YMax = boundaryCube.m_YMax;
	m_ZMax = boundaryCube.m_ZMax;
	m_XMin = boundaryCube.m_XMin;
	m_YMin = boundaryCube.m_YMin;
	m_ZMin = boundaryCube.m_ZMin;
}

void WireCube::hilightX()
{
	if (m_WhichHilight!=XAxis) {
		m_bOptionsChanged = true;
	}
	m_WhichHilight = XAxis;
}

void WireCube::hilightY()
{
	if (m_WhichHilight!=YAxis) {
		m_bOptionsChanged = true;
	}
	m_WhichHilight = YAxis;
}

void WireCube::hilightZ()
{
	if (m_WhichHilight!=ZAxis) {
		m_bOptionsChanged = true;
	}
	m_WhichHilight = ZAxis;
}

void WireCube::noHilight()
{
	if (m_WhichHilight!=NoAxis) {
		m_bOptionsChanged = true;
	}
	m_WhichHilight = NoAxis;
}

Vector WireCube::getPXScaleNob() const
{
	return Vector(
		(float)m_XMax,
		(float)((m_YMin + m_YMax)/2.0),
		(float)((m_ZMin + m_ZMax)/2.0),
		1.0f
		);
}

Vector WireCube::getPYScaleNob() const
{
	return Vector(
		(float)((m_XMin + m_XMax)/2.0),
		(float)m_YMax,
		(float)((m_ZMin + m_ZMax)/2.0),
		1.0f
		);
}

Vector WireCube::getPZScaleNob() const
{
	return Vector(
		(float)((m_XMin + m_XMax)/2.0),
		(float)((m_YMin + m_YMax)/2.0),
		(float)m_ZMax,
		1.0f
		);
}

Vector WireCube::getNXScaleNob() const
{
	return Vector(
		(float)m_XMin,
		(float)((m_YMin + m_YMax)/2.0),
		(float)((m_ZMin + m_ZMax)/2.0),
		1.0f
		);
}

Vector WireCube::getNYScaleNob() const
{
	return Vector(
		(float)((m_XMin + m_XMax)/2.0),
		(float)m_YMin,
		(float)((m_ZMin + m_ZMax)/2.0),
		1.0f
		);
}

Vector WireCube::getNZScaleNob() const
{
	return Vector(
		(float)((m_XMin + m_XMax)/2.0),
		(float)((m_YMin + m_YMax)/2.0),
		(float)m_ZMin,
		1.0f
		);
}

void WireCube::setXScaleNob(Vector value, const WireCube& boundary, bool symetric)
{
	Vector origin = getOrigin();
	if (value[0]<origin[0]) {
		setNXScaleNob(value, boundary);

		if (symetric) {
			value[0] = origin[0] + (origin[0]-value[0]);
			setPXScaleNob(value, boundary);
		}
	}
	else if (value[0]>origin[0]) {
		setPXScaleNob(value, boundary);

		if (symetric) {
			value[0] = origin[0] - (value[0]-origin[0]);
			setNXScaleNob(value, boundary);
		}

	}
}

void WireCube::setYScaleNob(Vector value, const WireCube& boundary, bool symetric)
{
	Vector origin = getOrigin();
	if (value[1]<origin[1]) {
		setNYScaleNob(value, boundary);
		
		if (symetric) {
			value[1] = origin[1] + (origin[1]-value[1]);
			setPYScaleNob(value, boundary);
		}
	}
	else if (value[1]>origin[1]) {
		setPYScaleNob(value, boundary);

		if (symetric) {
			value[1] = origin[1] - (value[1]-origin[1]);
			setNYScaleNob(value, boundary);
		}
	}
}

void WireCube::setZScaleNob(Vector value, const WireCube& boundary, bool symetric)
{
	Vector origin = getOrigin();
	if (value[2]<origin[2]) {
		setNZScaleNob(value, boundary);
		
		if (symetric) {
			value[2] = origin[2] + (origin[2]-value[2]);
			setPZScaleNob(value, boundary);
		}
	}
	else if (value[2]>origin[2]) {
		setPZScaleNob(value, boundary);

		if (symetric) {
			value[2] = origin[2] - (value[2]-origin[2]);
			setNZScaleNob(value, boundary);
		}
	}
}

void WireCube::setPXScaleNob(const Vector& value, const WireCube& boundary)
{
	m_XMax = (value[0] < boundary.m_XMax ? value[0] : boundary.m_XMax);
	m_bOptionsChanged = true;
}

void WireCube::setPYScaleNob(const Vector& value, const WireCube& boundary)
{
	m_YMax = (value[1] < boundary.m_YMax ? value[1] : boundary.m_YMax);
	m_bOptionsChanged = true;
}

void WireCube::setPZScaleNob(const Vector& value, const WireCube& boundary)
{
	m_ZMax = (value[2] < boundary.m_ZMax ? value[2] : boundary.m_ZMax);
	m_bOptionsChanged = true;
}

void WireCube::setNXScaleNob(const Vector& value, const WireCube& boundary)
{
	m_XMin = (value[0] > boundary.m_XMin ? value[0] : boundary.m_XMin);
	m_bOptionsChanged = true;
}

void WireCube::setNYScaleNob(const Vector& value, const WireCube& boundary)
{
	m_YMin = (value[1] > boundary.m_YMin ? value[1] : boundary.m_YMin);
	m_bOptionsChanged = true;
}

void WireCube::setNZScaleNob(const Vector& value, const WireCube& boundary)
{
	m_ZMin = (value[2] > boundary.m_ZMin ? value[2] : boundary.m_ZMin);
	m_bOptionsChanged = true;
}

void WireCube::prepareGeometry()
{
	unsigned int numberoflines = 12;
	unsigned int numberoflineverts = 8;

	/*if (m_bShowAxes) {
		m_Geometry.AllocateLines(numberoflineverts+6, numberoflines+3);
	}
	else {

	}*/
	m_Geometry.AllocateLines(numberoflineverts, numberoflines);
	m_Geometry.AllocateLineColors();

	// make 8 points  zyx
	m_Geometry.m_LineVerts[0*3+0] = (float) m_XMin;	// point 0
	m_Geometry.m_LineVerts[0*3+1] = (float) m_YMin;
	m_Geometry.m_LineVerts[0*3+2] = (float) m_ZMin;
	m_Geometry.m_LineColors[0*3+0] = 1.0;
	m_Geometry.m_LineColors[0*3+1] = 1.0;
	m_Geometry.m_LineColors[0*3+2] = 1.0;

	m_Geometry.m_LineVerts[1*3+0] = (float) m_XMax;	// point 1
	m_Geometry.m_LineVerts[1*3+1] = (float) m_YMin;
	m_Geometry.m_LineVerts[1*3+2] = (float) m_ZMin;
	m_Geometry.m_LineColors[1*3+0] = 1.0;
	m_Geometry.m_LineColors[1*3+1] = 1.0;
	m_Geometry.m_LineColors[1*3+2] = 1.0;

	m_Geometry.m_LineVerts[2*3+0] = (float) m_XMin;	// point 2
	m_Geometry.m_LineVerts[2*3+1] = (float) m_YMax;
	m_Geometry.m_LineVerts[2*3+2] = (float) m_ZMin;
	m_Geometry.m_LineColors[2*3+0] = 1.0;
	m_Geometry.m_LineColors[2*3+1] = 1.0;
	m_Geometry.m_LineColors[2*3+2] = 1.0;

	m_Geometry.m_LineVerts[3*3+0] = (float) m_XMax;	// point 3
	m_Geometry.m_LineVerts[3*3+1] = (float) m_YMax;
	m_Geometry.m_LineVerts[3*3+2] = (float) m_ZMin;
	m_Geometry.m_LineColors[3*3+0] = 1.0;
	m_Geometry.m_LineColors[3*3+1] = 1.0;
	m_Geometry.m_LineColors[3*3+2] = 1.0;

	m_Geometry.m_LineVerts[4*3+0] = (float) m_XMin;	// point 4
	m_Geometry.m_LineVerts[4*3+1] = (float) m_YMin;
	m_Geometry.m_LineVerts[4*3+2] = (float) m_ZMax;
	m_Geometry.m_LineColors[4*3+0] = 1.0;
	m_Geometry.m_LineColors[4*3+1] = 1.0;
	m_Geometry.m_LineColors[4*3+2] = 1.0;

	m_Geometry.m_LineVerts[5*3+0] = (float) m_XMax;	// point 5
	m_Geometry.m_LineVerts[5*3+1] = (float) m_YMin;
	m_Geometry.m_LineVerts[5*3+2] = (float) m_ZMax;
	m_Geometry.m_LineColors[5*3+0] = 1.0;
	m_Geometry.m_LineColors[5*3+1] = 1.0;
	m_Geometry.m_LineColors[5*3+2] = 1.0;

	m_Geometry.m_LineVerts[6*3+0] = (float) m_XMin;	// point 6
	m_Geometry.m_LineVerts[6*3+1] = (float) m_YMax;
	m_Geometry.m_LineVerts[6*3+2] = (float) m_ZMax;
	m_Geometry.m_LineColors[6*3+0] = 1.0;
	m_Geometry.m_LineColors[6*3+1] = 1.0;
	m_Geometry.m_LineColors[6*3+2] = 1.0;

	m_Geometry.m_LineVerts[7*3+0] = (float) m_XMax;	// point 7
	m_Geometry.m_LineVerts[7*3+1] = (float) m_YMax;
	m_Geometry.m_LineVerts[7*3+2] = (float) m_ZMax;
	m_Geometry.m_LineColors[7*3+0] = 1.0;
	m_Geometry.m_LineColors[7*3+1] = 1.0;
	m_Geometry.m_LineColors[7*3+2] = 1.0;

	// make 12 lines
	m_Geometry.m_Lines[0*2+0] = 0; // line 0
	m_Geometry.m_Lines[0*2+1] = 2;
	m_Geometry.m_Lines[1*2+0] = 2; // line 1
	m_Geometry.m_Lines[1*2+1] = 3;
	m_Geometry.m_Lines[2*2+0] = 3; // line 2
	m_Geometry.m_Lines[2*2+1] = 1;
	m_Geometry.m_Lines[3*2+0] = 1; // line 3
	m_Geometry.m_Lines[3*2+1] = 0;

	m_Geometry.m_Lines[4*2+0] = 4; // line 4
	m_Geometry.m_Lines[4*2+1] = 6;
	m_Geometry.m_Lines[5*2+0] = 6; // line 5
	m_Geometry.m_Lines[5*2+1] = 7;
	m_Geometry.m_Lines[6*2+0] = 7; // line 6
	m_Geometry.m_Lines[6*2+1] = 5;
	m_Geometry.m_Lines[7*2+0] = 5; // line 7
	m_Geometry.m_Lines[7*2+1] = 4;

	m_Geometry.m_Lines[8*2+0] = 0; // line 8
	m_Geometry.m_Lines[8*2+1] = 4;
	m_Geometry.m_Lines[9*2+0] = 2; // line 9
	m_Geometry.m_Lines[9*2+1] = 6;
	m_Geometry.m_Lines[10*2+0] = 3; // line 10
	m_Geometry.m_Lines[10*2+1] = 7;
	m_Geometry.m_Lines[11*2+0] = 1; // line 11
	m_Geometry.m_Lines[11*2+1] = 5;

	if (m_bShowAxes) {
		prepareAxes();
	}
	m_bOptionsChanged = false;
}

void WireCube::prepareAxes()
{
	m_Axes.AllocateLines(10, 6);
	m_Axes.AllocateLineColors();
	m_Axes.m_LineWidth = 2;
	m_Axes.AllocatePoints(6);

	double CenterX = (m_XMax + m_XMin) / 2.0;
	double CenterY = (m_YMax + m_YMin) / 2.0;
	double CenterZ = (m_ZMax + m_ZMin) / 2.0;
	if (m_bShowAxes) {
		// size nobds first
		m_Axes.m_Points[0*3+0] = (float) m_XMin;	// point 0 x axis
		m_Axes.m_Points[0*3+1] = (float) CenterY;
		m_Axes.m_Points[0*3+2] = (float) CenterZ;

		m_Axes.m_Points[1*3+0] = (float) m_XMax;	// point 1 x axis
		m_Axes.m_Points[1*3+1] = (float) CenterY;
		m_Axes.m_Points[1*3+2] = (float) CenterZ;

		m_Axes.m_Points[2*3+0] = (float) CenterX;	// point 2 y axis
		m_Axes.m_Points[2*3+1] = (float) m_YMin;
		m_Axes.m_Points[2*3+2] = (float) CenterZ;

		m_Axes.m_Points[3*3+0] = (float) CenterX;	// point 3 y axis
		m_Axes.m_Points[3*3+1] = (float) m_YMax;
		m_Axes.m_Points[3*3+2] = (float) CenterZ;

		m_Axes.m_Points[4*3+0] = (float) CenterX;	// point 4 z axis
		m_Axes.m_Points[4*3+1] = (float) CenterY;
		m_Axes.m_Points[4*3+2] = (float) m_ZMin;

		m_Axes.m_Points[5*3+0] = (float) CenterX;	// point 4 z axis
		m_Axes.m_Points[5*3+1] = (float) CenterY;
		m_Axes.m_Points[5*3+2] = (float) m_ZMax;




		m_Axes.m_LineVerts[0*3+0] = (float) CenterX;	// point 0 x axis
		m_Axes.m_LineVerts[0*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[0*3+2] = (float) CenterZ;
		if (m_WhichHilight==XAxis) {
			m_Axes.m_LineColors[0*3+0] = 0.0;
			m_Axes.m_LineColors[0*3+1] = 1.0;
			m_Axes.m_LineColors[0*3+2] = 1.0;
		}
		else {
			m_Axes.m_LineColors[0*3+0] = 1.0;
			m_Axes.m_LineColors[0*3+1] = 0.0;
			m_Axes.m_LineColors[0*3+2] = 0.0;
		}

		m_Axes.m_LineVerts[1*3+0] = (float) m_XMax;	// point 1 x axis
		m_Axes.m_LineVerts[1*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[1*3+2] = (float) CenterZ;
		if (m_WhichHilight==XAxis) {
			m_Axes.m_LineColors[1*3+0] = 0.0;
			m_Axes.m_LineColors[1*3+1] = 1.0;
			m_Axes.m_LineColors[1*3+2] = 1.0;
		}
		else {
			m_Axes.m_LineColors[1*3+0] = 1.0;
			m_Axes.m_LineColors[1*3+1] = 0.0;
			m_Axes.m_LineColors[1*3+2] = 0.0;
		}

		m_Axes.m_LineVerts[2*3+0] = (float) CenterX;	// point 2 y axis
		m_Axes.m_LineVerts[2*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[2*3+2] = (float) CenterZ;
		if (m_WhichHilight==YAxis) {
			m_Axes.m_LineColors[2*3+0] = 1.0;
			m_Axes.m_LineColors[2*3+1] = 0.0;
			m_Axes.m_LineColors[2*3+2] = 1.0;
		}
		else {
			m_Axes.m_LineColors[2*3+0] = 0.0;
			m_Axes.m_LineColors[2*3+1] = 1.0;
			m_Axes.m_LineColors[2*3+2] = 0.0;
		}

		m_Axes.m_LineVerts[3*3+0] = (float) CenterX;	// point 3 y axis
		m_Axes.m_LineVerts[3*3+1] = (float) m_YMax;
		m_Axes.m_LineVerts[3*3+2] = (float) CenterZ;
		if (m_WhichHilight==YAxis) {
			m_Axes.m_LineColors[3*3+0] = 1.0;
			m_Axes.m_LineColors[3*3+1] = 0.0;
			m_Axes.m_LineColors[3*3+2] = 1.0;
		}
		else {
			m_Axes.m_LineColors[3*3+0] = 0.0;
			m_Axes.m_LineColors[3*3+1] = 1.0;
			m_Axes.m_LineColors[3*3+2] = 0.0;
		}

		m_Axes.m_LineVerts[4*3+0] = (float) CenterX;	// point 4 z axis
		m_Axes.m_LineVerts[4*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[4*3+2] = (float) CenterZ;
		if (m_WhichHilight==ZAxis) {
			m_Axes.m_LineColors[4*3+0] = 1.0;
			m_Axes.m_LineColors[4*3+1] = 1.0;
			m_Axes.m_LineColors[4*3+2] = 0.0;
		}
		else {
			m_Axes.m_LineColors[4*3+0] = 0.0;
			m_Axes.m_LineColors[4*3+1] = 0.0;
			m_Axes.m_LineColors[4*3+2] = 1.0;
		}

		m_Axes.m_LineVerts[5*3+0] = (float) CenterX;	// point 5 z axis
		m_Axes.m_LineVerts[5*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[5*3+2] = (float) m_ZMax;
		if (m_WhichHilight==ZAxis) {
			m_Axes.m_LineColors[5*3+0] = 1.0;
			m_Axes.m_LineColors[5*3+1] = 1.0;
			m_Axes.m_LineColors[5*3+2] = 0.0;
		}
		else {
			m_Axes.m_LineColors[5*3+0] = 0.0;
			m_Axes.m_LineColors[5*3+1] = 0.0;
			m_Axes.m_LineColors[5*3+2] = 1.0;
		}

		// grey part of axes
		m_Axes.m_LineVerts[6*3+0] = (float) CenterX;	// point 6 center
		m_Axes.m_LineVerts[6*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[6*3+2] = (float) CenterZ;

		m_Axes.m_LineColors[6*3+0] = 0.5;
		m_Axes.m_LineColors[6*3+1] = 0.5;
		m_Axes.m_LineColors[6*3+2] = 0.5;

		m_Axes.m_LineVerts[7*3+0] = (float) m_XMin;	// point 7 x axis
		m_Axes.m_LineVerts[7*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[7*3+2] = (float) CenterZ;

		m_Axes.m_LineColors[7*3+0] = 0.5;
		m_Axes.m_LineColors[7*3+1] = 0.5;
		m_Axes.m_LineColors[7*3+2] = 0.5;

		m_Axes.m_LineVerts[8*3+0] = (float) CenterX;	// point 8 y axis
		m_Axes.m_LineVerts[8*3+1] = (float) m_YMin;
		m_Axes.m_LineVerts[8*3+2] = (float) CenterZ;

		m_Axes.m_LineColors[8*3+0] = 0.5;
		m_Axes.m_LineColors[8*3+1] = 0.5;
		m_Axes.m_LineColors[8*3+2] = 0.5;

		m_Axes.m_LineVerts[9*3+0] = (float) CenterX;	// point 9 z axis
		m_Axes.m_LineVerts[9*3+1] = (float) CenterY;
		m_Axes.m_LineVerts[9*3+2] = (float) m_ZMin;

		m_Axes.m_LineColors[9*3+0] = 0.5;
		m_Axes.m_LineColors[9*3+1] = 0.5;
		m_Axes.m_LineColors[9*3+2] = 0.5;

		m_Axes.m_Lines[0*2+0] = 0; // line 0
		m_Axes.m_Lines[0*2+1] = 1;
		m_Axes.m_Lines[1*2+0] = 2; // line 1
		m_Axes.m_Lines[1*2+1] = 3;
		m_Axes.m_Lines[2*2+0] = 4; // line 2
		m_Axes.m_Lines[2*2+1] = 5;
		m_Axes.m_Lines[3*2+0] = 6; // line 3
		m_Axes.m_Lines[3*2+1] = 7;
		m_Axes.m_Lines[4*2+0] = 6; // line 4
		m_Axes.m_Lines[4*2+1] = 8;
		m_Axes.m_Lines[5*2+0] = 6; // line 5
		m_Axes.m_Lines[5*2+1] = 9;
	}
}

