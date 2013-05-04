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

// Grid.cpp: implementation of the Grid class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/Grid.h>
#include <stdlib.h>
#include <math.h>
#include <VolumeWidget/Ray.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

const float Grid::defaultGridSpacing = 1.0f;
const float Grid::defaultGridExtent = 100.0f;
const unsigned int Grid::defaultMajorSpacing = 5;
const bool Grid::defaultbShowAxis = true;
const float Grid::defaultMajorGridColor[3] = { 0.5f, 0.5f, 0.5f };
const float Grid::defaultMinorGridColor[3] = { 0.3f, 0.3f, 0.3f };
const float Grid::defaultXAxisColor[3] = { 1.0f, 0.0f, 0.0f };
const float Grid::defaultYAxisColor[3] = { 0.0f, 1.0f, 0.0f };
const float Grid::defaultZAxisColor[3] = { 0.0f, 0.0f, 1.0f };

Grid::Grid()
{
	InitOptions();
}

Grid::~Grid()
{

}

void Grid::InitOptions()
{
	setGridTop();

	m_GridExtent = defaultGridExtent;
	m_GridSpacing = defaultGridSpacing;
	m_MajorSpacing = defaultMajorSpacing;
	m_bShowAxis = defaultbShowAxis;
	m_MajorGridColor[0] = defaultMajorGridColor[0]; m_MajorGridColor[1] = defaultMajorGridColor[1]; m_MajorGridColor[2] = defaultMajorGridColor[2];
	m_MinorGridColor[0] = defaultMinorGridColor[0]; m_MinorGridColor[1] = defaultMinorGridColor[1]; m_MinorGridColor[2] = defaultMinorGridColor[2];
	m_XAxisColor[0] = defaultXAxisColor[0]; m_XAxisColor[1] = defaultXAxisColor[1]; m_XAxisColor[2] = defaultXAxisColor[2];
	m_YAxisColor[0] = defaultYAxisColor[0]; m_YAxisColor[1] = defaultYAxisColor[1]; m_YAxisColor[2] = defaultYAxisColor[2];
	m_ZAxisColor[0] = defaultZAxisColor[0]; m_ZAxisColor[1] = defaultZAxisColor[1]; m_ZAxisColor[2] = defaultZAxisColor[2];
	m_bOptionChanged = true;
}

static const double EPS = 0.0001;

Vector Grid::Intersect(const Ray& ray) const
{
	Ray trans(ray);
	trans.m_Origin -= m_Origin;
	trans = m_Orientation.conjugate().applyRotation(trans);

	if (fabs(trans.m_Dir[2])<EPS) {
		return Vector::badVector();
	}

	float t = 0.0f-trans.m_Origin[2]/trans.m_Dir[2];
	if (t<0.0f) {
		return Vector::badVector();
	}

	Vector vec(trans.getPointOnRay(t));
	vec = m_Orientation.applyRotation(vec);
	vec += m_Origin;
	return vec;
}

void Grid::SetGridSize(float size)
{
	if (m_GridExtent!=size) {
		m_GridExtent = size;
		m_bOptionChanged = true;
	}
}

void Grid::SetGridSpacing(float spacing)
{
	if (m_GridSpacing!=spacing) {
		m_GridSpacing = spacing;
		m_bOptionChanged = true;
	}
}

void Grid::SetMajorSpacing(unsigned int majorSpacing)
{
	if (m_MajorSpacing!=majorSpacing) {
		m_MajorSpacing = majorSpacing;
		m_bOptionChanged = true;
	}
}

void Grid::SetShowAxis(bool axis)
{
	if (m_bShowAxis!=axis) {
		m_bShowAxis = axis;
		m_bOptionChanged = true;
	}
}

void Grid::SetMajorGridColor( float r, float g, float b )
{
	if ((m_MajorGridColor[0]!=r) || (m_MajorGridColor[1]!=g) || (m_MajorGridColor[2]!=b)) {
		m_MajorGridColor[0]=r;
		m_MajorGridColor[1]=g;
		m_MajorGridColor[2]=b;
		m_bOptionChanged = true;
	}	
}

void Grid::SetMinorGridColor( float r, float g, float b )
{
	if ((m_MinorGridColor[0]!=r) || (m_MinorGridColor[1]!=g) || (m_MinorGridColor[2]!=b)) {
		m_MinorGridColor[0]=r;
		m_MinorGridColor[1]=g;
		m_MinorGridColor[2]=b;
		m_bOptionChanged = true;
	}
}

void Grid::SetXAxisColor( float r, float g, float b )
{
	if ((m_XAxisColor[0]!=r) || (m_XAxisColor[1]!=g) || (m_XAxisColor[2]!=b)) {
		m_XAxisColor[0]=r;
		m_XAxisColor[1]=g;
		m_XAxisColor[2]=b;
		m_bOptionChanged = true;
	}
}

void Grid::SetYAxisColor( float r, float g, float b )
{
	if ((m_YAxisColor[0]!=r) || (m_YAxisColor[1]!=g) || (m_YAxisColor[2]!=b)) {
		m_YAxisColor[0]=r;
		m_YAxisColor[1]=g;
		m_YAxisColor[2]=b;
		m_bOptionChanged = true;
	}
}

void Grid::SetZAxisColor( float r, float g, float b )
{
	if ((m_ZAxisColor[0]!=r) || (m_ZAxisColor[1]!=g) || (m_ZAxisColor[2]!=b)) {
		m_ZAxisColor[0]=r;
		m_ZAxisColor[1]=g;
		m_ZAxisColor[2]=b;
		m_bOptionChanged = true;
	}
}

float Grid::GetGridSize() const
{
	return m_GridExtent;
}

float Grid::GetGridSpacing() const
{
	return m_GridSpacing;
}

unsigned int Grid::GetMajorSpacing() const
{
	return m_MajorSpacing;
}

bool Grid::GetShowAxis() const
{
	return m_bShowAxis;
}

const float* Grid::GetMajorGridColor() const
{
	return m_MajorGridColor;
}

const float* Grid::GetMinorGridColor() const
{
	return m_MinorGridColor;
}

const float* Grid::GetXAxisColor() const
{
	return m_XAxisColor;
}

const float* Grid::GetYAxisColor() const
{
	return m_YAxisColor;
}

const float* Grid::GetZAxisColor() const
{
	return m_ZAxisColor;
}

Geometry* Grid::GetMajorLines()
{
	if (m_bOptionChanged) {
		PrepareGeometry();
	}
	return &m_MajorLines;
}

Geometry* Grid::GetMinorLines()
{
	if (m_bOptionChanged) {
		PrepareGeometry();
	}
	return &m_MinorLines;
}

void Grid::setGridPerspective()
{
	m_Orientation = Quaternion();
	m_Origin = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	m_bOptionChanged = true;
}

void Grid::setGridTop()
{
	m_Orientation = Quaternion();
	m_Origin = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	m_bOptionChanged = true;
}

void Grid::setGridBottom()
{
	m_Orientation = Quaternion::rotation(3.141592653f, 1.0f, 0.0f, 0.0f);
	m_Origin = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	m_bOptionChanged = true;
}

void Grid::setGridRight()
{
	m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	m_Orientation.preMultiply(Quaternion::rotation(3.141592653f/2.0f, 0.0f, 0.0f, 1.0f));
	m_Origin = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	m_bOptionChanged = true;
}

void Grid::setGridLeft()
{
	m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	m_Orientation.preMultiply(Quaternion::rotation(-3.141592653f/2.0f, 0.0f, 0.0f, 1.0f));
	m_Origin = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	m_bOptionChanged = true;
}

void Grid::setGridFront()
{
	m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	m_Origin = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	m_bOptionChanged = true;
}

void Grid::setGridBack()
{
	m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	m_Orientation.preMultiply(Quaternion::rotation(3.141592653f, 0.0f, 0.0f, 1.0f));
	m_Origin = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	m_bOptionChanged = true;
}


void Grid::PrepareGeometry()
{
	unsigned int numberofmajorlines = PrepareMajorLines();
	PrepareMinorLines(numberofmajorlines);
	TransformGrid();
	m_bOptionChanged = false;
}

unsigned int Grid::PrepareMajorLines()
{
	float correctedSpacing = (m_GridSpacing==0?1.0f:m_GridSpacing);
	unsigned int correctedMajorSpacing = (m_MajorSpacing==0?1:m_MajorSpacing);
	unsigned int numberofmajorlines = (unsigned int)(m_GridExtent/(correctedSpacing*correctedMajorSpacing)) *2+1;
	unsigned int numberofmajorlineverts = numberofmajorlines*2;

	m_MajorLines.m_DiffuseColor[0] = m_MajorGridColor[0];
	m_MajorLines.m_DiffuseColor[1] = m_MajorGridColor[1];
	m_MajorLines.m_DiffuseColor[2] = m_MajorGridColor[2];

	m_MajorLines.AllocateLines( numberofmajorlineverts*2 +6, numberofmajorlines*2 +3);
	m_MajorLines.AllocateLineColors();
	int c;
	unsigned int nextvert = 0;
	unsigned int nextline = 0;
	float position;
	// vertical major lines
	for (c=-(int)numberofmajorlines/2; c<=(int)numberofmajorlines/2; c++) {
		position = (float)c*(float)correctedMajorSpacing*correctedSpacing;
		nextvert=MakeVerticalLine(nextline, nextvert, position, -m_GridExtent, 
			(c==0?0.0f:m_GridExtent), &m_MajorLines);
		nextline++;
	}
	// horizontal major lines
	for (c=-(int)numberofmajorlines/2; c<=(int)numberofmajorlines/2; c++) {
		position = (float)c*(float)correctedMajorSpacing*correctedSpacing;
		nextvert=MakeHorizontalLine(nextline, nextvert, position, -m_GridExtent, 
			(c==0?0.0f:m_GridExtent), &m_MajorLines);
		nextline++;
	}
	
	nextvert=MakeVerticalAxisLine(nextline, nextvert, 0.0f, 0, 
		m_GridExtent, &m_MajorLines);
	nextline++;
	nextvert=MakeHorizontalAxisLine(nextline, nextvert, 0.0f, 0, 
		m_GridExtent, &m_MajorLines);
	nextline++;
	nextvert=MakeDepthAxisLine(nextline, nextvert, 0.0f, 0, 
		m_GridExtent, &m_MajorLines);
	nextline++;

	m_MajorLines.m_LineWidth = 2.0f;
	return numberofmajorlines;
}

void Grid::PrepareMinorLines(unsigned int numberofmajorlines)
{
	float correctedSpacing = (m_GridSpacing==0?1.0f:m_GridSpacing);
	unsigned int correctedMajorSpacing = (m_MajorSpacing==0?1:m_MajorSpacing);
	unsigned int numberofminorlines = (unsigned int)(m_GridExtent/correctedSpacing) *2+1;
	unsigned int numberofminorlineverts = numberofminorlines*2;

	m_MinorLines.m_DiffuseColor[0] = m_MinorGridColor[0];
	m_MinorLines.m_DiffuseColor[1] = m_MinorGridColor[1];
	m_MinorLines.m_DiffuseColor[2] = m_MinorGridColor[2];

	m_MinorLines.AllocateLines( 
		(numberofminorlineverts-numberofmajorlines*2)*2, 
		(numberofminorlines-numberofmajorlines)*2 );
	int c;
	unsigned int nextvert = 0;
	unsigned int nextline = 0;
	float position;
	// vertical minor lines
	for (c=-(int)numberofminorlines/2; c<=(int)numberofminorlines/2; c++) {
		if (abs(c)%correctedMajorSpacing!=0) {
			position = (float)c*correctedSpacing;
			nextvert=MakeVerticalLine(nextline, nextvert, position, -m_GridExtent, m_GridExtent, &m_MinorLines);
			nextline++;
		}
	}
	//horizontal minor lines
	for (c=-(int)numberofminorlines/2; c<=(int)numberofminorlines/2; c++) {
		if (abs(c)%correctedMajorSpacing!=0) {
			position = (float)c*correctedSpacing;
			nextvert=MakeHorizontalLine(nextline, nextvert, position, -m_GridExtent, m_GridExtent, &m_MinorLines);
			nextline++;
		}
	}
	m_MinorLines.m_LineWidth = 1.0f;

}

void Grid::TransformGrid()
{
	unsigned int c;
	Vector result;
	for (c=0; c<m_MajorLines.m_NumLineVerts; c++) {
	  result = m_Orientation.applyRotation(Vector(m_MajorLines.m_LineVerts.get() + c*3));
	  result += m_Origin;
	  m_MajorLines.m_LineVerts[c*3+0] = result[0];
	  m_MajorLines.m_LineVerts[c*3+1] = result[1];
	  m_MajorLines.m_LineVerts[c*3+2] = result[2];
	}
	for (c=0; c<m_MinorLines.m_NumLineVerts; c++) {
	  result = m_Orientation.applyRotation(Vector(m_MinorLines.m_LineVerts.get() + c*3));
	  result += m_Origin;
	  m_MinorLines.m_LineVerts[c*3+0] = result[0];
	  m_MinorLines.m_LineVerts[c*3+1] = result[1];
	  m_MinorLines.m_LineVerts[c*3+2] = result[2];
	}
}



