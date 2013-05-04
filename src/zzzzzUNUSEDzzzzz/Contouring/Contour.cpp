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

// Contour.cpp: implementation of the Contour class.
//
//////////////////////////////////////////////////////////////////////

#include <Contouring/Contour.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


Contour::Contour()
{
	setDefaults();
}

Contour::~Contour()
{

}

void Contour::renderContour(const ContourExtractor& contourExtractor)
{
	if (!m_ContourReady) {
		extract(contourExtractor);
	}
#ifndef DUALLIB
	m_ContourGeometry.renderContour();
#else
	m_ContourGeometry.renderQuadMesh();
#endif
}

/*
void Contour::addToIPoly(const ContourExtractor& contourExtractor, IPolyCntl* cp, const Matrix& matrix, int& nextVert)
{
	if (!m_ContourReady) {
		extract(contourExtractor);
	}
	m_ContourGeometry.addToIPoly(cp, matrix, nextVert);
}
*/

void Contour::addToGeometry(const ContourExtractor& contourExtractor, Geometry* geometry, const Matrix& matrix, int& nextVert, int& nextTri)
{
	if (!m_ContourReady) {
		extract(contourExtractor);
	}
	m_ContourGeometry.addToGeometry(geometry, matrix, nextVert, nextTri);
}

int Contour::getNumVerts(const ContourExtractor& contourExtractor)
{
	if (!m_ContourReady) {
		extract(contourExtractor);
	}
	return m_ContourGeometry.getNumVerts();
}

int Contour::getNumTris(const ContourExtractor& contourExtractor)
{
	if (!m_ContourReady) {
		extract(contourExtractor);
	}
	return m_ContourGeometry.getNumTris();
}

void Contour::setID(int id)
{
	m_ID = id;
}

int Contour::getID()
{
	return m_ID;
}

void Contour::setIsovalue(float isovalue)
{
	m_Isovalue = isovalue;
	m_ContourReady = false;
}

void Contour::setSingleColor(float R, float G, float B, bool clobber)
{
	// set the color vars for the extraction
	m_R = R; m_G = G; m_B = B;

	if (clobber) {
		// set the color for geometry that was already extracted
		m_ContourGeometry.setSingleColor((GLfloat)R,(GLfloat)G,(GLfloat)B);
	}
}

void Contour::resetContour()
{
	m_ContourReady = false;
}

void Contour::setWireframeMode(bool state)
{
  m_ContourGeometry.setWireframeMode(state);
}

void Contour::setSurfWithWire(bool state)
{
  m_ContourGeometry.setSurfWithWire(state);
}

bool Contour::useColors()
{
	return m_ContourGeometry.useColors();
}

void Contour::setDefaults()
{
	m_ID = 0;
	m_Isovalue = 0.0f;
	m_ContourReady = false;

	m_R = 0.5;
	m_G = 0.5;
	m_B = 0.5;
}

void Contour::extract(const ContourExtractor& contourExtractor)
{
	contourExtractor.extractContour(&m_ContourGeometry, m_Isovalue, m_R,m_G,m_B);
	m_ContourReady = true;
}



