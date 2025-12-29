/*
  Copyright 2002-2003 The University of Texas at Austin

        Authors: Anthony Thane <thanea@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeLibrary.

  VolumeLibrary is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeLibrary is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// PolygonArray.cpp: implementation of the PolygonArray class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRenderer/Polygon.h>
#include <VolumeRenderer/PolygonArray.h>

using namespace OpenGLVolumeRendering;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

PolygonArray::PolygonArray(unsigned int sizeGuess)
    : m_PolygonArray(sizeGuess) {}

PolygonArray::~PolygonArray() {}

void PolygonArray::clearPolygons() { m_PolygonArray.clear(); }

void PolygonArray::addPolygon(const Polygon &polygon) {
  //	doubleArray();
  //	m_PolygonArray[m_NumPolygons] = polygon;
  //	m_NumPolygons++;

  m_PolygonArray.push_back(polygon);
}

Polygon *PolygonArray::getPolygon(unsigned int i) {
  if (i < m_PolygonArray.size())
    return (&m_PolygonArray[i]);
  return 0;
}

unsigned int PolygonArray::getNumPolygons() { return m_PolygonArray.size(); }

#if 0
void PolygonArray::doubleArray()
{
	if (m_NumPolygons == m_ArraySize) {
		Polygon* temp = new Polygon[m_ArraySize*2];
		if (temp) {
			unsigned int c;
			for (c=0; c<m_NumPolygons; c++) {
				temp[c] = m_PolygonArray[c];
			}
			delete [] m_PolygonArray;
			m_PolygonArray = temp;
			m_ArraySize*=2;
		}
	}
}

void PolygonArray::allocateArray(unsigned int sizeGuess)
{
	delete [] m_PolygonArray;
	m_PolygonArray = new Polygon[sizeGuess];
	m_ArraySize = sizeGuess;
}
#endif
