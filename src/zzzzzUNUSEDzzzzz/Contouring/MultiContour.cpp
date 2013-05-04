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

// MultiContour.cpp: implementation of the MultiContour class.
//
//////////////////////////////////////////////////////////////////////

#include <Contouring/MultiContour.h>
#include <VolumeWidget/Matrix.h>
#include <cvcraw_geometry/Geometry.h>
//#include "../ipoly/src/ipoly.h"
//#include "../ipoly/src/ipolyutil.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MultiContour::MultiContour()
{
	setDefaults();
}

MultiContour::~MultiContour()
{
	removeAll();
	delete m_Matrix;
	delete m_SaveMatrix;
}

static inline double maxOfThree(double n1, double n2, double n3) {
	double max = (n1>n2?n1:n2);
	return (max>n3?max:n3);
}

void MultiContour::setData(unsigned char* data, 
						   unsigned int width, unsigned int height, unsigned int depth,
						   double aspectX, double aspectY, double aspectZ,
						   double subMinX, double subMinY, double subMinZ,
						   double subMaxX, double subMaxY, double subMaxZ,
						   double minX, double minY, double minZ,
						   double maxX, double maxY, double maxZ)
{
	m_SingleExtractor.setData(data, width, height, depth,
		aspectX, aspectY, aspectZ,
		subMinX, subMinY, subMinZ,
		subMaxX, subMaxY, subMaxZ);
	m_ContourExtractor = &m_SingleExtractor;

	resetContours();


	double maxAspect = maxOfThree(aspectX, aspectY, aspectZ);
	aspectX = aspectX/maxAspect;
	aspectY = aspectY/maxAspect;
	aspectZ = aspectZ/maxAspect;

	m_Matrix->reset();
	// scale to 0 to 1
	m_Matrix->preMultiplication(Matrix::scale(
		(float)(1.0/(double)(width-1)),
		(float)(1.0/(double)(height-1)),
		(float)(1.0/(double)(depth-1))
		));
	// center
	m_Matrix->preMultiplication(Matrix::translation(
		(float)(-(subMaxX+subMinX)/2.0),
		(float)(-(subMaxY+subMinY)/2.0),
		(float)(-(subMaxZ+subMinZ)/2.0)
		));
	// scale to aspect ratio
	m_Matrix->preMultiplication(Matrix::scale(
		(float)(aspectX/(subMaxX-subMinX)),
		(float)(aspectY/(subMaxY-subMinY)),
		(float)(aspectZ/(subMaxZ-subMinZ))
		));

	// set up the save matrix
	m_SaveMatrix->reset();
	// scale to 0 to 1
	m_SaveMatrix->preMultiplication(Matrix::scale(
		(float)(1.0/(double)(width-1)),
		(float)(1.0/(double)(height-1)),
		(float)(1.0/(double)(depth-1))
		));
	// align
	m_SaveMatrix->preMultiplication(Matrix::translation((float)(-subMinX),
							    (float)(-subMinY),
							    (float)(-subMinZ)));
	// scale to min to max
	m_SaveMatrix->preMultiplication(Matrix::scale(
		(float)((maxX-minX)/(subMaxX-subMinX)),
		(float)((maxY-minY)/(subMaxY-subMinY)),
		(float)((maxZ-minZ)/(subMaxZ-subMinZ))
		));
	// translate
	m_SaveMatrix->preMultiplication(Matrix::translation(
		(float)(minX),
		(float)(minY),
		(float)(minZ)
		));

	m_DataLoaded = true;
}

void MultiContour::setData(unsigned char* data, unsigned char* red, unsigned char* green, unsigned char* blue,
						   unsigned int width, unsigned int height, unsigned int depth,
						   double aspectX, double aspectY, double aspectZ,
						   double subMinX, double subMinY, double subMinZ,
						   double subMaxX, double subMaxY, double subMaxZ,
						   double minX, double minY, double minZ,
						   double maxX, double maxY, double maxZ)
{
	m_RGBAExtractor.setData(data, red, green, blue, 
		width, height, depth,
		aspectX, aspectY, aspectZ,
		subMinX, subMinY, subMinZ,
		subMaxX, subMaxY, subMaxZ);
	m_ContourExtractor = &m_RGBAExtractor;

	resetContours();


	double maxAspect = maxOfThree(aspectX, aspectY, aspectZ);
	aspectX = aspectX/maxAspect;
	aspectY = aspectY/maxAspect;
	aspectZ = aspectZ/maxAspect;

	m_Matrix->reset();
	// scale to 0 to 1
	m_Matrix->preMultiplication(Matrix::scale(
		(float)(1.0/(double)(width-1)),
		(float)(1.0/(double)(height-1)),
		(float)(1.0/(double)(depth-1))
		));
	// center
	m_Matrix->preMultiplication(Matrix::translation(
		(float)(-(subMaxX+subMinX)/2.0),
		(float)(-(subMaxY+subMinY)/2.0),
		(float)(-(subMaxZ+subMinZ)/2.0)
		));
	// scale to aspect ratio
	m_Matrix->preMultiplication(Matrix::scale(
		(float)(aspectX/(subMaxX-subMinX)),
		(float)(aspectY/(subMaxY-subMinY)),
		(float)(aspectZ/(subMaxZ-subMinZ))
		));

	// set up the save matrix
	m_SaveMatrix->reset();
	// scale to 0 to 1
	m_SaveMatrix->preMultiplication(Matrix::scale(
		(float)(1.0/(double)(width-1)),
		(float)(1.0/(double)(height-1)),
		(float)(1.0/(double)(depth-1))
		));
	// align
	m_SaveMatrix->preMultiplication(Matrix::translation(
		(float)(-subMinX),
		(float)(-subMinY),
		(float)(-subMinZ)
		));
	// scale to min to max
	m_SaveMatrix->preMultiplication(Matrix::scale(
		(float)((maxX-minX)/(subMaxX-subMinX)),
		(float)((maxY-minY)/(subMaxY-subMinY)),
		(float)((maxZ-minZ)/(subMaxZ-subMinZ))
		));
	// translate
	m_SaveMatrix->preMultiplication(Matrix::translation(
		(float)(minX),
		(float)(minY),
		(float)(minZ)
		));

	m_DataLoaded = true;
}

void MultiContour::addContour(int ID, float isovalue, float R, float G, float B)
{
  MultiContourNode* newNode = new MultiContourNode(ID, isovalue, R,G,B, m_WireframeRender, m_SurfWithWire, m_Head);
  m_Head = newNode;
}

void MultiContour::removeContour(int ID)
{
	MultiContourNode* toDelete;
	MultiContourNode* node = m_Head;

	// empty list
	if (m_Head == 0) {
		return;
	}

	// head is a special case
	if (m_Head->m_Contour.getID() == ID) {
		toDelete = m_Head;
		m_Head = m_Head->m_Next;
		toDelete->m_Next = 0;

		delete toDelete;
		return;
	}

	// do remainder of list
	while (node->m_Next && node->m_Next->m_Contour.getID()!=ID) {
		node = node->m_Next;
	}

	// remove the node
	if (node->m_Next) {
		toDelete = node->m_Next;
		node->m_Next = node->m_Next->m_Next;
		toDelete->m_Next = 0;

		delete toDelete;
		return;
	}
}

void MultiContour::removeAll()
{
	delete m_Head;
	m_Head = 0;
}

void MultiContour::setIsovalue(int ID, float isovalue)
{
	MultiContourNode* node = m_Head;

	while (node && node->m_Contour.getID()!=ID) {
		node = node->m_Next;
	}
	if (node) {
		node->m_Contour.setIsovalue(isovalue);
	}
}

void MultiContour::setColor(int ID, float R, float G, float B)
{
	MultiContourNode* node = m_Head;
	// if we're in RGBA mode, we don't want to overwrite the contour's color
	// clobberColor == false -> (m_ContourExtractor == &m_RGBAExtractor)
	bool clobberColor = (m_ContourExtractor == &m_SingleExtractor);

	while (node && node->m_Contour.getID()!=ID) {
		node = node->m_Next;
	}
	if (node) {
#ifdef __GNUC__
#warning TODO: need an option to clobber color or not.
#endif
		node->m_Contour.setSingleColor(R,G,B, clobberColor && false);
	}
}

void MultiContour::setWireframeMode(bool state)
{
	m_WireframeRender = state;
	
	if (m_DataLoaded) {
		MultiContourNode* node = m_Head;

		while (node) {
			node->m_Contour.setWireframeMode(state);
			node = node->m_Next;
		}
	}
}

void MultiContour::setSurfWithWire(bool state)
{
  m_SurfWithWire = state;
	
  if (m_DataLoaded) {
    MultiContourNode* node = m_Head;
    
    while (node) {
      node->m_Contour.setSurfWithWire(state);
      node = node->m_Next;
    }
  }
}

void MultiContour::renderContours() const
{
	if (m_DataLoaded) {
		MultiContourNode* node = m_Head;

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		if (m_Matrix) 
			glMultMatrixf(m_Matrix->getMatrix());

		while (node) {
			node->m_Contour.renderContour(*m_ContourExtractor);
			node = node->m_Next;
		}

		glPopMatrix();
	}

}

void MultiContour::forceExtraction() const
{
	if (m_DataLoaded) {
		MultiContourNode* node = m_Head;

		while (node) {
			node->m_Contour.extract(*m_ContourExtractor);
			node = node->m_Next;
		}

	}
}

bool MultiContour::render()
{
	renderContours();
	return true;
}

/*iPoly* MultiContour::getIPoly()
{
	int nextVert = 0;


	if (m_DataLoaded && m_Head) {
		iPoly* ipoly = IPolyNew();
		IPolyCntl* cp = IPolyBegin(ipoly, IPOLY_DEFAULT);
		IPolyInitVerts(cp, 1024, IPolyNormsYes, IPolySizesNo, NoColor);
		IPolyInitFaces(cp, 1024, IPolySizesNo, NoColor);

		MultiContourNode* node = m_Head;


		while (node) {
			node->m_Contour.addToIPoly(m_ContourExtractor, cp, *m_SaveMatrix, nextVert);
			node = node->m_Next;
		}

		IPolyEnd(cp);
		return ipoly;
	}
	else {
		return 0;
	}

}*/

Geometry* MultiContour::getGeometry()
{
	int nextVert = 0, nextTri = 0;
	Geometry* geometry = 0;

	if (m_DataLoaded && m_Head) {
		geometry = new Geometry;
		int numVerts = getNumVerts();
		int numTris = getNumTris();
		geometry->AllocateTris(numVerts, numTris);
		if (m_Head->m_Contour.useColors()) {
			geometry->AllocateTriVertColors();
		}

		MultiContourNode* node = m_Head;

		while (node) {
			node->m_Contour.addToGeometry(*m_ContourExtractor, geometry, *m_SaveMatrix, nextVert, nextTri);
			node = node->m_Next;
		}

		geometry->SetTriNormalsReady();
	}

	return geometry;
}

int MultiContour::getNumVerts()
{
	int numVerts = 0;
	if (m_DataLoaded) {
		MultiContourNode* node = m_Head;

		while (node) {
			numVerts += node->m_Contour.getNumVerts(*m_ContourExtractor);
			node = node->m_Next;
		}
	}
	return numVerts;
}

int MultiContour::getNumTris()
{
	int numVerts = 0;
	if (m_DataLoaded) {
		MultiContourNode* node = m_Head;

		while (node) {
			numVerts += node->m_Contour.getNumTris(*m_ContourExtractor);
			node = node->m_Next;
		}
	}
	return numVerts;
}

void MultiContour::setDefaults()
{
	m_Head = 0;
	m_DataLoaded = false;
	m_WireframeRender = false;
	m_SurfWithWire = false;
	m_Matrix = new Matrix;
	m_SaveMatrix = new Matrix;
}

void MultiContour::resetContours()
{
	MultiContourNode* node = m_Head;

	while (node) {
		node->m_Contour.resetContour();
		node = node->m_Next;
	}
}

