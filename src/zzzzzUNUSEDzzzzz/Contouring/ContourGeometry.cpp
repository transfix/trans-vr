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

// ContourGeometry.cpp: implementation of the ContourGeometry class.
//
//////////////////////////////////////////////////////////////////////

#include <Contouring/ContourGeometry.h>
#include <VolumeWidget/Matrix.h>
#include <VolumeWidget/Vector.h>
#include <cvcraw_geometry/Geometry.h>
#include <math.h>
//#include "../ipoly/src/ipoly.h"
//#include "../ipoly/src/ipolyutil.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ContourGeometry::ContourGeometry()
{
	setDefaults();
}

ContourGeometry::~ContourGeometry()
{
	destroyTriangleBuffers();
	destroyVertexBuffers();
	destroyQuadBuffers();
}

void ContourGeometry::setWireframeMode(bool state)
{
	m_WireframeRender = state;
}

void ContourGeometry::setSurfWithWire(bool state)
{
	m_SurfWithWire = state;
}

void ContourGeometry::renderQuadMesh()
{
	//if (!m_Quads || !m_Vertex1 || m_Normal1)
	//	return;

	glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_NORMALIZE);
	if (m_WireframeRender) {
		int i;

		if (m_UseColors) { //draw with color
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glEnable(GL_COLOR_MATERIAL);

			for (i=0; i<(int)m_NumQuads; i++) {
				glBegin(GL_LINE_LOOP);
				
				glColor3fv(&(m_Color1[m_Quads[i*4+0]*3]));
				glNormal3fv(&(m_Normal1[m_Quads[i*4+0]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+0]*3]));
				
				glColor3fv(&(m_Color1[m_Quads[i*4+1]*3]));
				glNormal3fv(&(m_Normal1[m_Quads[i*4+1]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+1]*3]));
				
				glColor3fv(&(m_Color1[m_Quads[i*4+2]*3]));
				glNormal3fv(&(m_Normal1[m_Quads[i*4+2]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+2]*3]));
				
				glColor3fv(&(m_Color1[m_Quads[i*4+3]*3]));
				glNormal3fv(&(m_Normal1[m_Quads[i*4+3]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+3]*3]));
				
				glEnd();
			}
		}
		else { // draw without color
			for (i=0; i<(int)m_NumQuads; i++) {
				glBegin(GL_LINE_LOOP);
			
				glNormal3fv(&(m_Normal1[m_Quads[i*4+0]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+0]*3]));
				
				glNormal3fv(&(m_Normal1[m_Quads[i*4+1]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+1]*3]));
				
				glNormal3fv(&(m_Normal1[m_Quads[i*4+2]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+2]*3]));
				
				glNormal3fv(&(m_Normal1[m_Quads[i*4+3]*3]));
				glVertex3fv(&(m_Vertex1[m_Quads[i*4+3]*3]));
				
				glEnd();
			}
		}
	}
	else { // quad render
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		if (m_UseColors) {
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glEnable(GL_COLOR_MATERIAL);
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_FLOAT, 0, m_Color1);
		}
	
		glVertexPointer(3, GL_FLOAT, 0, m_Vertex1);
		glNormalPointer(GL_FLOAT, 0, m_Normal1);
		glDrawElements(GL_QUADS, m_NumQuads*4, GL_UNSIGNED_INT, m_Quads);
		
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
	glPopAttrib();
}

void ContourGeometry::renderContour()
{
	if (!m_Triangles || !m_Vertex1 || ! m_Normal1) 
		return;

	doInterpolation(m_Isovalue);
	glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_NORMALIZE);
#if 0
	if (m_WireframeRender) {
		int i;

		if (m_UseColors) { //draw with color
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glEnable(GL_COLOR_MATERIAL);

			for (i=0; i<(int)m_NumTriangles; i++) {
				glBegin(GL_LINE_LOOP);
				
				glColor3fv(&(m_Color1[m_Triangles[i*3+0]*3]));
				glNormal3fv(&(m_Normal1[m_Triangles[i*3+0]*3]));
				glVertex4fv(&(m_Vertex1[m_Triangles[i*3+0]*4]));
				
				glColor3fv(&(m_Color1[m_Triangles[i*3+1]*3]));
				glNormal3fv(&(m_Normal1[m_Triangles[i*3+1]*3]));
				glVertex4fv(&(m_Vertex1[m_Triangles[i*3+1]*4]));
				
				glColor3fv(&(m_Color1[m_Triangles[i*3+2]*3]));
				glNormal3fv(&(m_Normal1[m_Triangles[i*3+2]*3]));
				glVertex4fv(&(m_Vertex1[m_Triangles[i*3+2]*4]));
				
				glEnd();
			}
		}
		else { // draw without color
			for (i=0; i<(int)m_NumTriangles; i++) {
				glBegin(GL_LINE_LOOP);
			
				glNormal3fv(&(m_Normal1[m_Triangles[i*3+0]*3]));
				glVertex4fv(&(m_Vertex1[m_Triangles[i*3+0]*4]));
				
				glNormal3fv(&(m_Normal1[m_Triangles[i*3+1]*3]));
				glVertex4fv(&(m_Vertex1[m_Triangles[i*3+1]*4]));
				
				glNormal3fv(&(m_Normal1[m_Triangles[i*3+2]*3]));
				glVertex4fv(&(m_Vertex1[m_Triangles[i*3+2]*4]));
				
				glEnd();
			}
		}
	}
	else { // triangle render
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		if (m_UseColors) {
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glEnable(GL_COLOR_MATERIAL);
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_FLOAT, 0, m_Color1);
		}
	
		glVertexPointer(4, GL_FLOAT, 0, m_Vertex1);
		glNormalPointer(GL_FLOAT, 0, m_Normal1);
		glDrawElements(GL_TRIANGLES, m_NumTriangles*3, GL_UNSIGNED_INT, m_Triangles);
		
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
#endif
	
	GLint params[2];

	//back up current setting
	glGetIntegerv(GL_POLYGON_MODE,params);

	{ // triangle render
	  glEnableClientState(GL_VERTEX_ARRAY);
	  glEnableClientState(GL_NORMAL_ARRAY);
	  if (m_UseColors) {
	    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	    glEnable(GL_COLOR_MATERIAL);
	    glEnableClientState(GL_COLOR_ARRAY);
	    glColorPointer(3, GL_FLOAT, 0, m_Color1);
	  }
	
	  glVertexPointer(4, GL_FLOAT, 0, m_Vertex1);
	  glNormalPointer(GL_FLOAT, 0, m_Normal1);

	  if(!m_WireframeRender ||
	     (m_WireframeRender && m_SurfWithWire))
	    {
	      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	      if(m_SurfWithWire)
		{
		  glEnable(GL_POLYGON_OFFSET_FILL);
		  glPolygonOffset(1.0,1.0);
		}

	      glDrawElements(GL_TRIANGLES, m_NumTriangles*3, GL_UNSIGNED_INT, m_Triangles);

	      if(m_SurfWithWire)
		{
		  glPolygonOffset(0.0,0.0);
		  glDisable(GL_POLYGON_OFFSET_FILL);
		}
	    }

	  if(m_WireframeRender)
	    {
	      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	      if(m_SurfWithWire)
		{
		  glDisable(GL_LIGHTING);
		  glDisableClientState(GL_COLOR_ARRAY);
		  //		  glEnable(GL_POLYGON_OFFSET_LINE);
		  //		  glPolygonOffset(-1.0,-1.0);
		  glColor3f(0.0,0.0,0.0); //black wireframe
		}

	      glDrawElements(GL_TRIANGLES, m_NumTriangles*3, GL_UNSIGNED_INT, m_Triangles);

	      if(m_SurfWithWire)
		{
		  //		  glPolygonOffset(0.0,0.0);
		  //glDisable(GL_POLYGON_OFFSET_LINE);
		}
	    }
		
	  glDisableClientState(GL_VERTEX_ARRAY);
	  glDisableClientState(GL_NORMAL_ARRAY);
	  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	  glDisableClientState(GL_COLOR_ARRAY);
	}

	//restore previous setting for polygon mode
	glPolygonMode(GL_FRONT,params[0]);
	glPolygonMode(GL_BACK,params[1]);
	
	glPopAttrib();
}
/*
void ContourGeometry::addToIPoly(IPolyCntl* cp, const Matrix& matrix, int& nextVert)
{
	if (!m_Triangles || !m_Vertex1 || ! m_Normal1) 
		return;

	Matrix inverseTranspose = matrix.inverseTranspose();

	doInterpolation(m_Isovalue);
	
	unsigned int c;
	Vector vector;
	float arrayNormal[3];
	double arrayPoint[3]; 
	Vector point, normal;
	for (c=0; c<m_NumVertices; c++) {
		point[0] = m_Vertex1[c*4+0];
		point[1] = m_Vertex1[c*4+1];
		point[2] = m_Vertex1[c*4+2];
		point[3] = 1.0;
		normal[0] = m_Normal1[c*3+0];
		normal[1] = m_Normal1[c*3+1];
		normal[2] = m_Normal1[c*3+2];
		normal[3] = 0.0;
		// set the data up for the right coord space
		point = matrix*point;
		normal = inverseTranspose*normal;

		arrayPoint[0] = point[0];
		arrayPoint[1] = point[1];
		arrayPoint[2] = point[2];
		arrayNormal[0] = normal[0];
		arrayNormal[1] = normal[1];
		arrayNormal[2] = normal[2];
		IPolyAddVert(cp, arrayPoint, arrayNormal, IPolyNullSize, IPolyNullColor, IPolyNullValue);
	}
	int verts[3];
	for (c=0; c<m_NumTriangles; c++) {
		verts[0] = m_Triangles[c*3+0]+nextVert;
		verts[1] = m_Triangles[c*3+1]+nextVert;
		verts[2] = m_Triangles[c*3+2]+nextVert;
		IPolyAddFaceVerts(cp, 3, verts, IPolyNullSize, IPolyNullColor, IPolyNullValue);
	}
	nextVert+=m_NumVertices;
	
}
*/
void ContourGeometry::addToGeometry(Geometry* geometry, const Matrix& matrix, int& nextVert, int& nextTri)
{
	if (!m_Triangles || !m_Vertex1 || ! m_Normal1) 
		return;

	int nextVertPrivate = nextVert;

	Matrix inverseTranspose = matrix.inverseTranspose();

	doInterpolation(m_Isovalue);
	
	unsigned int c;
	Vector vector;
	Vector point, normal;
	for (c=0; c<m_NumVertices; c++) {
		point[0] = m_Vertex1[c*4+0];
		point[1] = m_Vertex1[c*4+1];
		point[2] = m_Vertex1[c*4+2];
		point[3] = 1.0;
		normal[0] = m_Normal1[c*3+0];
		normal[1] = m_Normal1[c*3+1];
		normal[2] = m_Normal1[c*3+2];
		normal[3] = 0.0;
		// set the data up for the right coord space
		point = matrix*point;
		normal = inverseTranspose*normal;

		geometry->m_TriVerts[nextVertPrivate*3+0] = point[0];
		geometry->m_TriVerts[nextVertPrivate*3+1] = point[1];
		geometry->m_TriVerts[nextVertPrivate*3+2] = point[2];
		geometry->m_TriVertNormals[nextVertPrivate*3+0] = normal[0];
		geometry->m_TriVertNormals[nextVertPrivate*3+1] = normal[1];
		geometry->m_TriVertNormals[nextVertPrivate*3+2] = normal[2];
		if (useColors()) {
			geometry->m_TriVertColors[nextVertPrivate*3+0] = m_Color1[c*3+0];
			geometry->m_TriVertColors[nextVertPrivate*3+1] = m_Color1[c*3+1];
			geometry->m_TriVertColors[nextVertPrivate*3+2] = m_Color1[c*3+2];
		}
		nextVertPrivate++;
	}
	for (c=0; c<m_NumTriangles; c++) {
		geometry->m_Tris[nextTri*3+0] = m_Triangles[c*3+0]+nextVert;
		geometry->m_Tris[nextTri*3+1] = m_Triangles[c*3+1]+nextVert;
		geometry->m_Tris[nextTri*3+2] = m_Triangles[c*3+2]+nextVert;
		nextTri++;
	}
	nextVert=nextVertPrivate;
	
}
	
bool ContourGeometry::addQuadVertex(GLfloat vx, GLfloat vy, GLfloat vz, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat cx, GLfloat cy, GLfloat cz)
{
	if (m_NumVertices==m_NumVerticesAllocated) {
		if (!doubleVertexBuffers()) {
			return false;
		}
	}

	m_Vertex1[m_NumVertices*3+0] = vx;
	m_Vertex1[m_NumVertices*3+1] = vy;
	m_Vertex1[m_NumVertices*3+2] = vz;
	m_Normal1[m_NumVertices*3+0] = nx;
	m_Normal1[m_NumVertices*3+1] = ny;
	m_Normal1[m_NumVertices*3+2] = nz;
	m_Color1[m_NumVertices*3+0] = cx;
	m_Color1[m_NumVertices*3+1] = cy;
	m_Color1[m_NumVertices*3+2] = cz;

	m_NumVertices++;
	return true;
}

bool ContourGeometry::addQuad(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int v4)
{
	// add a single quad, doubling the array if necessary
	if (m_NumQuads==m_NumQuadsAllocated) {
		if (!doubleQuadBuffers()) {
			return false;
		}
	}

	m_Quads[m_NumQuads*4+0] = v1;
	m_Quads[m_NumQuads*4+1] = v2;
	m_Quads[m_NumQuads*4+2] = v3;
	m_Quads[m_NumQuads*4+3] = v4;

	m_NumQuads++;
	return true;
}

int ContourGeometry::getNumVerts()
{
	return m_NumVertices;
}

int ContourGeometry::getNumTris()
{
	return m_NumTriangles;
}

void ContourGeometry::setDefaults()
{
	// set default values for all member variables
	m_Normal1 = 0;
	m_Normal2 = 0;
	m_Vertex1 = 0;
	m_Vertex2 = 0;
	m_Color1 = 0;
	m_Color2 = 0;
	m_NumVertices = 0;
	m_NumVerticesAllocated = 0;

	m_Triangles = 0;
	m_NumTriangles = 0;
	m_NumTrianglesAllocated = 0;

	m_Quads = 0;
	m_NumQuads = 0;
	m_NumQuadsAllocated = 0;

	m_Isovalue = 0.0f;

	m_HardwareAccelerated = false;
	m_InterpolationDone = false;
	m_UseColors = false;
	m_WireframeRender = false;
	m_SurfWithWire = false;
}

void ContourGeometry::setUseColors(bool useColors)
{
	m_UseColors = useColors;
}

void ContourGeometry::setIsovalue(GLfloat isovalue)
{
	m_Isovalue = isovalue;
}

void ContourGeometry::setSingleColor(GLfloat R, GLfloat G, GLfloat B)
{
	int i;

	// reassign all the vertex colors
	for (i=0; i < (int)m_NumVertices; i++) {
		m_Color1[i*3+0] = R;
		m_Color1[i*3+1] = G;
		m_Color1[i*3+2] = B;
	}
}

bool ContourGeometry::useColors()
{
	return m_UseColors;
}

bool ContourGeometry::allocateVertexBuffers(unsigned int initialSize)
{
	// if we already have a buffer big enough, just return true and use that buffer
	if (initialSize>m_NumVerticesAllocated) {
		destroyVertexBuffers();
		m_NumVertices = 0;
		m_InterpolationDone = false;
		return forceAllocateVertexBuffers(initialSize);
	}
	else {
		m_NumVertices = 0;
		m_InterpolationDone = false;
		return true;
	}
}

bool ContourGeometry::forceAllocateVertexBuffers(unsigned int size)
{
	// allocate the buffers without checking to see if we already have a big enough buffer
	m_Vertex1 = new GLfloat[size*4];
	m_Vertex2 = new GLfloat[size*4];
	m_Normal1 = new GLfloat[size*3];
	m_Normal2 = new GLfloat[size*3];
	m_Color1 = new GLfloat[size*3];
	m_Color2 = new GLfloat[size*3];

	if (m_Vertex1 && m_Vertex2 && m_Normal1 && m_Normal2 && m_Color1 && m_Color2) {
		m_NumVerticesAllocated = size;
		return true;
	}
	else {
		destroyVertexBuffers();
		return false;
	}
}

bool ContourGeometry::doubleVertexBuffers()
{
	// double the size of the vertex buffer and copy the vertices to the new buffer
	if (
		!doubleFloatArray(m_Vertex1, m_NumVerticesAllocated*4) ||
		!doubleFloatArray(m_Vertex2, m_NumVerticesAllocated*4) ||
		!doubleFloatArray(m_Normal1, m_NumVerticesAllocated*3) ||
		!doubleFloatArray(m_Normal2, m_NumVerticesAllocated*3) ||
		!doubleFloatArray(m_Color1, m_NumVerticesAllocated*3) ||
		!doubleFloatArray(m_Color2, m_NumVerticesAllocated*3)) {
		destroyVertexBuffers();
		return false;
	}
	else { // success
		m_NumVerticesAllocated*=2;
		return true;
	}
}

void ContourGeometry::destroyVertexBuffers()
{
	// free vertex buffer memory
	delete [] m_Vertex1;
	m_Vertex1 = 0;

	delete [] m_Vertex2;
	m_Vertex2 = 0;

	delete [] m_Normal1;
	m_Normal1 = 0;

	delete [] m_Normal2;
	m_Normal2 = 0;

	delete [] m_Color1;
	m_Color1 = 0;

	delete [] m_Color2;
	m_Color2 = 0;

	m_NumVerticesAllocated = 0;
	m_NumVertices = 0;
}

bool ContourGeometry::allocateTriangleBuffers(unsigned int initialSize)
{
	// if we already have a buffer big enough, just return true and use that buffer
	if (initialSize>m_NumTrianglesAllocated) {
		destroyTriangleBuffers();
		m_NumTriangles = 0;
		m_InterpolationDone = false;
		return forceAllocateTriangleBuffers(initialSize);
	}
	else {
		m_NumTriangles = 0;
		m_InterpolationDone = false;
		return true;
	}
}

bool ContourGeometry::forceAllocateTriangleBuffers(unsigned int size)
{
	// allocate the buffers without checking to see if we already have a big enough buffer
	m_Triangles = new GLuint[size*3];

	if (m_Triangles) {
		m_NumTrianglesAllocated = size;
		return true;
	}
	else {
		destroyTriangleBuffers();
		return false;
	}
}

bool ContourGeometry::doubleTriangleBuffers()
{
	// double the size of the triangle buffer and copy the triangles to the new buffer
	if (
		!doubleIntArray(m_Triangles, m_NumTrianglesAllocated*3)) {
		destroyTriangleBuffers();
		return false;
	}
	else { // success
		m_NumTrianglesAllocated*=2;
		return true;
	}
}

void ContourGeometry::destroyTriangleBuffers()
{
	// free triangle buffer memory
	delete [] m_Triangles;
	m_Triangles = 0;

	m_NumTrianglesAllocated = 0;
	m_NumTriangles = 0;
}

bool ContourGeometry::allocateQuadBuffers(unsigned int initialSize)
{
	// if we already have a buffer big enough, just return true and use that buffer
	if (initialSize>m_NumQuadsAllocated) {
		destroyQuadBuffers();
		m_NumQuads = 0;
		//m_InterpolationDone = false;
		return forceAllocateQuadBuffers(initialSize);
	}
	else {
		m_NumQuads= 0;
		//m_InterpolationDone = false;
		return true;
	}
}

bool ContourGeometry::forceAllocateQuadBuffers(unsigned int size)
{
	// allocate the buffers without checking to see if we already have a big enough buffer
	m_Quads = new GLuint[size*4];

	if (m_Quads) {
		m_NumQuadsAllocated = size;
		return true;
	}
	else {
		destroyQuadBuffers();
		return false;
	}
}

bool ContourGeometry::doubleQuadBuffers()
{
	// double the size of the quad buffer and copy the quads to the new buffer
	if (!doubleIntArray(m_Quads, m_NumQuadsAllocated*4)) {
		destroyQuadBuffers();
		return false;
	}
	else { // success
		m_NumQuadsAllocated*=2;
		return true;
	}
}

void ContourGeometry::destroyQuadBuffers()
{
	// free quad buffer memory
	delete [] m_Quads;
	m_Quads = 0;

	m_NumQuadsAllocated = 0;
	m_NumQuads = 0;
}

bool ContourGeometry::doubleFloatArray(GLfloat*& array, unsigned int size)
{
	// double a single float array, copy over the values, and delete the old array
	GLfloat* newArray = new GLfloat[size*2];
	if (!newArray) {
		return false;
	}
	unsigned int c;
	for (c=0; c<size; c++) {
		newArray[c] = array[c];
	}
	delete [] array;
	array = newArray;
	return true;
}

bool ContourGeometry::doubleIntArray(GLuint*& array, unsigned int size)
{
	// double a single uint array, copy over the values, and delete the old array
	GLuint* newArray = new GLuint[size*2];
	if (!newArray) {
		return false;
	}
	unsigned int c;
	for (c=0; c<size; c++) {
		newArray[c] = array[c];
	}
	delete [] array;
	array = newArray;
	return true;
}


bool ContourGeometry::addTriangle(unsigned int v1, unsigned int v2, unsigned int v3)
{
	// add a single triangle, doubling the array if necessary
	if (m_NumTriangles==m_NumTrianglesAllocated) {
		if (!doubleTriangleBuffers()) {
			return false;
		}
	}

	m_Triangles[m_NumTriangles*3+0] = v1;
	m_Triangles[m_NumTriangles*3+1] = v2;
	m_Triangles[m_NumTriangles*3+2] = v3;

	m_NumTriangles++;
	return true;
}

void ContourGeometry::doInterpolation(GLfloat isovalue)
{
	if (!m_InterpolationDone) {
		unsigned int c;
		if (m_UseColors) {
			for (c=0; c<m_NumVertices; c++) {
				interpArray(m_Vertex1+c*4, m_Vertex2+c*4, m_Normal1+c*3, m_Normal2+c*3, m_Vertex1[c*4+3], m_Vertex2[c*4+3], isovalue);
			}
		}
		else {
			for (c=0; c<m_NumVertices; c++) {
				interpArray(m_Vertex1+c*4, m_Vertex2+c*4, m_Normal1+c*3, m_Normal2+c*3, m_Vertex1[c*4+3], m_Vertex2[c*4+3], isovalue);
			}
		}
		m_InterpolationDone = true;
	}
}

static void cross(float* dest, const float* v1, const float* v2)
{

	dest[0] = v1[1]*v2[2] - v1[2]*v2[1];
	dest[1] = v1[2]*v2[0] - v1[0]*v2[2];
	dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

static void normalize(float* v)
{
	float len = (float)sqrt(
		v[0] * v[0] +
		v[1] * v[1] +
		v[2] * v[2]);
	if (len!=0.0) {
		v[0]/=len; //*biggestDim;
		v[1]/=len; //*biggestDim;
		v[2]/=len; //*biggestDim;
	}
	else {
		v[0] = 1.0;
	}
}

static void add(float* dest, const float* v)
{
	dest[0] += v[0];
	dest[1] += v[1];
	dest[2] += v[2];
}

static void set(float* dest, const float* v)
{
	dest[0] = v[0];
	dest[1] = v[1];
	dest[2] = v[2];
}

void ContourGeometry::CalculateQuadSmoothNormals()
{
	unsigned int c, v0, v1, v2, v3;
	float normal[3];

	float zero[3] = {0.0f, 0.0f, 0.0f};

	for (c=0; c<m_NumVertices; c++) {
		set(m_Normal1+c*3, zero);
	}

		
	// for each Quadangle
	for (c=0; c<m_NumQuads; c++) {
		v0 = m_Quads[c*4+0];
		v1 = m_Quads[c*4+1];
		v2 = m_Quads[c*4+2];
		v3 = m_Quads[c*4+3];
		CalculateQuadNormal(normal, v0, v1, v3);
		add(m_Normal1+v0*3, normal);
		add(m_Normal1+v1*3, normal);
		add(m_Normal1+v3*3, normal);
		CalculateQuadNormal(normal, v2, v3, v1);
		add(m_Normal1+v2*3, normal);
		add(m_Normal1+v3*3, normal);
		add(m_Normal1+v1*3, normal);
	}
	
	// normalize the vectors
	for (c=0; c<m_NumVertices; c++) {
		normalize(m_Normal1+c*3);
	}
}

void ContourGeometry::CalculateQuadNormal(float* norm, unsigned int v0, unsigned int v1, unsigned int v2)
{
	float vec1[3], vec2[3];
	vec1[0] = vec2[0] = -m_Vertex1[v0*3+0];
	vec1[1] = vec2[1] = -m_Vertex1[v0*3+1];
	vec1[2] = vec2[2] = -m_Vertex1[v0*3+2];
	vec1[0] += m_Vertex1[v1*3+0];
	vec1[1] += m_Vertex1[v1*3+1];
	vec1[2] += m_Vertex1[v1*3+2];
	
	vec2[0] += m_Vertex1[v2*3+0];
	vec2[1] += m_Vertex1[v2*3+1];
	vec2[2] += m_Vertex1[v2*3+2];


	cross(norm, vec1, vec2);
}

