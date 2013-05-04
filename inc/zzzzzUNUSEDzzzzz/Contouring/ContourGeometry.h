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

// ContourGeometry.h: interface for the ContourGeometry class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_CONTOURGEOMETRY_H__D1ABB49A_F4FB_4D74_B19A_12B037CBB7F7__INCLUDED_)
#define AFX_CONTOURGEOMETRY_H__D1ABB49A_F4FB_4D74_B19A_12B037CBB7F7__INCLUDED_

#if defined(_WIN32) && !defined(APIENTRY) && !defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

#include <glew/glew.h>

class Matrix;
class Geometry;
struct IPolyCntl;

///\class ContourGeometry ContourGeometry.h
///\author Anthony Thane
///\author John Wiggins
///\brief A container for isocontours. The contour can be made up of triangles
/// or quads.
class ContourGeometry  
{
public:
	ContourGeometry();
	virtual ~ContourGeometry();

///\fn void renderContour()
///\brief Renders the contour using OpenGL (triangles only)
	void renderContour();
///\fn void renderQuadMesh()
///\brief Renders the contour using OpenGL. This is only used with the dual
/// contouring code. (see SingleExtractor)
	void renderQuadMesh();
//	void addToIPoly(IPolyCntl* cp, const Matrix& matrix, int& nextVert);
///\fn void addToGeometry(Geometry* geometry, const Matrix& matrix, int& nextVert, int& nextTri)
///\brief Adds the contour to a Geometry object
///\param geometry The Geometry object to add the contour to
///\param matrix A Matrix object to hold a scale and translation transformation
///\param nextVert The next available vertex index in the Geometry object
///\param nextTri The next available triangle index in the Geometry object
	void addToGeometry(Geometry* geometry, const Matrix& matrix, int& nextVert, int& nextTri);
///\fn int getNumVerts()
///\brief Returns the number of vertices in the contour
///\return The number of vertices
	int getNumVerts();
///\fn int getNumTris()
///\brief Returns the number of triangles in the contour
///\return The number of triangles
	int getNumTris();

///\fn bool addQuadVertex(GLfloat vx, GLfloat vy, GLfloat vz, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat cx, GLfloat cy, GLfloat cz)
///\brief Adds a vertex to the quad mesh
///\param vx The X coordinate of the vertex
///\param vy The Y coordinate of the vertex
///\param vz The Z coordinate of the vertex
///\param nx The X component of the vertex's normal
///\param ny The Y component of the vertex's normal
///\param nz The Z component of the vertex's normal
///\param cx The red component of the vertex's color
///\param cy The green component of the vertex's color
///\param cz The blue component of the vertex's color
///\return A bool indicating success or failure
	bool addQuadVertex(GLfloat vx, GLfloat vy, GLfloat vz, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat cx, GLfloat cy, GLfloat cz);
///\fn bool addQuad(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int v4)
///\brief Adds a quad to the quad mesh
///\param v1 The index of the first vertex
///\param v2 The index of the second vertex
///\param v3 The index of the third vertex
///\param v4 The index of the fourth vertex
///\return A bool indicating success or failure
	bool addQuad(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int v4);

///\fn bool allocateVertexBuffers(unsigned int initialSize)
///\brief Allocates memory for vertices (including color & normal)
///\param initialSize The number of vertices to allocate
///\return A bool indicating success or failure
	bool allocateVertexBuffers(unsigned int initialSize);
///\fn void destroyVertexBuffers()
///\brief Deallocates any memory allocated for vertices
	void destroyVertexBuffers();
///\fn bool allocateTriangleBuffers(unsigned int initialSize)
///\brief Allocates memory for triangles
///\param initialSize The number of triangles to allocate
///\return A bool indicating success or failure
	bool allocateTriangleBuffers(unsigned int initialSize);
///\fn void destroyTriangleBuffers()
///\brief Deallocates any memory allocated for triangles
	void destroyTriangleBuffers();
///\fn bool allocateQuadBuffers(unsigned int initialSize)
///\brief Allocates memory for quads
///\param initialSize The number of quads to allocate
///\return A bool indicating success or failure
	bool allocateQuadBuffers(unsigned int initialSize);
///\fn void destroyQuadBuffers()
///\brief Deallocates any memory allocated for quads
	void destroyQuadBuffers();

///\fn void setWireframeMode(bool state)
///\brief Sets the rendering mode to wireframe or surface rendering
///\param state true -> render as a wireframe; false -> render as a surface
	void setWireframeMode(bool state);

///\fn void setSurfWithWire(bool state)
///\brief If the rendering mode is wireframe, chooses to draw surface under wireframe
///\param state true -> render surface with wireframe; false -> render without surface
	void setSurfWithWire(bool state);

///\fn void setUseColors(bool useColors)
///\brief Sets tye coloring mode for the contour
///\param useColors true -> render with color; false -> render without color (OpenGL's default shading)
	void setUseColors(bool useColors);
///\fn void setIsovalue(GLfloat isovalue)
///\brief Sets the isovalue of the contour
///\param isovalue The isovalue
	void setIsovalue(GLfloat isovalue);
///\fn void setSingleColor(GLfloat R, GLfloat G, GLfloat B)
///\brief Assigns a single color to the entire contour
///\param R The red component of the color
///\param G The green component of the color
///\param B The blue component of the color
	void setSingleColor(GLfloat R, GLfloat G, GLfloat B);
///\fn bool useColors()
///\brief Returns whether or not the contour renders in color or not
///\return A bool
	bool useColors();

///\fn inline bool addEdge(unsigned int& id, GLfloat v1x, GLfloat v1y, GLfloat v1z,GLfloat n1x, GLfloat n1y, GLfloat n1z, GLfloat v2x, GLfloat v2y, GLfloat v2z, GLfloat n2x, GLfloat n2y, GLfloat n2z, GLfloat den1, GLfloat den2)
///\brief Adds an edge to the triangle mesh
///\param id A unique id for the edge
///\param v1x The X coordinate of the first vertex
///\param v1y The Y coordinate of the first vertex
///\param v1z The Z coordinate of the first vertex
///\param n1x The X component of the first vertex's normal
///\param n1y The Y component of the first vertex's normal
///\param n1z The Z component of the first vertex's normal
///\param v2x The X coordinate of the second vertex
///\param v2y The Y coordinate of the second vertex
///\param v2z The Z coordinate of the second vertex
///\param n2x The X component of the second vertex's normal
///\param n2y The Y component of the second vertex's normal
///\param n2z The Z component of the second vertex's normal
///\param den1 The density of the volume at the first vertex
///\param den2 The density of the volume at the second vertex
///\returns A bool indicating success or failure
	inline bool addEdge(unsigned int& id, GLfloat v1x, GLfloat v1y, GLfloat v1z,
		GLfloat n1x, GLfloat n1y, GLfloat n1z,
		GLfloat v2x, GLfloat v2y, GLfloat v2z,
		GLfloat n2x, GLfloat n2y, GLfloat n2z,
		GLfloat den1, GLfloat den2);
///\fn inline bool addEdge(unsigned int& id, GLfloat v1x, GLfloat v1y, GLfloat v1z, GLfloat n1x, GLfloat n1y, GLfloat n1z, GLfloat c1x, GLfloat c1y, GLfloat c1z, GLfloat v2x, GLfloat v2y, GLfloat v2z, GLfloat n2x, GLfloat n2y, GLfloat n2z, GLfloat c2x, GLfloat c2y, GLfloat c2z, GLfloat den1, GLfloat den2)
///\brief Adds an edge to the triangle mesh
///\param id A unique id for the edge
///\param v1x The X coordinate of the first vertex
///\param v1y The Y coordinate of the first vertex
///\param v1z The Z coordinate of the first vertex
///\param n1x The X component of the first vertex's normal
///\param n1y The Y component of the first vertex's normal
///\param n1z The Z component of the first vertex's normal
///\param c1x The red component of the first vertex's color
///\param c1y The green component of the first vertex's color
///\param c1z The blue component of the first vertex's color
///\param v2x The X coordinate of the second vertex
///\param v2y The Y coordinate of the second vertex
///\param v2z The Z coordinate of the second vertex
///\param n2x The X component of the second vertex's normal
///\param n2y The Y component of the second vertex's normal
///\param n2z The Z component of the second vertex's normal
///\param c2x The red component of the second vertex's color
///\param c2y The green component of the second vertex's color
///\param c2z The blue component of the second vertex's color
///\param den1 The density of the volume at the first vertex
///\param den2 The density of the volume at the second vertex
///\returns A bool indicating success or failure
	inline bool addEdge(unsigned int& id, GLfloat v1x, GLfloat v1y, GLfloat v1z,
		GLfloat n1x, GLfloat n1y, GLfloat n1z,
		GLfloat c1x, GLfloat c1y, GLfloat c1z,
		GLfloat v2x, GLfloat v2y, GLfloat v2z,
		GLfloat n2x, GLfloat n2y, GLfloat n2z,
		GLfloat c2x, GLfloat c2y, GLfloat c2z,
		GLfloat den1, GLfloat den2);
///\fn bool addTriangle(unsigned int v1, unsigned int v2, unsigned int v3)
///\brief Adds a triangle to the contour
///\param v1 The index of the first vertex
///\param v2 The index of the second vertex
///\param v3 The index of the third vertex
	bool addTriangle(unsigned int v1, unsigned int v2, unsigned int v3);
///\fn inline void setVertexAndDensity(GLfloat* array, unsigned int id, GLfloat x, GLfloat y, GLfloat z, GLfloat d)
///\brief Assigns a coordinate and density to a position in a vertex buffer
///\param array A pointer to a vertex buffer
///\param id The position in the buffer
///\param x The X coordinate
///\param y The Y coordinate
///\param z The Z coordinate
///\param d The density
	inline void setVertexAndDensity(GLfloat* array, unsigned int id, GLfloat x, GLfloat y, GLfloat z, GLfloat d);
///\fn inline void setColor(GLfloat* array, unsigned int id, GLfloat r, GLfloat g, GLfloat b)
///\brief Assigns a color to a position in a color buffer
///\param array A pointer to a color buffer
///\param id The position in the buffer
///\param r The red component of the color
///\param g The green component of the color
///\param b The blue component of the color
	inline void setColor(GLfloat* array, unsigned int id, GLfloat r, GLfloat g, GLfloat b);
///\fn inline void setNormal(GLfloat* array, unsigned int id, GLfloat x, GLfloat y, GLfloat z)
///\brief Assigns a normal to a position in a normal buffer
///\param array A pointer to a normal buffer
///\param id The position in the buffer
///\param x The X component of the normal
///\param y The Y component of the normal
///\param z The Z component of the normal
	inline void setNormal(GLfloat* array, unsigned int id, GLfloat x, GLfloat y, GLfloat z);
	
///\fn void CalculateQuadSmoothNormals()
///\brief Calculates normals for the quad mesh
	void CalculateQuadSmoothNormals();

protected:
	void setDefaults();

	// handle vertex arrays
	bool forceAllocateVertexBuffers(unsigned int size);
	bool doubleVertexBuffers();

	// handle triangle arrays
	bool forceAllocateTriangleBuffers(unsigned int size);
	bool doubleTriangleBuffers();
	
	// handle quad arrays
	bool forceAllocateQuadBuffers(unsigned int size);
	bool doubleQuadBuffers();

	bool doubleFloatArray(GLfloat* & array, unsigned int size);
	bool doubleIntArray(GLuint* & array, unsigned int size);

	void doInterpolation(GLfloat isovalue);
	inline void interpArray(GLfloat* v1, GLfloat* v2, GLfloat* n1, GLfloat* n2, GLfloat d1, GLfloat d2, GLfloat isovalue);
	inline void interpArray(GLfloat* v1, GLfloat* v2, GLfloat* n1, GLfloat* n2, GLfloat* c1, GLfloat* c2, GLfloat d1, GLfloat d2, GLfloat isovalue);
	
	void CalculateQuadNormal(float* norm, unsigned int v0, unsigned int v1, unsigned int v2);

	GLfloat* m_Normal1;
	GLfloat* m_Normal2;
	GLfloat* m_Vertex1;
	GLfloat* m_Vertex2;
	GLfloat* m_Color1;
	GLfloat* m_Color2;
	unsigned int m_NumVertices;
	unsigned int m_NumVerticesAllocated;

	GLuint* m_Triangles;
	unsigned int m_NumTriangles;
	unsigned int m_NumTrianglesAllocated;
	
	GLuint* m_Quads;
	unsigned int m_NumQuads;
	unsigned int m_NumQuadsAllocated;

	GLfloat m_Isovalue;

	bool m_HardwareAccelerated;
	bool m_InterpolationDone;
	bool m_UseColors;
	bool m_WireframeRender;
	bool m_SurfWithWire;
};

inline bool ContourGeometry::addEdge(unsigned int& id, GLfloat v1x, GLfloat v1y, GLfloat v1z,
		GLfloat n1x, GLfloat n1y, GLfloat n1z,
		GLfloat c1x, GLfloat c1y, GLfloat c1z,
		GLfloat v2x, GLfloat v2y, GLfloat v2z,
		GLfloat n2x, GLfloat n2y, GLfloat n2z,
		GLfloat c2x, GLfloat c2y, GLfloat c2z,
		GLfloat den1, GLfloat den2)
{
	// add a single edge, doubling the array if necessary
	if (m_NumVertices==m_NumVerticesAllocated) {
		if (!doubleVertexBuffers()) {
			return false;
		}
	}
	setVertexAndDensity(m_Vertex1, m_NumVertices, v1x, v1y, v1z, den1);
	setVertexAndDensity(m_Vertex2, m_NumVertices, v2x, v2y, v2z, den2);
	setNormal(m_Normal1, m_NumVertices, n1x, n1y, n1z);
	setNormal(m_Normal2, m_NumVertices, n2x, n2y, n2z);
	setColor(m_Color1, m_NumVertices, c1x, c1y, c1z);
	setColor(m_Color2, m_NumVertices, c2x, c2y, c2z);
	id = m_NumVertices;
	m_NumVertices++;
	return true;
}

inline bool ContourGeometry::addEdge(unsigned int& id, GLfloat v1x, GLfloat v1y, GLfloat v1z,
		GLfloat n1x, GLfloat n1y, GLfloat n1z,
		GLfloat v2x, GLfloat v2y, GLfloat v2z,
		GLfloat n2x, GLfloat n2y, GLfloat n2z,
		GLfloat den1, GLfloat den2)
{
	// add a single edge, doubling the array if necessary
	if (m_NumVertices==m_NumVerticesAllocated) {
		if (!doubleVertexBuffers()) {
			return false;
		}
	}
	setVertexAndDensity(m_Vertex1, m_NumVertices, v1x, v1y, v1z, den1);
	setVertexAndDensity(m_Vertex2, m_NumVertices, v2x, v2y, v2z, den2);
	setNormal(m_Normal1, m_NumVertices, n1x, n1y, n1z);
	setNormal(m_Normal2, m_NumVertices, n2x, n2y, n2z);
	id = m_NumVertices;
	m_NumVertices++;
	return true;
}

inline void ContourGeometry::setVertexAndDensity(GLfloat* array, unsigned int id, GLfloat x, GLfloat y, GLfloat z, GLfloat d)
{
	// set the values in the array given x,y,z,d
	array[id*4+0] = x;
	array[id*4+1] = y;
	array[id*4+2] = z;
	array[id*4+3] = d;
}

inline void ContourGeometry::setColor(GLfloat* array, unsigned int id, GLfloat r, GLfloat g, GLfloat b)
{
	// set the values in the array given r,g,b
	array[id*3+0] = r;
	array[id*3+1] = g;
	array[id*3+2] = b;
}

inline void ContourGeometry::setNormal(GLfloat* array, unsigned int id, GLfloat x, GLfloat y, GLfloat z)
{
	// set the values in the array given x,y,z
	array[id*3+0] = x;
	array[id*3+1] = y;
	array[id*3+2] = z;
}

inline void ContourGeometry::interpArray(GLfloat* v1, GLfloat* v2, GLfloat* n1, GLfloat* n2, GLfloat d1, GLfloat d2, GLfloat isovalue)
{
	if (d1==d2) {
		//v1[3] = 0.0;
		// leave v1 and n1 the way it is
	}
	else {
		float alpha = (isovalue-d1) / (d2-d1);
		float oneMinusAlpha = 1.0f-alpha;
		// put the result of interpolation inside v1 and n1
		v1[0] = v1[0]*oneMinusAlpha + v2[0]*alpha;
		v1[1] = v1[1]*oneMinusAlpha + v2[1]*alpha;
		v1[2] = v1[2]*oneMinusAlpha + v2[2]*alpha;
		v1[3] = 1.0;
		n1[0] = n1[0]*oneMinusAlpha + n2[0]*alpha;
		n1[1] = n1[1]*oneMinusAlpha + n2[1]*alpha;
		n1[2] = n1[2]*oneMinusAlpha + n2[2]*alpha;
	}

}

inline void ContourGeometry::interpArray(GLfloat* v1, GLfloat* v2, GLfloat* n1, GLfloat* n2, GLfloat* c1, GLfloat* c2, GLfloat d1, GLfloat d2, GLfloat isovalue)
{
	if (d1==d2) {
		// leave v1 and n1 the way it is
	}
	else {
		float alpha = (isovalue-d1) / (d2-d1);
		float oneMinusAlpha = 1.0f-alpha;
		// put the result of interpolation inside v1 and n1
		v1[0] = v1[0]*oneMinusAlpha + v2[0]*alpha;
		v1[1] = v1[1]*oneMinusAlpha + v2[1]*alpha;
		v1[2] = v1[2]*oneMinusAlpha + v2[2]*alpha;
		v1[3] = 1.0;
		n1[0] = n1[0]*oneMinusAlpha + n2[0]*alpha;
		n1[1] = n1[1]*oneMinusAlpha + n2[1]*alpha;
		n1[2] = n1[2]*oneMinusAlpha + n2[2]*alpha;
		c1[0] = c1[0]*oneMinusAlpha + c2[0]*alpha;
		c1[1] = c1[1]*oneMinusAlpha + c2[1]*alpha;
		c1[2] = c1[2]*oneMinusAlpha + c2[2]*alpha;
	}
}

#endif // !defined(AFX_CONTOURGEOMETRY_H__D1ABB49A_F4FB_4D74_B19A_12B037CBB7F7__INCLUDED_)
