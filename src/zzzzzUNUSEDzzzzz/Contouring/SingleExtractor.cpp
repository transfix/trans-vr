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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
//
// SingleExtractor.cpp: implementation of the SingleExtractor class.
//
//////////////////////////////////////////////////////////////////////

#include <Contouring/SingleExtractor.h>
#include <Contouring/ContourGeometry.h>
#include <Contouring/MarchingCubesBuffers.h>
#include <VolumeWidget/Matrix.h>


// arand, 5-5-2011: these are actually unused
//#include <duallib/duallib.h>
//#include <duallib/geoframe.h>

#include <stdio.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SingleExtractor::SingleExtractor()
{
	setDefaults();
}

SingleExtractor::~SingleExtractor()
{
#ifdef DUALLIB
	delete m_Octree;
#else
	delete m_Buffers;
#endif
}

#ifdef DUALLIB
void SingleExtractor::extractContour(ContourGeometry* contourGeometry, float isovalue, float R, float G, float B) const
{
	geoframe geom;
	int i,j;
	float ext[6];

	// set the isovalue
	m_Octree->set_isovalue(isovalue);
	// extract contour from the found cells, 
	m_Octree->polygonize(geom);

	printf("duallib contour: %d verts / %d quads\n", geom.numverts, geom.numquads);

	// convert geoframe to ContourGeometry
	contourGeometry->destroyVertexBuffers();
	contourGeometry->destroyQuadBuffers();
	contourGeometry->allocateVertexBuffers(geom.numverts);
	contourGeometry->allocateQuadBuffers(geom.numquads);

	ext[0] = geom.verts[0][0];
	ext[1] = geom.verts[0][0];
	ext[2] = geom.verts[0][1];
	ext[3] = geom.verts[0][1];
	ext[4] = geom.verts[0][2];
	ext[5] = geom.verts[0][2];
	for (i=0; i < geom.numverts; i++) {
		for (j=0; j < 3; j++) {
			if (ext[j*2+0] > geom.verts[i][j]) ext[j*2+0] = geom.verts[i][j];
			else if (ext[j*2+1] < geom.verts[i][j]) ext[j*2+1] = geom.verts[i][j];
		}

		contourGeometry->addQuadVertex(
				geom.verts[i][0], geom.verts[i][1], geom.verts[i][2],
				geom.normals[i][0], geom.normals[i][1], geom.normals[i][2],
				R, G, B);
	}

	for (i=0; i < geom.numquads; i++)
		contourGeometry->addQuad(geom.quads[i][0],geom.quads[i][1],
		                         geom.quads[i][2], geom.quads[i][3]);

	contourGeometry->CalculateQuadSmoothNormals();

	printf("duallib contour extents = [%f,%f,%f]->[%f,%f,%f]\n",
			ext[0],ext[2],ext[4], ext[1],ext[3],ext[5]);
}
#else
void SingleExtractor::extractContour(ContourGeometry* contourGeometry, float isovalue, float R, float G, float B) const
{
	contourGeometry->allocateVertexBuffers(m_Width*m_Height);
	contourGeometry->allocateTriangleBuffers(m_Width*m_Height);
	//contourGeometry->setUseColors(false);
	contourGeometry->setUseColors(true);
	contourGeometry->setIsovalue(isovalue);
	unsigned int numCellsX = m_Width-1;
	unsigned int numCellsY = m_Height-1;
	unsigned int numCellsZ = m_Depth-1;

	bool cacheAvailable[12];

	unsigned int offsetTable[8];
	unsigned int withinSliceOffsetTable[8];
	unsigned int cacheOffsetTable[12];
	buildOffsetTable(offsetTable, m_Width, m_Height);
	buildWithinSliceOffsetTable(withinSliceOffsetTable, m_Width);
	buildEdgeCacheOffsetTable(cacheOffsetTable, m_Width);

	unsigned int cubeCase;

	unsigned int i,j,k, voxelOffset, offsetInSlice, newEdge, edgeCounter, triangleCounter, edgeID;

	// the new vertex
	GLfloat v1x, v1y, v1z, v2x, v2y, v2z, den1, den2;
	GLfloat n1x, n1y, n1z, n2x, n2y, n2z;

	// edges involved in a triangle
	unsigned int e1,e2,e3;
	unsigned int rowStart, inSliceRowStart;

	classifyVertices(0, m_Buffers->m_VertClassifications[0], isovalue);

	for (k=0; k<numCellsZ; k++) { // for every slice
		classifyVertices(k+1, m_Buffers->m_VertClassifications[1], isovalue);

		for (j=0; j<numCellsY; j++) { // for every row
			rowStart = m_Width*m_Height*k+m_Width*j;
			inSliceRowStart = m_Width*j;
			for (i=0; i<numCellsX; i++) { // for every cell
				if (i==0 || i==1 || i==numCellsX-1) {
					// isCached only changes at the start and end of a scan line
					// as well as the second voxel of every scan line
					isCached(cacheAvailable, i,j,k,m_Width,m_Height);
				}
				voxelOffset = rowStart+i;
				offsetInSlice = inSliceRowStart+i;
				cubeCase = determineCase(m_Data, withinSliceOffsetTable, offsetInSlice, isovalue);
				if (cubeCase!=0 && cubeCase!=255) {
					// for each edge involved
					for (edgeCounter=0; edgeCounter<cubeedges[cubeCase][0]; edgeCounter++) {
						edgeID = cubeedges[cubeCase][edgeCounter+1];
						// if the edge isnt cached yet, cache it
						if (!cacheAvailable[edgeID]) {

							// add the edge and get its index
							/*computeVertFromOffset(v1x, v1y, v1z, offsetTable[edges[edgeID][0]],
								m_Width, m_Height);*/
							/*computeVertFromOffset(v2x, v2y, v2z, offsetTable[edges[edgeID][1]],
								m_Width, m_Height);*/
							v1x = (GLfloat)(i + verts[edges[edgeID][0]][2]);
							v1y = (GLfloat)(j + verts[edges[edgeID][0]][1]);
							v1z = (GLfloat)(k + verts[edges[edgeID][0]][0]);
							v2x = (GLfloat)(i + verts[edges[edgeID][1]][2]);
							v2y = (GLfloat)(j + verts[edges[edgeID][1]][1]);
							v2z = (GLfloat)(k + verts[edges[edgeID][1]][0]);
							den1 = m_Data[voxelOffset+offsetTable[edges[edgeID][0]]];
							den2 = m_Data[voxelOffset+offsetTable[edges[edgeID][1]]];
							getNormal(m_Data, 
								i + verts[edges[edgeID][0]][2], 
								j + verts[edges[edgeID][0]][1], 
								k + verts[edges[edgeID][0]][0], 
								n1x, n1y, n1z);
							getNormal(m_Data, 
								i + verts[edges[edgeID][1]][2], 
								j + verts[edges[edgeID][1]][1], 
								k + verts[edges[edgeID][1]][0], 
								n2x, n2y, n2z);
							// no color
							//contourGeometry->addEdge(newEdge, v1x, v1y, v1z, n1x, n1y, n1z,
							//	v2x, v2y, v2z, n2x, n2y, n2z, den1, den2);
							// color
							contourGeometry->addEdge(newEdge, v1x, v1y, v1z, n1x, n1y, n1z,
								R,G,B, v2x, v2y, v2z, n2x, n2y, n2z, R,G,B, den1, den2);

							// The location in the cache is determined by two table lookups:
							// FromEdgeToChacheLookup[edgeID] finds which table the edge is cached in
							// offsetInSlice+cacheOffsetTable[edgeID] determines the location in that table
							m_Buffers->m_EdgeCaches[FromEdgeToChacheLookup[edgeID]][offsetInSlice+cacheOffsetTable[edgeID]] = newEdge;

						}
					}
					// All appropriate edges are now cached
					// Build the triangles, using indexes from the cache
					// for each triangle
					
					for (triangleCounter=0; triangleCounter<cubes[cubeCase][0]; triangleCounter++) {
						e1 = cubes[cubeCase][triangleCounter*3+1];
						e2 = cubes[cubeCase][triangleCounter*3+2];
						e3 = cubes[cubeCase][triangleCounter*3+3];
						
						contourGeometry->addTriangle(
							m_Buffers->m_EdgeCaches[FromEdgeToChacheLookup[e1]][offsetInSlice+cacheOffsetTable[e1]], 
							m_Buffers->m_EdgeCaches[FromEdgeToChacheLookup[e2]][offsetInSlice+cacheOffsetTable[e2]], 
							m_Buffers->m_EdgeCaches[FromEdgeToChacheLookup[e3]][offsetInSlice+cacheOffsetTable[e3]]);
					}

				}

			}

		}
		// swap the edge buffers
		m_Buffers->swapEdgeBuffers();
	}

}
#endif

static inline double maxOfThree(double n1, double n2, double n3) {
	double max = (n1>n2?n1:n2);
	return (max>n3?max:n3);
}

#ifdef DUALLIB
void SingleExtractor::setData(unsigned char* data,
							   unsigned int width, unsigned int height, unsigned int depth,
							   double aspectX, double aspectY, double aspectZ,
							   double subMinX, double subMinY, double subMinZ,
							   double subMaxX, double subMaxY, double subMaxZ)
{
	int dim[3];
	float orig[3], span[3];

	dim[0] = width;
	dim[1] = height;
	dim[2] = depth;

	orig[0] = (float)subMinX;
	orig[1] = (float)subMinY;
	orig[2] = (float)subMinZ;

	span[0] = (float)(aspectX / (width-1));
	span[1] = (float)(aspectY / (height-1));
	span[2] = (float)(aspectZ / (depth-1));

	printf("duallib init: dim = [%d,%d,%d] / orig = [%f,%f,%f] / span = [%f,%f,%f]\n", dim[0],dim[1],dim[2], orig[0],orig[1],orig[2], span[0],span[1],span[2]);
	m_Octree->Octree_init(data, dim, orig, span);
	printf("aspect = [%f,%f,%f] / subMax = [%f,%f,%f]\n", aspectX,aspectY,aspectZ, subMaxX,subMaxY,subMaxZ);
}
#else
void SingleExtractor::setData(unsigned char* data,
							   unsigned int width, unsigned int height, unsigned int depth,
							   double aspectX, double aspectY, double aspectZ,
							   double subMinX, double subMinY, double subMinZ,
							   double subMaxX, double subMaxY, double subMaxZ)
{
	m_Data = data;
	m_Width = width;
	m_Height = height;
	m_Depth = depth;
	//m_Buffers = new MarchingCubesBuffers();
	m_Buffers->allocateEdgeBuffers(width, height);
}
#endif

void SingleExtractor::setDefaults()
{
	// set default values for all member variables
	m_Data = 0;
	m_Width = 0;
	m_Height = 0;
	m_Depth = 0;

#ifdef DUALLIB
	m_Octree = new Octree;
#else
	m_Buffers = new MarchingCubesBuffers();
#endif
}

void SingleExtractor::classifyVertices(unsigned int k, unsigned int* cacheMemory, GLfloat isovalue) const
{
	unsigned int i;
	unsigned int sourceOffset = k*m_Width*m_Height;
	unsigned int destOffset = 0;

	for (i=0; i<m_Width*m_Height; i++) {
		cacheMemory[destOffset++] = ((GLfloat)m_Data[sourceOffset++] < isovalue);
	}
}

inline unsigned int SingleExtractor::determineCase(unsigned char* data, unsigned int* offsetTable, unsigned int index, GLfloat isovalue) const
{
	unsigned int cubeCase = 0;

	// determine the marching cube case
/*
	if ((GLfloat)data[index+offsetTable[0]] < isovalue) cubeCase |= 1<<0;
	if ((GLfloat)data[index+offsetTable[1]] < isovalue) cubeCase |= 1<<1;
	if ((GLfloat)data[index+offsetTable[2]] < isovalue) cubeCase |= 1<<2;
	if ((GLfloat)data[index+offsetTable[3]] < isovalue) cubeCase |= 1<<3;
	if ((GLfloat)data[index+offsetTable[4]] < isovalue) cubeCase |= 1<<4;
	if ((GLfloat)data[index+offsetTable[5]] < isovalue) cubeCase |= 1<<5;
	if ((GLfloat)data[index+offsetTable[6]] < isovalue) cubeCase |= 1<<6;
	if ((GLfloat)data[index+offsetTable[7]] < isovalue) cubeCase |= 1<<7;
*/
	/*
	cubeCase |= ((GLfloat)data[index+offsetTable[0]] < isovalue)<<0;
	cubeCase |= ((GLfloat)data[index+offsetTable[1]] < isovalue)<<1;
	cubeCase |= ((GLfloat)data[index+offsetTable[2]] < isovalue)<<2;
	cubeCase |= ((GLfloat)data[index+offsetTable[3]] < isovalue)<<3;
	cubeCase |= ((GLfloat)data[index+offsetTable[4]] < isovalue)<<4;
	cubeCase |= ((GLfloat)data[index+offsetTable[5]] < isovalue)<<5;
	cubeCase |= ((GLfloat)data[index+offsetTable[6]] < isovalue)<<6;
	cubeCase |= ((GLfloat)data[index+offsetTable[7]] < isovalue)<<7;
	*/
#ifndef DUALLIB
	cubeCase |= (m_Buffers->m_VertClassifications[0][index+offsetTable[0]])<<0;
	cubeCase |= (m_Buffers->m_VertClassifications[0][index+offsetTable[1]])<<1;
	cubeCase |= (m_Buffers->m_VertClassifications[1][index+offsetTable[2]])<<2;
	cubeCase |= (m_Buffers->m_VertClassifications[1][index+offsetTable[3]])<<3;
	cubeCase |= (m_Buffers->m_VertClassifications[0][index+offsetTable[4]])<<4;
	cubeCase |= (m_Buffers->m_VertClassifications[0][index+offsetTable[5]])<<5;
	cubeCase |= (m_Buffers->m_VertClassifications[1][index+offsetTable[6]])<<6;
	cubeCase |= (m_Buffers->m_VertClassifications[1][index+offsetTable[7]])<<7;
#endif
	return cubeCase;
}

void SingleExtractor::calcOffsets(unsigned int* offsets, unsigned int width, unsigned int height)
{
	// calculate the table of offsets given the width and height
	offsets[0] = verts[0][0]*width*height + verts[0][1]*width + verts[0][2];
	offsets[1] = verts[1][0]*width*height + verts[1][1]*width + verts[1][2];
	offsets[2] = verts[2][0]*width*height + verts[2][1]*width + verts[2][2];
	offsets[3] = verts[3][0]*width*height + verts[3][1]*width + verts[3][2];
	offsets[4] = verts[4][0]*width*height + verts[4][1]*width + verts[4][2];
	offsets[5] = verts[5][0]*width*height + verts[5][1]*width + verts[5][2];
	offsets[6] = verts[6][0]*width*height + verts[6][1]*width + verts[6][2];
	offsets[7] = verts[7][0]*width*height + verts[7][1]*width + verts[7][2];
	offsets[8] = verts[8][0]*width*height + verts[8][1]*width + verts[8][2];
}


