/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

// RGBAExtractor.h: interface for the RGBAExtractor class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RGBAEXTRACTOR_H__79C7CB25_6CC3_4FF6_870D_AA1018CEABF5__INCLUDED_)
#define AFX_RGBAEXTRACTOR_H__79C7CB25_6CC3_4FF6_870D_AA1018CEABF5__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <boost/scoped_ptr.hpp>

#include <Contouring/ContourExtractor.h>

///\class RGBAExtractor RGBAExtractor.h
///\author Anthony Thane
///\brief RGBAExtractor is a ContourExtractor for RGBA volumes.
class RGBAExtractor : public ContourExtractor  
{
public:
	RGBAExtractor();
	virtual ~RGBAExtractor();

	virtual void extractContour(ContourGeometry* contourGeometry, float isovalue, float R, float G, float B) const;

///\fn void setData(unsigned char* data, unsigned char* red, unsigned char* green, unsigned char* blue, unsigned int width, unsigned int height, unsigned int depth, double aspectX, double aspectY, double aspectZ, double subMinX, double subMinY, double subMinZ, double subMaxX, double subMaxY, double subMaxZ)
///\brief Assigns volume data to extract a contour from
///\param data The function values
///\param red The red color components of the function values
///\param green The green color components of the function values
///\param blue The blue color components of the function values
///\param width The width of the volume
///\param height The height of the volume
///\param depth The depth of the volume
///\param aspectX The size of the volume bounding box along the X axis
///\param aspectY The size of the volume bounding box along the Y axis
///\param aspectZ The size of the volume bounding box along the Z axis
///\param subMinX The subvolume extent minimum X coordinate
///\param subMinY The subvolume extent minimum Y coordinate
///\param subMinZ The subvolume extent minimum Z coordinate
///\param subMaxX The subvolume extent maximum X coordinate
///\param subMaxY The subvolume extent maximum Y coordinate
///\param subMaxZ The subvolume extent maximum Z coordinate
	void setData(unsigned char* data, unsigned char* red, unsigned char* green, unsigned char* blue,
		unsigned int width, unsigned int height, unsigned int depth,
		double aspectX, double aspectY, double aspectZ,
		double subMinX, double subMinY, double subMinZ,
		double subMaxX, double subMaxY, double subMaxZ);
protected:
	void setDefaults();

	void classifyVertices(unsigned int k, unsigned int* cacheMemory, GLfloat isovalue) const;
	inline void getNormal(unsigned char* data, unsigned int i, unsigned int j, unsigned int k, float& nx, float& ny, float& nz) const;
	inline void isCached(bool* cachedTable, unsigned int i, unsigned int j, unsigned int k, unsigned int width, unsigned int height) const;
	inline unsigned int determineCase(unsigned char* data, unsigned int* offsetTable, unsigned int index, GLfloat isovalue) const;
	inline void buildOffsetTable(unsigned int* table, unsigned int width, unsigned int height) const;
	inline void buildWithinSliceOffsetTable(unsigned int* table, unsigned int width) const;
	inline void buildEdgeCacheOffsetTable(unsigned int * table, unsigned int width) const;
	inline void computeVertFromOffset(GLfloat& vx, GLfloat& vy, GLfloat& vz, unsigned int offset, unsigned int width, unsigned int height) const;

	void calcOffsets(unsigned int* offsets, unsigned int width, unsigned int height);

	unsigned char* m_Data;
	unsigned char* m_Red;
	unsigned char* m_Green;
	unsigned char* m_Blue;

	// dimensions of the data
	unsigned int m_Width;
	unsigned int m_Height;
	unsigned int m_Depth;

	// buffers used to speed up marching cubes
	//MarchingCubesBuffers* m_Buffers;
	boost::scoped_ptr<MarchingCubesBuffers> m_Buffers;
};


inline void RGBAExtractor::getNormal(unsigned char* data, 
										unsigned int i, unsigned int j, unsigned int k,
										float& nx, float& ny, float& nz) const
{
	double dx,dy,dz,length;
	int negZOffset, negYOffset, negXOffset;
	int posZOffset, posYOffset, posXOffset;
	double scaleX = 1.0, scaleY = 1.0, scaleZ = 1.0;
	unsigned int inputindex;
	if (k==0) { // border offset
		negZOffset = 0; posZOffset =  m_Width*m_Height;
	}
	else if (k==m_Depth-1) { // border offset
		negZOffset = -(int)(m_Width*m_Height); posZOffset = 0;
	}
	else { // normal offset
		negZOffset = -(int)(m_Width*m_Height); posZOffset =  m_Width*m_Height;
		scaleZ = 0.5;
	}

	if (j==0) { // border offset
		negYOffset = 0; posYOffset =  m_Width;
	}
	else if (j==m_Height-1) { // border offset
		negYOffset = -(int)m_Width; posYOffset = 0;
	}				
	else { // normal offset
		negYOffset = -(int)m_Width; posYOffset =  m_Width;
		scaleY = 0.5;
	}

	if (i==0) { // border offset
		negXOffset = 0; posXOffset =  1;
	}
	else if (i==m_Width-1) { // border offset
		negXOffset = -1; posXOffset = 0;
	}				
	else { // normal offset
		negXOffset = -1; posXOffset =  1;
		scaleX = 0.5;
	}

	inputindex = k*m_Width*m_Height+j*m_Width+i;
	dx = (double)(data[inputindex+negXOffset] - data[inputindex+posXOffset]) * scaleX;
	dy = (double)(data[inputindex+negYOffset] - data[inputindex+posYOffset]) * scaleY;
	dz = (double)(data[inputindex+negZOffset] - data[inputindex+posZOffset]) * scaleZ;
	length = sqrt(dx*dx+dy*dy+dz*dz);
	if (length!=0) {
		nx = (float)(dx/length);
		ny = (float)(dy/length);
		nz = (float)(dz/length);
	}

}

inline void RGBAExtractor::buildEdgeCacheOffsetTable(unsigned int * table, unsigned int width) const
{

	// calculates where edges are cached in their respective table
	// is an offset from k*width*height + j*width + i

	// edge 0
	table[0] = 0;

	// edge 1
	table[1] = 1;

	// edge 2
	table[2] = 0;

	// edge 3
	table[3] = 0;

	// edge 4
	table[4] = width;

	// edge 5
	table[5] = width+1;

	// edge 6
	table[6] = width;

	// edge 7
	table[7] = width;

	// edge 8
	table[8] = 0;

	// edge 9
	table[9] = 1;

	// edge 10
	table[10] = 0;

	// edge 11
	table[11] = 1;

}

inline void RGBAExtractor::isCached(bool* cachedTable, unsigned int i, unsigned int j, unsigned int k, unsigned int width, unsigned int height) const
{

	unsigned int dummy = 0;
	dummy = width;
	dummy = height;	
	// bool ret = false;
	// ret = j!=0 || k!=0;

	// edge 0
	cachedTable[0] = j!=0 || k!=0;

	// edge 1
	cachedTable[1] = j!=0;

	// edge 2
	cachedTable[2] = j!=0;

	// edge 3
	cachedTable[3] = i!=0 || j!=0;

	// edge 4
	cachedTable[4] = k!=0;

	// edge 5
	cachedTable[5] = false;

	// edge 6
	cachedTable[6] = false;

	// edge 7
	cachedTable[7] = i!=0;

	// edge 8
	cachedTable[8] = k!=0 || i!=0;

	// edge 9
	cachedTable[9] = k!=0;

	// edge 10
	cachedTable[10] = i!=0;

	// edge 11
	cachedTable[11] = false;
	
}


inline void RGBAExtractor::buildOffsetTable(unsigned int* table, unsigned int width, unsigned int height) const
{
	// determine the offset of each vertex of the cube give the width and height of the data
	unsigned int c;
	for (c=0; c<8; c++) {
		table[c] = verts[c][2] + width*verts[c][1] + width*height*verts[c][0];
	}
}

inline void RGBAExtractor::buildWithinSliceOffsetTable(unsigned int* table, unsigned int width) const
{
	// determine the offset of each vertex of the cube give the width and height of the data
	unsigned int c;
	for (c=0; c<8; c++) {
		table[c] = verts[c][2] + width*verts[c][1];
	}
}

inline void RGBAExtractor::computeVertFromOffset(GLfloat& vx, GLfloat& vy, GLfloat& vz, unsigned int offset, unsigned int width, unsigned int height) const
{
	vx = (GLfloat)(offset % (width));
	vy = (GLfloat)((offset/width) % height);
	vz = (GLfloat)(offset / (width*height));

}

#endif // !defined(AFX_RGBAEXTRACTOR_H__79C7CB25_6CC3_4FF6_870D_AA1018CEABF5__INCLUDED_)
