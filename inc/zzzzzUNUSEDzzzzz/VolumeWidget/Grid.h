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

// Grid.h: interface for the Grid class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GRID_H__966FA4B1_8684_46C0_928C_7ACF0DEFEBE4__INCLUDED_)
#define AFX_GRID_H__966FA4B1_8684_46C0_928C_7ACF0DEFEBE4__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeWidget/Quaternion.h>
#include <VolumeWidget/Vector.h>
#include <cvcraw_geometry/Geometry.h>

class Grid  
{
public:
	Grid();
	virtual ~Grid();

	void InitOptions();

	Vector Intersect(const Ray& ray) const;

	void SetGridSize(float size);
	void SetGridSpacing(float spacing);
	void SetMajorSpacing(unsigned int majorSpacing);
	void SetShowAxis(bool axis);
	void SetMajorGridColor( float r, float g, float b );
	void SetMinorGridColor( float r, float g, float b );
	void SetXAxisColor( float r, float g, float b );
	void SetYAxisColor( float r, float g, float b );
	void SetZAxisColor( float r, float g, float b );
	
	float GetGridSize() const;
	float GetGridSpacing() const;
	unsigned int GetMajorSpacing() const;
	bool GetShowAxis() const;
	const float* GetMajorGridColor() const;
	const float* GetMinorGridColor() const;
	const float* GetXAxisColor() const;
	const float* GetYAxisColor() const;
	const float* GetZAxisColor() const;

	Geometry* GetMajorLines();
	Geometry* GetMinorLines();

	void setGridPerspective();
	void setGridTop();
	void setGridBottom();
	void setGridRight();
	void setGridLeft();
	void setGridFront();
	void setGridBack();

	static const float defaultGridSpacing;
	static const float defaultGridExtent;
	static const unsigned int defaultMajorSpacing;
	static const bool defaultbShowAxis;
	static const float defaultMajorGridColor[3];
	static const float defaultMinorGridColor[3];
	static const float defaultXAxisColor[3];
	static const float defaultYAxisColor[3];
	static const float defaultZAxisColor[3];

protected:
	void PrepareGeometry();
	unsigned int PrepareMajorLines();
	void PrepareMinorLines(unsigned int numberofmajorlines);
	void TransformGrid();

	float m_GridExtent;
	float m_GridSpacing;
	unsigned int m_MajorSpacing;
	bool m_bShowAxis;
	bool m_bOptionChanged;
	float m_MajorGridColor[3];
	float m_MinorGridColor[3];
	float m_XAxisColor[3];
	float m_YAxisColor[3];
	float m_ZAxisColor[3];

	Geometry m_MinorLines;
	Geometry m_MajorLines;

	Quaternion m_Orientation;
	Vector m_Origin;

private:
	inline unsigned int MakeHorizontalLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines);
	inline unsigned int MakeVerticalLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines);
	inline unsigned int MakeHorizontalAxisLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines);
	inline unsigned int MakeVerticalAxisLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines);
	inline unsigned int MakeDepthAxisLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines);
};

inline unsigned int Grid::MakeHorizontalLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines)
{
	unsigned int v1 = nextvertex;
	nextvertex++;
	unsigned int v2 = nextvertex;
	nextvertex++;
	lines->m_LineVerts[v1*3+0] = low;
	lines->m_LineVerts[v1*3+1] = position;
	lines->m_LineVerts[v1*3+2] = 0.0f;
	lines->m_LineVerts[v2*3+0] = high;
	lines->m_LineVerts[v2*3+1] = position;
	lines->m_LineVerts[v2*3+2] = 0.0f;
	lines->m_Lines[linenumber*2+0] = v1;
	lines->m_Lines[linenumber*2+1] = v2;
	return nextvertex;
}

inline unsigned int Grid::MakeVerticalLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines)
{
	unsigned int v1 = nextvertex;
	nextvertex++;
	unsigned int v2 = nextvertex;
	nextvertex++;
	lines->m_LineVerts[v1*3+0] = position;
	lines->m_LineVerts[v1*3+1] = low;
	lines->m_LineVerts[v1*3+2] = 0.0f;
	lines->m_LineVerts[v2*3+0] = position;
	lines->m_LineVerts[v2*3+1] = high;
	lines->m_LineVerts[v2*3+2] = 0.0f;
	lines->m_Lines[linenumber*2+0] = v1;
	lines->m_Lines[linenumber*2+1] = v2;
	return nextvertex;
}

inline unsigned int Grid::MakeHorizontalAxisLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines)
{
	unsigned int v1 = nextvertex;
	nextvertex++;
	unsigned int v2 = nextvertex;
	nextvertex++;
	lines->m_LineVerts[v1*3+0] = low;
	lines->m_LineVerts[v1*3+1] = position;
	lines->m_LineVerts[v1*3+2] = 0.0f;
	lines->m_LineVerts[v2*3+0] = high;
	lines->m_LineVerts[v2*3+1] = position;
	lines->m_LineVerts[v2*3+2] = 0.0f;

	lines->m_LineColors[v1*3+0] = m_XAxisColor[0];
	lines->m_LineColors[v1*3+1] = m_XAxisColor[1];
	lines->m_LineColors[v1*3+2] = m_XAxisColor[2];
	lines->m_LineColors[v2*3+0] = m_XAxisColor[0];
	lines->m_LineColors[v2*3+1] = m_XAxisColor[1];
	lines->m_LineColors[v2*3+2] = m_XAxisColor[2];

	lines->m_Lines[linenumber*2+0] = v1;
	lines->m_Lines[linenumber*2+1] = v2;
	return nextvertex;
}

inline unsigned int Grid::MakeVerticalAxisLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines)
{
	unsigned int v1 = nextvertex;
	nextvertex++;
	unsigned int v2 = nextvertex;
	nextvertex++;
	lines->m_LineVerts[v1*3+0] = position;
	lines->m_LineVerts[v1*3+1] = low;
	lines->m_LineVerts[v1*3+2] = 0.0f;
	lines->m_LineVerts[v2*3+0] = position;
	lines->m_LineVerts[v2*3+1] = high;
	lines->m_LineVerts[v2*3+2] = 0.0f;

	lines->m_LineColors[v1*3+0] = m_YAxisColor[0];
	lines->m_LineColors[v1*3+1] = m_YAxisColor[1];
	lines->m_LineColors[v1*3+2] = m_YAxisColor[2];
	lines->m_LineColors[v2*3+0] = m_YAxisColor[0];
	lines->m_LineColors[v2*3+1] = m_YAxisColor[1];
	lines->m_LineColors[v2*3+2] = m_YAxisColor[2];
	
	lines->m_Lines[linenumber*2+0] = v1;
	lines->m_Lines[linenumber*2+1] = v2;
	return nextvertex;
}

inline unsigned int Grid::MakeDepthAxisLine(unsigned int linenumber, unsigned int nextvertex, float position, float low, float high, Geometry* lines)
{
	float dummy = 0.f;
	dummy = position;
	unsigned int v1 = nextvertex;
	nextvertex++;
	unsigned int v2 = nextvertex;
	nextvertex++;
	lines->m_LineVerts[v1*3+0] = 0.0f;
	lines->m_LineVerts[v1*3+1] = 0.0f;
	lines->m_LineVerts[v1*3+2] = low;
	lines->m_LineVerts[v2*3+0] = 0.0f;
	lines->m_LineVerts[v2*3+1] = 0.0f;
	lines->m_LineVerts[v2*3+2] = high;

	lines->m_LineColors[v1*3+0] = m_ZAxisColor[0];
	lines->m_LineColors[v1*3+1] = m_ZAxisColor[1];
	lines->m_LineColors[v1*3+2] = m_ZAxisColor[2];
	lines->m_LineColors[v2*3+0] = m_ZAxisColor[0];
	lines->m_LineColors[v2*3+1] = m_ZAxisColor[1];
	lines->m_LineColors[v2*3+2] = m_ZAxisColor[2];

	lines->m_Lines[linenumber*2+0] = v1;
	lines->m_Lines[linenumber*2+1] = v2;
	return nextvertex;
}

#endif // !defined(AFX_GRID_H__966FA4B1_8684_46C0_928C_7ACF0DEFEBE4__INCLUDED_)
