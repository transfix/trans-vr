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

// WireCubeRenderable.h: interface for the WireCubeRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_WIRECUBERENDERABLE_H__66B5A637_C149_4E8F_8318_0B4E3BF5CD84__INCLUDED_)
#define AFX_WIRECUBERENDERABLE_H__66B5A637_C149_4E8F_8318_0B4E3BF5CD84__INCLUDED_

#include <VolumeWidget/GeometryRenderable.h>
#include <cvcraw_geometry/Geometry.h>
#include <VolumeWidget/Extents.h>

//class RenderableView;

///\class WireCubeRenderable WireCubeRenderable.h
///\brief This class represents the volume bounding box wireframe in Volume
///	Rover.
///\author Anthony Thane
///\author John Wiggins
class WireCubeRenderable : public GeometryRenderable  
{
public:
	WireCubeRenderable();
	virtual ~WireCubeRenderable();

///\fn void setAspectRatio(double x, double y, double z)
///\brief This function sets the size of the bounding box.
///\param x The size in X
///\param y The size in Y
///\param z The size in Z
	void setAspectRatio(double x, double y, double z);
	
///\fn void setColor(float r, float g, float b)
///\brief This function sets the color of the bounding box.
///\param r The red component of the color
///\param g The green component of the color
///\param b The blue component of the color
	void setColor(float r, float g, float b);

	/*virtual bool initForContext();
	virtual bool deinitForContext();
	virtual bool render();*/

protected:
	void prepareGeometry();

	Extents m_Boundary;

	bool m_GeometriesAllocated;
	float m_R,m_G,m_B;
	Geometry m_WireCubeGeometry;

};

#endif // !defined(AFX_WIRECUBERENDERABLE_H__66B5A637_C149_4E8F_8318_0B4E3BF5CD84__INCLUDED_)
