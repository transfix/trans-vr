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

// Contour.h: interface for the Contour class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_CONTOUR_H__25453561_B42B_46A3_B5AB_15FA0A287BEF__INCLUDED_)
#define AFX_CONTOUR_H__25453561_B42B_46A3_B5AB_15FA0A287BEF__INCLUDED_

#include <Contouring/ContourGeometry.h>
#include <Contouring/ContourExtractor.h>

class Geometry;

///\class Contour Contour.h
///\author Anthony Thane
///\author John Wiggins
///\brief The Contour class is a container for an isocontour. 
class Contour  
{
public:
	Contour();
	virtual ~Contour();

///\fn void renderContour(const ContourExtractor& contourExtractor);
///\brief Renders a contour. Possibly extracts the contour before rendering.
///\param contourExtractor A ContourExtractor to do the extraction
	void renderContour(const ContourExtractor& contourExtractor);
///\fn void extract(const ContourExtractor& contourExtractor);
///\brief Extracts the contour.
///\param contourExtractor A ContourExtractor to do the extraction
	void extract(const ContourExtractor& contourExtractor);
	//void addToIPoly(const ContourExtractor& contourExtractor, IPolyCntl* cp, const Matrix& matrix, int& nextVert);
///\fn void addToGeometry(const ContourExtractor& contourExtractor, Geometry* geometry, const Matrix& matrix, int& nextVert, int& nextTri);
///\brief Adds the contour to a Geometry object
///\param contourExtractor The ContourExtractor to extract the contour if neccessary
///\param geometry The Geometry object to add the contour to
///\param matrix A Matrix object to hold a scale and translation transformation
///\param nextVert The next available vertex index in the Geometry object
///\param nextTri The next available triangle index in the Geometry object
	void addToGeometry(const ContourExtractor& contourExtractor, Geometry* geometry, const Matrix& matrix, int& nextVert, int& nextTri);
///\fn int getNumVerts(const ContourExtractor& contourExtractor);
///\brief Returns the number of vertices in the isocontour
///\param contourExtractor A ContourExtractor to extract the contour if neccessary
///\return The number of vertices
	int getNumVerts(const ContourExtractor& contourExtractor);
///\fn int getNumTris(const ContourExtractor& contourExtractor);
///\brief Returns the number of triangles in the isocontour
///\param contourExtractor A ContourExtractor to extract the contour if neccessary
///\return The number of triangles
	int getNumTris(const ContourExtractor& contourExtractor);

///\fn void setID(int id);
///\brief Sets the ID of the contour (from IsocontourMap)
///\param id The new id of the contour
	void setID(int id);
///\fn int getID();
///\brief Returns the ID of the contour
///\return The ID of the contour
	int getID();

///\fn void setIsovalue(float isovalue);
///\brief Sets the isovalue of the contour
///\param isovalue The isovalue
	void setIsovalue(float isovalue);
///\fn void setSingleColor(float R, float G, float B, bool clobber);
///\brief Assigns a single color to the contour's geometry
///\param R The red component of the new color
///\param G The green component of the new color
///\param B The blue component of the new color
///\param clobber If true, assigns the color to geometry that was already extracted. If false, the color will be set the next time the contour is extracted.
	void setSingleColor(float R, float G, float B, bool clobber);
///\fn void resetContour();
///\brief Forces contour extraction
	void resetContour();
///\fn void setWireframeMode(bool state);
///\brief Toggles wireframe rendering for this contour
///\param state True -> wireframe rendering is enabled; False -> wireframe rendering is disabled.
	void setWireframeMode(bool state);

	void setSurfWithWire(bool state);
	
///\fn bool useColors();
///\brief Returns true if the contour renders in color
///\return A boolean
	bool useColors();


protected:
	void setDefaults();

	int m_ID;
	float m_Isovalue;
	float m_R,m_G,m_B;
	bool m_ContourReady;
	ContourGeometry m_ContourGeometry;

};

#endif // !defined(AFX_CONTOUR_H__25453561_B42B_46A3_B5AB_15FA0A287BEF__INCLUDED_)
