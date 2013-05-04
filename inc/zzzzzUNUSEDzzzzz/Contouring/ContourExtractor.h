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

// ContourExtractor.h: interface for the ContourExtractor class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_CONTOUREXTRACTOR_H__9E1013AE_AF22_4CD9_81E7_B7D7BE2ECEA5__INCLUDED_)
#define AFX_CONTOUREXTRACTOR_H__9E1013AE_AF22_4CD9_81E7_B7D7BE2ECEA5__INCLUDED_

#include <Contouring/ContourGeometry.h>
#include <math.h>
#include <Contouring/cubes.h>

class Matrix;
class MarchingCubesBuffers;

///\class ContourExtractor ContourExtractor.h
///\author Anthony Thane
///\author John Wiggins
///\brief ContourExtractor is an abstract base class for objects that extract
/// isocontours.
class ContourExtractor  
{
public:
	ContourExtractor();
	virtual ~ContourExtractor();

///\fn virtual void extractContour(ContourGeometry* contourGeometry, float isovalue, float R, float G, float B) const
///\brief Extracts an isocontour
///\param contourGeometry The ContourGeometry object to load with the extracted geometry
///\param isovalue The isovalue of the contour to be extracted
///\param R The red component of the contour's color (may be ignored)
///\param G The green component of the contour's color (may be ignored)
///\param B The blue component of the contour's color (may be ignored)
	virtual void extractContour(ContourGeometry* contourGeometry, float isovalue, float R, float G, float B) const = 0;

protected:

};




#endif // !defined(AFX_CONTOUREXTRACTOR_H__9E1013AE_AF22_4CD9_81E7_B7D7BE2ECEA5__INCLUDED_)
