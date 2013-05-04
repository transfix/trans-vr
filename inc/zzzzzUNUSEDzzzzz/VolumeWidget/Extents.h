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

// Extents.h: interface for the Extents class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_EXTENTS_H__541859B1_96FC_453D_8524_1D6789C4F864__INCLUDED_)
#define AFX_EXTENTS_H__541859B1_96FC_453D_8524_1D6789C4F864__INCLUDED_

#include <VolumeWidget/Vector.h>

///\class Extents Extents.h
///\author Anthony Thane
///\brief The Extents class defines a bounding box for a volume
class Extents  
{
public:
	Extents();
///\fn Extents(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
///\brief The constructor
///\param xMin The minimum X coordinate
///\param xMax The maximum X coordinate
///\param yMin The minimum Y coordinate
///\param yMax The maximum Y coordinate
///\param zMin The minimum Z coordinate
///\param zMax The maximum Z coordinate
	Extents(
		double xMin, double xMax,
		double yMin, double yMax,
		double zMin, double zMax
		);
	virtual ~Extents();

///\fn void setExtents(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
///\brief Assigns a new bounding box
///\param xMin The minimum X coordinate
///\param xMax The maximum X coordinate
///\param yMin The minimum Y coordinate
///\param yMax The maximum Y coordinate
///\param zMin The minimum Z coordinate
///\param zMax The maximum Z coordinate
	void setExtents(
		double xMin, double xMax,
		double yMin, double yMax,
		double zMin, double zMax
		);

///\fn Vector getOrigin() const
///\brief Returns the origin of the bounding box
///\return A Vector object
	Vector getOrigin() const;
///\fn void setOrigin(Vector vector, const Extents& boundaryExtents)
///\brief Translates the origin by some vector
///\param vector A Vector describing the translation
///\param boundaryExtents A larger bounding box that this bounding box should remain inside of. (think 'rover widget')
	void setOrigin(Vector vector, const Extents& boundaryExtents);
///\fn void move(const Vector& vector)
///\brief Translates the origin by some vector
///\param vector The Vector describing the translation
	void move(const Vector& vector);

///\fn bool withinCube(const Vector& vector) const
///\brief Tells whether a point is inside the bounding box
///\param vector A 3D point
///\return true if the point is inside the bounding box
	bool withinCube(const Vector& vector) const;
///\fn void clampTo(const Extents& boundaryExtents)
///\brief Clips the bounding box to fit inside of another bounding box
///\param boundaryExtents The clipping cube
	void clampTo(const Extents& boundaryExtents);

///\fn double getXMin() const
///\brief Returns the minimum X coordinate
	double getXMin() const;
///\fn double getYMin() const
///\brief Returns the minimum Y coordinate
	double getYMin() const;
///\fn double getZMin() const
///\brief Returns the minimum Z coordinate
	double getZMin() const;
///\fn double getXMax() const
///\brief Returns the maximum X coordinate
	double getXMax() const;
///\fn double getYMax() const
///\brief Returns the maximum Y coordinate
	double getYMax() const;
///\fn double getZMax() const
///\brief Returns the maximum Z coordinate
	double getZMax() const;

///\fn void setXMin(double xMin)
///\brief Sets the minimum X coordinate
///\param xMin The new minimum X coordinate
	void setXMin(double xMin);
///\fn void setYMin(double yMin)
///\brief Sets the minimum Y coordinate
///\param yMin The new minimum Y coordinate
	void setYMin(double yMin);
///\fn void setZMin(double zMin)
///\brief Sets the minimum Z coordinate
///\param zMin The new minimum Z coordinate
	void setZMin(double zMin);
///\fn void setXMax(double xMax)
///\brief Sets the maximum X coordinate
///\param xMax The new maximum X coordinate
	void setXMax(double xMax);
///\fn void setYMax(double yMax)
///\brief Sets the maximum Y coordinate
///\param yMax The new maximum Y coordinate
	void setYMax(double yMax);
///\fn void setZMax(double zMax)
///\brief Sets the maximum Z coordinate
///\param zMax The new maximum Z coordinate
	void setZMax(double zMax);

protected:
	double m_XMin, m_XMax;
	double m_YMin, m_YMax;
	double m_ZMin, m_ZMax;


};

#endif // !defined(AFX_EXTENTS_H__541859B1_96FC_453D_8524_1D6789C4F864__INCLUDED_)
