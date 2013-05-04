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

// WireCube.h: interface for the WireCube class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_WIRECUBE_H__2F81BE7A_8B57_4421_9B82_AFE92105870D__INCLUDED_)
#define AFX_WIRECUBE_H__2F81BE7A_8B57_4421_9B82_AFE92105870D__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <cvcraw_geometry/Geometry.h>
#include <VolumeWidget/Vector.h>

class WireCube  
{
public:
	WireCube();
	WireCube(
		double xMin, double xMax,
		double yMin, double yMax,
		double zMin, double zMax
		);
	virtual ~WireCube();

	void setExtents(
		double xMin, double xMax,
		double yMin, double yMax,
		double zMin, double zMax
		);

	Vector getOrigin() const;
	void setOrigin(Vector vector, const WireCube& boundary);
	void move(const Vector& vector);

	void showAxes();
	void hideAxes();
	void toggleAxes();
	bool AxesVisible();

	bool withinCube(const Vector& vector);
	void clampTo(const WireCube& boundaryCube);

	void hilightX();
	void hilightY();
	void hilightZ();
	void noHilight();

	Vector getPXScaleNob() const;
	Vector getPYScaleNob() const;
	Vector getPZScaleNob() const;
	Vector getNXScaleNob() const;
	Vector getNYScaleNob() const;
	Vector getNZScaleNob() const;

	void setXScaleNob(Vector value, const WireCube& boundary, bool symetric = true);
	void setYScaleNob(Vector value, const WireCube& boundary, bool symetric = true);
	void setZScaleNob(Vector value, const WireCube& boundary, bool symetric = true);

	void setPXScaleNob(const Vector& value, const WireCube& boundary);
	void setPYScaleNob(const Vector& value, const WireCube& boundary);
	void setPZScaleNob(const Vector& value, const WireCube& boundary);
	void setNXScaleNob(const Vector& value, const WireCube& boundary);
	void setNYScaleNob(const Vector& value, const WireCube& boundary);
	void setNZScaleNob(const Vector& value, const WireCube& boundary);

	Geometry* getGeometry();
	Geometry* getAxes();

	double m_XMin, m_XMax;
	double m_YMin, m_YMax;
	double m_ZMin, m_ZMax;

private:
	enum Axes { XAxis, YAxis, ZAxis, NoAxis };

	void prepareGeometry();
	void prepareAxes();

	Axes m_WhichHilight;

	Geometry m_Geometry;
	Geometry m_Axes;
	bool m_bOptionsChanged; // a boolean signalling when the geometry needs to be regenerated
	bool m_bShowAxes;
};

#endif // !defined(AFX_WIRECUBE_H__2F81BE7A_8B57_4421_9B82_AFE92105870D__INCLUDED_)
