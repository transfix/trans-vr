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

// Rover3DWidget.h: interface for the Rover3DWidget class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ROVER3DWIDGET_H__0728E187_9100_44E6_B363_3FC056575CCB__INCLUDED_)
#define AFX_ROVER3DWIDGET_H__0728E187_9100_44E6_B363_3FC056575CCB__INCLUDED_

#include <qobject.h>
#include <VolumeWidget/Mouse3DHandler.h>
#include <VolumeWidget/Extents.h>
#include <VolumeWidget/GeometryRenderable.h>
#include <cvcraw_geometry/Geometry.h>
#include <VolumeWidget/Aggregate3DHandler.h>
#include <VolumeWidget/Mouse3DAdapter.h>
#include <VolumeRover/Axis3DHandler.h>


class Rover3DWidget : public QObject
{
	Q_OBJECT
public:
	Rover3DWidget();
	virtual ~Rover3DWidget();

	enum Axis {XAxis, YAxis, ZAxis, NoAxis};

	const Extents& getBoundary() const;
	const Extents& getSubVolume() const;

	void setOrigin(Vector vector);
	
	void setXScaleNob(Vector value, bool symetric = true);
	void setYScaleNob(Vector value, bool symetric = true);
	void setZScaleNob(Vector value, bool symetric = true);

	void setPXScaleNob(const Vector& value);
	void setPYScaleNob(const Vector& value);
	void setPZScaleNob(const Vector& value);
	void setNXScaleNob(const Vector& value);
	void setNYScaleNob(const Vector& value);
	void setNZScaleNob(const Vector& value);

	void setAspectRatio(double x, double y, double z);
	
	void setColor(float r, float g, float b);

	virtual void roverDown(Axis3DHandler::Axis axis);
	virtual void roverMoving(Axis3DHandler::Axis axis);
	virtual void roverReleased(Axis3DHandler::Axis axis);

	GeometryRenderable* getWireCubes();
	GeometryRenderable* getAxes();
	Mouse3DHandler* get3DHandler();
	MouseHandler* getHandler();

signals:
	void RoverExploring();
	void RoverReleased();

protected:
	void setCurrentHighlight(Axis3DHandler::Axis axis);
	void prepareGeometry();
	void prepareAxes();
	void prepareWireCubes();
	void prepareHandler();
	
	Extents m_Boundary;
	Extents m_SubVolume;
	Axis m_CurrentHighlight;

	bool m_GeometriesAllocated;
	GeometryRenderable m_WireCubes;
	GeometryRenderable m_Axes;
	Geometry m_WireCubeGeometry;
	Geometry m_AxesGeometry;
	
	float m_R,m_G,m_B;

	Aggregate3DHandler m_Handler;
	Mouse3DAdapter m_Adapter;


};

#endif // !defined(AFX_ROVER3DWIDGET_H__0728E187_9100_44E6_B363_3FC056575CCB__INCLUDED_)
