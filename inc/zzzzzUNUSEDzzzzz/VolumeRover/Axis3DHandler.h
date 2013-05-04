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

// Axis3DHandler.h: interface for the Axis3DHandler class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_AXIS3DHANDLER_H__5BC573E6_A739_4530_9799_87571210E0E7__INCLUDED_)
#define AFX_AXIS3DHANDLER_H__5BC573E6_A739_4530_9799_87571210E0E7__INCLUDED_

#include <VolumeWidget/Mouse3DHandler.h>
#include <VolumeWidget/Vector.h>
class Rover3DWidget;

class Axis3DHandler : public Mouse3DHandler  
{
public:

	enum Axis {XAxis, YAxis, ZAxis};
	enum Section {NScaleNob, Middle, PScaleNob};

	Axis3DHandler(Rover3DWidget* rover3DWidget, Axis axis);
	virtual ~Axis3DHandler();

	virtual Mouse3DHandler* clone() const;

	virtual bool mousePress3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);
	virtual bool mouseRelease3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);
	virtual bool mouseDoubleClick3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);
	virtual bool mouseMove3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);

	virtual float getNearestClickDistance(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);

	Axis getAxis() const;


protected:
	bool closeToMouse(SimpleOpenGLWidget* simpleOpenGLWidget, MouseEvent3DPrivate* e, const Vector& vector);
	float nearestTOnAxis(MouseEvent3DPrivate* e) const;
	Vector nearestPointOnAxis(MouseEvent3DPrivate* e) const;
	void setPScaleNob(const Vector& value) const;
	void setNScaleNob(const Vector& value) const;
	Vector getPNob();
	Vector getNNob();
	Rover3DWidget* const m_Rover3DWidget;
	const Axis m_Axis;

	bool m_MouseDown;
	Section m_Section;
	Vector m_PositionOnAxis;


};

#endif // !defined(AFX_AXIS3DHANDLER_H__5BC573E6_A739_4530_9799_87571210E0E7__INCLUDED_)
