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

// MouseEvent3DPrivate.cpp: implementation of the MouseEvent3DPrivate class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/MouseEvent3DPrivate.h>
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <qevent.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MouseEvent3DPrivate::MouseEvent3DPrivate(SimpleOpenGLWidget* simpleOpenGLWidget, const QMouseEvent& qmouseevent)
{
	Ray ray = simpleOpenGLWidget->getView().GetPickRay(qmouseevent.x(), qmouseevent.y());

	m_Pos = qmouseevent.pos();
	m_GlobalPos = qmouseevent.globalPos();
	m_Button = qmouseevent.button();
	m_State = qmouseevent.state();
	m_StateAfter = qmouseevent.stateAfter();
	m_Accept = qmouseevent.isAccepted();
	m_Ray = ray;
}

MouseEvent3DPrivate::~MouseEvent3DPrivate()
{

}

const QPoint& MouseEvent3DPrivate::pos() const
{
	return m_Pos;
}

const QPoint& MouseEvent3DPrivate::globalPos() const
{
	return m_GlobalPos;
}

int MouseEvent3DPrivate::x() const
{
	return m_Pos.x();
}

int MouseEvent3DPrivate::y() const
{
	return m_Pos.y();
}

int MouseEvent3DPrivate::globalX() const
{
	return m_GlobalPos.x();
}

int MouseEvent3DPrivate::globalY() const
{
	return m_GlobalPos.y();
}

Ray MouseEvent3DPrivate::ray() const
{
	return m_Ray;
}

Qt::ButtonState MouseEvent3DPrivate::button() const
{
	return m_Button;
}

Qt::ButtonState MouseEvent3DPrivate::state() const
{
	return m_State;
}

Qt::ButtonState MouseEvent3DPrivate::stateAfter() const
{
	return m_StateAfter;
}

bool MouseEvent3DPrivate::isAccepted() const
{
	return m_Accept;
}

void MouseEvent3DPrivate::accept()
{
	m_Accept = true;
}

void MouseEvent3DPrivate::ignore()
{
	m_Accept = false;
}

