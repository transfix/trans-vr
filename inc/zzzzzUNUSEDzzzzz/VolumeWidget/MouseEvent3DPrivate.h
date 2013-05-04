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

// MouseEvent3DPrivate.h: interface for the MouseEvent3DPrivate class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MOUSEEVENT3DPRIVATE_H__F06D12C2_6C8B_4E5A_B389_FD5AC1E560DD__INCLUDED_)
#define AFX_MOUSEEVENT3DPRIVATE_H__F06D12C2_6C8B_4E5A_B389_FD5AC1E560DD__INCLUDED_

#include <qnamespace.h>
#include <qpoint.h>
#include <qevent.h>
#include <VolumeWidget/Ray.h>
class SimpleOpenGLWidget;
class QMouseEvent;

#include <Qt>
#include <QMouseEvent>

class MouseEvent3DPrivate //: public Qt
{
public:
	MouseEvent3DPrivate(SimpleOpenGLWidget* simpleOpenGLWidget, const QMouseEvent& qmouseevent);
	virtual ~MouseEvent3DPrivate();

	const QPoint& pos()const; 
	const QPoint& globalPos() const;
	int x()const; 
	int y()const; 
	int globalX() const;
	int globalY() const;

	Ray ray() const;

	Qt::ButtonState button() const;
	Qt::ButtonState state() const;
	Qt::ButtonState stateAfter() const;
	bool isAccepted() const;
	void accept();
	void ignore();
	
protected:
	QPoint m_Pos;
	QPoint m_GlobalPos;
	Qt::ButtonState m_Button;
	Qt::ButtonState m_State;
	Qt::ButtonState m_StateAfter;
	bool m_Accept;
	Ray m_Ray;
};

#endif // !defined(AFX_MOUSEEVENT3DPRIVATE_H__F06D12C2_6C8B_4E5A_B389_FD5AC1E560DD__INCLUDED_)
