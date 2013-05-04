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

// MouseEvent3D.h: interface for the MouseEvent3D class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MOUSEEVENT3D_H__1C65A386_D980_4EA5_897A_1ACC67FE83D2__INCLUDED_)
#define AFX_MOUSEEVENT3D_H__1C65A386_D980_4EA5_897A_1ACC67FE83D2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <qnamespace.h>
#include <qpoint.h>
#include <qevent.h>
#include <VolumeWidget/Vector.h>
#include <VolumeWidget/Ray.h>
#include <VolumeWidget/View.h>
#include <VolumeWidget/Grid.h>

class OpenGLViewer;
//Added by qt3to4:
//#include <QShowEvent>
//#include <QWheelEvent>
//#include <QHideEvent>

#include <QMouseEvent>
#include <Qt>

class MouseEvent3D //: public Qt
{
public:
	MouseEvent3D(const QMouseEvent& qmouseevent, const View& view, const Grid& grid, OpenGLViewer *ptrOpenGLViewer);
	virtual ~MouseEvent3D();
	
	const QPoint& pos()const; 
	const QPoint& globalPos() const;
	int x()const; 
	int y()const; 
	int globalX() const;
	int globalY() const;

	const Vector& worldPos() const;
	float worldX() const;
	float worldY() const;
	float worldZ() const;
	Ray ray() const;

	Qt::ButtonState button() const;
	Qt::ButtonState state() const;
	Qt::ButtonState stateAfter() const;
	bool isAccepted() const;
	bool isOnGrid() const;
	void accept();
	void ignore();
	OpenGLViewer* getOpenGLViewer();
	
protected:
	QPoint m_Pos;
	QPoint m_GlobalPos;
	Vector m_WorldPos;
	Qt::ButtonState m_Button;
	Qt::ButtonState m_State;
	Qt::ButtonState m_StateAfter;
	bool m_Accept;
	bool m_OnGrid;
	Ray m_Ray;
	OpenGLViewer *m_ptrOpenGLViewer;
};

#endif // !defined(AFX_MOUSEEVENT3D_H__1C65A386_D980_4EA5_897A_1ACC67FE83D2__INCLUDED_)
