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

// Mouse3DAdapter.h: interface for the Mouse3DAdapter class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MOUSE3DADAPTER_H__7CAFF265_5FF6_4CA2_A323_3CF1AEED9AD9__INCLUDED_)
#define AFX_MOUSE3DADAPTER_H__7CAFF265_5FF6_4CA2_A323_3CF1AEED9AD9__INCLUDED_

#include <VolumeWidget/MouseHandler.h>
class Mouse3DHandler;

class Mouse3DAdapter : public MouseHandler  
{
public:
	Mouse3DAdapter(Mouse3DHandler* mouse3DHandler);
	virtual ~Mouse3DAdapter();

	virtual MouseHandler* clone() const;

	virtual bool mousePressEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool mouseReleaseEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool mouseDoubleClickEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool mouseMoveEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool wheelEvent(SimpleOpenGLWidget*, QWheelEvent* e);

protected:
	Mouse3DHandler* const m_Mouse3DHandler;
};

#endif // !defined(AFX_MOUSE3DADAPTER_H__7CAFF265_5FF6_4CA2_A323_3CF1AEED9AD9__INCLUDED_)
