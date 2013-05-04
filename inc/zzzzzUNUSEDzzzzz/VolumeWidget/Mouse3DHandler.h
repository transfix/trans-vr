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

// Mouse3DHandler.h: interface for the Mouse3DHandler class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MOUSE3DHANDLER_H__903C010F_05A1_4B96_9A61_D8E7D95F8245__INCLUDED_)
#define AFX_MOUSE3DHANDLER_H__903C010F_05A1_4B96_9A61_D8E7D95F8245__INCLUDED_

#include <qobject.h>
class SimpleOpenGLWidget;
class MouseEvent3DPrivate;

class Mouse3DHandler : public QObject
{
public:
	Mouse3DHandler();
	Mouse3DHandler(const Mouse3DHandler& copy);
	virtual ~Mouse3DHandler();

	virtual Mouse3DHandler* clone() const = 0;

	virtual bool mousePress3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e) = 0;
	virtual bool mouseRelease3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e) = 0;
	virtual bool mouseDoubleClick3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e) = 0;
	virtual bool mouseMove3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e) = 0;

	virtual float getNearestClickDistance(SimpleOpenGLWidget*, MouseEvent3DPrivate* e) = 0;
	//virtual bool wheelEvent(SimpleOpenGLWidget*, QWheelEvent* e) = 0;
	
};

#endif // !defined(AFX_MOUSE3DHANDLER_H__903C010F_05A1_4B96_9A61_D8E7D95F8245__INCLUDED_)
