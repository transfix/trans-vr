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

// Aggregate3DHandler.h: interface for the Aggregate3DHandler class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_AGGREGATE3DHANDLER_H__CB998AC8_2925_4C9D_87EE_537EE0140682__INCLUDED_)
#define AFX_AGGREGATE3DHANDLER_H__CB998AC8_2925_4C9D_87EE_537EE0140682__INCLUDED_

#include <VolumeWidget/Mouse3DHandler.h>
#include <VolumeWidget/ExpandableArray.h>

///\class Aggregate3DHandler Aggregate3DHandler.h
///\author Anthony Thane
///\brief Aggregate3DHandler is a Mouse3DHandler that brings together a
/// collection of Mouse3DHandler objects.
class Aggregate3DHandler : public Mouse3DHandler  
{
public:
	Aggregate3DHandler();
	Aggregate3DHandler(const Aggregate3DHandler& copy);
	virtual ~Aggregate3DHandler();

	virtual Mouse3DHandler* clone() const;

///\fn int add(Mouse3DHandler* handler)
///\brief Adds a Mouse3DHandler to the collection
///\param handler The Mouse3DHandler being added
///\return The new number of handlers in the collection
	int add(Mouse3DHandler* handler);
///\fn Mouse3DHandler* remove( unsigned int index )
///\brief Removes a Mouse3DHandler from the collection
///\param index The index of the handler to be removed
///\return A pointer to the handler that was removed
	Mouse3DHandler* remove( unsigned int index );

	virtual bool mousePress3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);
	virtual bool mouseRelease3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);
	virtual bool mouseDoubleClick3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);
	virtual bool mouseMove3DEvent(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);

	virtual float getNearestClickDistance(SimpleOpenGLWidget*, MouseEvent3DPrivate* e);

protected:
	ExpandableArray<Mouse3DHandler> m_Mouse3DHandlers;
	Mouse3DHandler* m_CurrentCapture;
};

#endif // !defined(AFX_AGGREGATE3DHANDLER_H__CB998AC8_2925_4C9D_87EE_537EE0140682__INCLUDED_)
