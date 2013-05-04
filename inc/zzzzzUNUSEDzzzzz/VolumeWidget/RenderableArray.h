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

// RenderableArray.h: interface for the RenderableArray class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RENDERABLEARRAY_H__01D4CFBD_5B33_4E25_A0EB_BA07EA21CEFE__INCLUDED_)
#define AFX_RENDERABLEARRAY_H__01D4CFBD_5B33_4E25_A0EB_BA07EA21CEFE__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeWidget/Renderable.h>
#include <VolumeWidget/IntQueue.h>

///\class RenderableArray RenderableArray.h
///\brief The RenderableArray class is a Renderable which serves as a
///	container for multiple Renderable objects.
///author Anthony Thane
class RenderableArray : public Renderable  
{
public:
	RenderableArray();
	virtual ~RenderableArray();

	virtual bool initForContext();
	virtual bool deinitForContext();
	virtual bool render();

	virtual void setWireframeMode(bool state);

	virtual void setSurfWithWire(bool state);

///\fn int add( Renderable* renderable )
///\brief This function adds a Renderable to the array
///\param renderable A pointer to a Renderable object.
///\return An index in the array where the object is stored
	int add( Renderable* renderable );
///\fn bool set( Renderable* renderable, unsigned int index )
///\brief This function assigns a Renderable object to a specific index in the
///	array.
///\param renderable A pointer to a Renderable object.
///\param index An index in the array
///\return A boolean indicating success or failure. The function will fail if
///	the index is out of range.
	bool set( Renderable* renderable, unsigned int index );
///\fn Renderable* remove( unsigned int index )
///\brief This function removes a Renderable object from the array.
///\param index The index of the object to be removed
///\return A pointer to the removed object or NULL if the index is out of
///	range.
	Renderable* remove( unsigned int index );
///\fn Renderable* get( unsigned int index )
///\brief This function returns a pointer to a specified member of the array.
///\param index An index into the array (indices may be non-contiguous)
///\return A pointer to a Renderable object or NULL if the index is out of
///	range.
	Renderable* get( unsigned int index );
///\fn Renderable* getIth(unsigned int i) const
///\brief This function returns a pointer to the ith member of the array.
///\param i An index into the array (contiguous from 0 to getNumberOfRenderables())
///\return A pointer to a Renderable object or NULL if the index is out of
///	range.
	Renderable* getIth(unsigned int i) const;
///\fn unsigned int getNumberOfRenderables()
///\brief This function returns the number of objects that the array contains.
///\return The number of objects
	unsigned int getNumberOfRenderables();

///\fn void clear()
///\brief This function removes all Renderables from the array.
	void clear(bool isDelete=true);
protected:
	void initArrays();
	void initObjectArray();
	void initIndexArray();
	void deleteArrays();
	void doubleObjectArray();
	void doubleIndexArray();

	unsigned int m_NumberOfObjects;
	unsigned int m_SizeOfObjectsArray;
	Renderable** m_Renderables;
	int* m_ObjToIndex;

	unsigned int m_NextIndexEntry;
	unsigned int m_SizeOfIndexToObjectsArray;
	int* m_IndexToObj;

	Queue m_HoleList;

	bool m_WireframeMode;
	bool m_SurfWithWire;
};

#endif // !defined(AFX_RENDERABLEARRAY_H__01D4CFBD_5B33_4E25_A0EB_BA07EA21CEFE__INCLUDED_)
