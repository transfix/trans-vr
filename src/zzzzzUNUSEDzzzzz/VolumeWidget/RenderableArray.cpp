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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// RenderableArray.cpp: implementation of the RenderableArray class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/RenderableArray.h>
//#include <GL/gl.h>
#include <glew/glew.h>

const unsigned int InitialArraySize = 16;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RenderableArray::RenderableArray()
{
	initArrays();
	
	m_WireframeMode = false;
}

RenderableArray::~RenderableArray()
{
	deleteArrays();
}

bool RenderableArray::initForContext()
{
	unsigned int c;
	bool final = true, current;
	for(c=0;c<m_NumberOfObjects;c++)
	{
		current = m_Renderables[c]->initForContext();
		final = final && current;
	}
	return final;
}

bool RenderableArray::deinitForContext()
{
	unsigned int c;
	bool final = true, current;
	for(c=0;c<m_NumberOfObjects;c++)
	{
		current = m_Renderables[c]->deinitForContext();
		final = final && current;
	}
	return final;
}

bool RenderableArray::render()
{
	unsigned int c;
	bool final = true, current;
	for(c=0;c<m_NumberOfObjects;c++)
	{
		current = m_Renderables[c]->render();
		final = final && current;
	}
	return final;
}

void RenderableArray::setWireframeMode(bool state)
{
	unsigned int c;
	
	for(c=0;c<m_NumberOfObjects;c++)
		m_Renderables[c]->setWireframeMode(state);

	m_WireframeMode = state;
}

void RenderableArray::setSurfWithWire(bool state)
{
	unsigned int c;
	
	for(c=0;c<m_NumberOfObjects;c++)
		m_Renderables[c]->setSurfWithWire(state);

	m_SurfWithWire = state;
}

int RenderableArray::add( Renderable* renderable )
{
	doubleObjectArray();
	// make sure the current wireframe mode is inherited
	renderable->setWireframeMode(m_WireframeMode);
	m_Renderables[m_NumberOfObjects] = renderable;
	int indexPosition;
	if (m_HoleList.isEmpty()) {
		doubleIndexArray();
		indexPosition = m_NextIndexEntry;
		m_NextIndexEntry++;
	}
	else {
		indexPosition = m_HoleList.deQueue();
	}
	m_ObjToIndex[m_NumberOfObjects] = indexPosition;
	m_IndexToObj[indexPosition] = m_NumberOfObjects;
	m_NumberOfObjects++;
	return m_NumberOfObjects-1;
}

bool RenderableArray::set( Renderable* renderable, unsigned int index )
{
	index = m_IndexToObj[index];
	if( index < m_NumberOfObjects ) 
	{
		m_Renderables[index] = renderable;
		return true;
	}
	return false;	
}

Renderable* RenderableArray::remove( unsigned int index )
{
	int object = m_IndexToObj[index];
	if( object < (int)m_NumberOfObjects && object >=0 ) 
	{
		m_HoleList.enQueue(index);
		m_NumberOfObjects--;
		Renderable* temp = m_Renderables[object];
		m_Renderables[object] = m_Renderables[m_NumberOfObjects];
		m_ObjToIndex[object] = m_ObjToIndex[m_NumberOfObjects];
		m_IndexToObj[m_ObjToIndex[object]] = object;
		return temp;
	}
	else
	{
		return 0;
	}
}

Renderable* RenderableArray::get( unsigned int index )
{
	int object = m_IndexToObj[index];
	if( object < (int)m_NumberOfObjects && object >=0 ) 
	{
		return m_Renderables[object];
	}
	else
	{
		return 0;
	}
}

Renderable* RenderableArray::getIth(unsigned int i) const
{
	if( i < m_NumberOfObjects ) 
	{
		return m_Renderables[i];
	}
	else
	{
		return 0;
	}
}

unsigned int RenderableArray::getNumberOfRenderables()
{
	return m_NumberOfObjects;
}

void RenderableArray::clear(bool isDelete)
{
	unsigned int c;
	for(c=0;c<m_NumberOfObjects;c++)
	{
		if(isDelete)
		{
			delete m_Renderables[c];
			m_Renderables[c] = 0;
		}
	}
	m_NumberOfObjects=0;
	m_NextIndexEntry = 0;
	m_HoleList.clearQueue();
}

void RenderableArray::initArrays()
{
	initObjectArray();
	initIndexArray();
}

void RenderableArray::initObjectArray()
{
	m_NumberOfObjects = 0;
	m_SizeOfObjectsArray = InitialArraySize;

	m_Renderables = new Renderable*[m_SizeOfObjectsArray];
	m_ObjToIndex = new int[m_SizeOfObjectsArray];

	unsigned int c;
	for(c=0;c<m_SizeOfObjectsArray;c++) {
		m_Renderables[c] = 0;
	}
}

void RenderableArray::initIndexArray()
{
	m_NextIndexEntry = 0;
	m_SizeOfIndexToObjectsArray = InitialArraySize;
	m_IndexToObj = new int[m_SizeOfObjectsArray];
}

void RenderableArray::deleteArrays()
{
  //This class appears to be responsible for each Renderable object pointed to by m_Renderables
  //juding from usage of RenderableArray::add() in newvolumemainwindow.cpp.  So, lets delete
  //the Renderables before deleting the pointer array and hope nothing explodes! -transfix (06/24/2008)
  for(unsigned int i = 0; i < m_NumberOfObjects; i++)
    delete m_Renderables[i];
  delete [] m_Renderables;
  delete [] m_IndexToObj;
  delete [] m_ObjToIndex;
}

void RenderableArray::doubleObjectArray()
{
	unsigned int c;
	if (m_NumberOfObjects >= m_SizeOfObjectsArray) {
		Renderable** oldRenderables = m_Renderables;
		int* oldObjToIndex = m_ObjToIndex;
		m_Renderables = new Renderable*[m_SizeOfObjectsArray*2];
		m_ObjToIndex = new int[m_SizeOfObjectsArray*2];

		for(c=0;c<m_SizeOfObjectsArray;c++) {
			m_Renderables[c] = oldRenderables[c];
			m_ObjToIndex[c] = oldObjToIndex[c];
		}
		m_SizeOfObjectsArray *= 2;

		delete [] oldRenderables;
		delete [] oldObjToIndex;
	}
}

void RenderableArray::doubleIndexArray()
{
	unsigned int c;
	if (m_NextIndexEntry >= m_SizeOfIndexToObjectsArray) {
		int* oldIndexToObjectsArray = m_IndexToObj;
	
		m_IndexToObj = new int[m_SizeOfIndexToObjectsArray*2];

		for(c=0;c<m_NextIndexEntry;c++) {
			m_IndexToObj[c] = oldIndexToObjectsArray[c];
		}
		m_SizeOfIndexToObjectsArray *= 2;

		delete [] oldIndexToObjectsArray;
	}
}


