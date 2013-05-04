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

// IPolyRenderable.h: interface for the IPolyRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_IPOLYRENDERABLE_H__B392B786_8F45_4222_9A02_0F1BE89237C8__INCLUDED_)
#define AFX_IPOLYRENDERABLE_H__B392B786_8F45_4222_9A02_0F1BE89237C8__INCLUDED_

#include <VolumeWidget/Renderable.h>
#include "../ipoly/src/ipoly.h"
#include <VolumeWidget/GeometryRenderable.h>

///\class IPolyRenderable IPolyRenderable.h
///\deprecated This class is no longer used in Volume Rover.
///\brief This class renders IPoly meshes using a GeometryRenderable.
///\author Anthony Thane
class IPolyRenderable : public Renderable  
{
public:
	IPolyRenderable();
	IPolyRenderable(const Geometry& geometry);
	IPolyRenderable(iPoly* ipoly);
	virtual ~IPolyRenderable();

	virtual bool render();
///\fn bool loadFile(const char* name)
///\brief Loads a file
///\param name A path to a file
///\return A boolean indicating success or failure
	bool loadFile(const char* name);
///\fn bool saveFile(const char* name)
///\brief Saves a file
///\param name A path to a file
///\return A boolean indicating success or failure
	bool saveFile(const char* name);

protected:
	void copyiPolyToGeometry();

	iPoly_P m_iPoly;
	GeometryRenderable m_GeometryRenderable;
};

#endif // !defined(AFX_IPOLYRENDERABLE_H__B392B786_8F45_4222_9A02_0F1BE89237C8__INCLUDED_)
