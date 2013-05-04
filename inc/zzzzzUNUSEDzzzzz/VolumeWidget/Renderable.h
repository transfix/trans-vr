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

// Renderable.h: interface for the Renderable class.
//
//////////////////////////////////////////////////////////////////////

/* $Id: Renderable.h 1498 2010-03-10 22:50:29Z transfix $ */

#if !defined(AFX_RENDERABLE_H__080A644A_E9B0_49F9_A5DF_3FB04658D88B__INCLUDED_)
#define AFX_RENDERABLE_H__080A644A_E9B0_49F9_A5DF_3FB04658D88B__INCLUDED_

class View;

///\class Renderable Renderable.h
///\author Anthony Thane
///\brief This is an abstract base class for objects which can be rendered
///	using OpenGL.
class Renderable  
{
public:
	Renderable();
	virtual ~Renderable();

///\fn virtual bool initForContext()
///\brief This function initializes the object to render using the current
///	OpenGL context.
///\return A bool indicating success or failure.
	virtual bool initForContext() {return true;};
///\fn virtual bool deinitForContext()
///\brief This function cleans up any objects or memory that might have been
///	allocated in initForContext().
///\return A bool indicating success or failure.
	virtual bool deinitForContext() {return true;};
///\fn virtual bool render() = 0
///\brief This is the guts of the class. Derived classes should render
///	something using OpenGL in this function.
///\return A bool indicating success or failure.
	virtual bool render() = 0;

///\fn virtual void setWireframeMode(bool state)
///\brief This function enables wireframe rendering for derived classes where
///	it makes sense.
///\param state If true, enables wireframe rendering. If false, disables
///	wireframe rendering.
	virtual void setWireframeMode(bool) {};

	virtual void setSurfWithWire(bool) {};
};

#endif // !defined(AFX_RENDERABLE_H__080A644A_E9B0_49F9_A5DF_3FB04658D88B__INCLUDED_)
