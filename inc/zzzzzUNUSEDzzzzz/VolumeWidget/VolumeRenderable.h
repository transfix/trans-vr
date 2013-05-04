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

// VolumeRenderable.h: interface for the VolumeRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMERENDERABLE_H__6BC775C2_3AB9_4B9F_9F49_C5F727FB8B62__INCLUDED_)
#define AFX_VOLUMERENDERABLE_H__6BC775C2_3AB9_4B9F_9F49_C5F727FB8B62__INCLUDED_

#include <VolumeWidget/Renderable.h>
#include <VolumeLibrary/VolumeRenderer.h>

///\class VolumeRenderable VolumeRenderable.h
///\deprecated This class has been replaced by RoverRenderable.
///\author Anthony Thane
class VolumeRenderable : public Renderable  
{
public:
	VolumeRenderable();
	virtual ~VolumeRenderable();

	virtual bool initForContext();
	virtual bool deinitForContext();
	virtual bool render();

	void setOpaqueRenderable(Renderable* renderable);
	void unsetOpaqueRenderable(Renderable* renderable);
	void setSuperOpaqueRenderable(Renderable* renderable);
	void unsetSuperOpaqueRenderable(Renderable* renderable);

	void enableVolumeRendering();
	void disableVolumeRendering();

	VolumeRenderer& getVolumeRenderer();
	const VolumeRenderer& getVolumeRenderer() const;

	virtual void setAspectRatio(double x, double y, double z);

protected:
	Renderable* m_OpaqueRenderable;
	Renderable* m_SuperOpaqueRenderable;
	VolumeRenderer m_VolumeRenderer;

	bool m_VolumeRenderingEnabled;
};

#endif // !defined(AFX_VOLUMERENDERABLE_H__6BC775C2_3AB9_4B9F_9F49_C5F727FB8B62__INCLUDED_)
