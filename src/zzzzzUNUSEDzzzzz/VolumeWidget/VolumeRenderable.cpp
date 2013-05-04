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

// VolumeRenderable.cpp: implementation of the VolumeRenderable class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/VolumeRenderable.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeRenderable::VolumeRenderable()
{
	m_OpaqueRenderable = 0;
	m_SuperOpaqueRenderable = 0;
	m_VolumeRenderingEnabled = false;
}

VolumeRenderable::~VolumeRenderable()
{

}

bool VolumeRenderable::initForContext()
{
	bool final = true, current;
	if (m_OpaqueRenderable) {
		current = m_OpaqueRenderable->initForContext();
		final = final && current;
	}
	if (m_SuperOpaqueRenderable) {
		current = m_SuperOpaqueRenderable->initForContext();
		final = final && current;
	}
	current = m_VolumeRenderer.initRenderer();
	final = final && current;

	return final;
}

bool VolumeRenderable::deinitForContext()
{
	bool final = true, current;
	if (m_OpaqueRenderable) {
		current = m_OpaqueRenderable->deinitForContext();
		final = final && current;
	}
	if (m_SuperOpaqueRenderable) {
		current = m_SuperOpaqueRenderable->deinitForContext();
		final = final && current;
	}

	return final;
}

bool VolumeRenderable::render()
{
	// render opaque stuff first
	bool final = true, current;
	glPushAttrib(GL_DEPTH_BUFFER_BIT);
	glEnable( GL_DEPTH_TEST );
	if (m_OpaqueRenderable) {
		current = m_OpaqueRenderable->render();
		final = final && current;
	}

	if (m_VolumeRenderingEnabled) {
		m_VolumeRenderer.renderVolume();
	}

	if (m_SuperOpaqueRenderable) {
		glDisable(GL_DEPTH_TEST);
		current = m_SuperOpaqueRenderable->render();
		final = final && current;
	}
	glPopAttrib();

	return final;
}

void VolumeRenderable::setOpaqueRenderable(Renderable* renderable)
{
	m_OpaqueRenderable = renderable;
}

void VolumeRenderable::unsetOpaqueRenderable(Renderable* renderable)
{
	m_OpaqueRenderable = 0;
}

void VolumeRenderable::setSuperOpaqueRenderable(Renderable* renderable)
{
	m_SuperOpaqueRenderable = renderable;
}

void VolumeRenderable::unsetSuperOpaqueRenderable(Renderable* renderable)
{
	m_SuperOpaqueRenderable = 0;
}

void VolumeRenderable::enableVolumeRendering()
{
	m_VolumeRenderingEnabled = true;
}

void VolumeRenderable::disableVolumeRendering()
{
	m_VolumeRenderingEnabled = false;
}

VolumeRenderer& VolumeRenderable::getVolumeRenderer()
{
	return m_VolumeRenderer;
}

const VolumeRenderer& VolumeRenderable::getVolumeRenderer() const
{
	return m_VolumeRenderer;
}

void VolumeRenderable::setAspectRatio(double x, double y, double z)
{
	m_VolumeRenderer.setAspectRatio(x,y,z);
}

