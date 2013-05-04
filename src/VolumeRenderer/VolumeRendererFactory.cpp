/*
  Copyright 2002-2003 The University of Texas at Austin

	Authors: Anthony Thane <thanea@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeLibrary.

  VolumeLibrary is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeLibrary is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

// VolumeRendererFactory.cpp: implementation of the VolumeRendererFactory class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRenderer/VolumeRendererFactory.h>
#include <VolumeRenderer/RGBABase.h>
#include <VolumeRenderer/UnshadedBase.h>
#include <VolumeRenderer/SimpleRGBAImpl.h>
#include <VolumeRenderer/PalettedImpl.h>
#include <VolumeRenderer/FragmentProgramImpl.h>
#include <VolumeRenderer/SGIColorTableImpl.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

OpenGLVolumeRendering::VolumeRendererFactory::VolumeRendererFactory()
{

}

OpenGLVolumeRendering::VolumeRendererFactory::~VolumeRendererFactory()
{

}

OpenGLVolumeRendering::RGBABase* OpenGLVolumeRendering::VolumeRendererFactory::getRGBARenderer()
{
	RGBABase* renderer;
	// this should work on most platforms
	renderer = new SimpleRGBAImpl();
	if (renderer->initRenderer()) {
		return renderer;
	}

	// failed
	delete renderer;
	renderer = 0;
	return 0;
}

OpenGLVolumeRendering::UnshadedBase* OpenGLVolumeRendering::VolumeRendererFactory::getUnshadedRenderer()
{
	UnshadedBase* renderer;
	// first we try the paletted version which we know works on 
	// Nvidia
	renderer = new PalettedImpl();
	if (renderer->initRenderer()) {
		return renderer;
	}
	//failed
	delete renderer;
	renderer = 0;

	// looks like NVIDIA might not support the paletted texture
	// extension any more, this is the alternative
	renderer = new FragmentProgramImpl();
	if (renderer->initRenderer()) {
		return renderer;
	}
	// failed
	delete renderer;
	renderer = 0;

	// next we try the sgi version
	renderer = new SGIColorTableImpl();
	if (renderer->initRenderer()) {
		return renderer;
	}

	// out of options
	delete renderer;
	renderer = 0;
	return 0;
}

