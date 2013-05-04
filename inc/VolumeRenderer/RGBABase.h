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

// OpenGLVolumeRGBABase.h: interface for the OpenGLVolumeRGBABase class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_OPENGLVOLUMERGBABASE_H__2615681E_5453_4C3E_A370_4FF06C85BD9F__INCLUDED_)
#define AFX_OPENGLVOLUMERGBABASE_H__2615681E_5453_4C3E_A370_4FF06C85BD9F__INCLUDED_

#include <VolumeRenderer/RendererBase.h>

namespace OpenGLVolumeRendering {

	/// The base class for non-colormapped volume renderers
	class RGBABase : public RendererBase  
	{
	public:
		RGBABase();
		virtual ~RGBABase();

		/// Initializes the renderer.  Should be called again if the renderer is
		/// moved to a different openGL context.  If this returns false, do not try
		/// to use it to do volumeRendering
		virtual bool initRenderer();

		/// Uploads colormapped data
		virtual bool uploadRGBAData(const GLubyte* data, int width, int height, int depth) = 0;

		/// Tests to see if the given parameters would return an error
		virtual bool testRGBAData(int width, int height, int depth) = 0;

	};

};

#endif // !defined(AFX_OPENGLVOLUMERGBABASE_H__2615681E_5453_4C3E_A370_4FF06C85BD9F__INCLUDED_)
