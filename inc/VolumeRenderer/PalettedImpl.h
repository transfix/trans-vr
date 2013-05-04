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

// PalettedImpl.h: interface for the PalettedImpl class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_PALETTEDIMPL_H__0A257040_7C0E_44A2_BF71_58D906390F88__INCLUDED_)
#define AFX_PALETTEDIMPL_H__0A257040_7C0E_44A2_BF71_58D906390F88__INCLUDED_

#include <glew/glew.h>

#include <VolumeRenderer/UnshadedBase.h>

namespace OpenGLVolumeRendering {

	/// A volume renderer which uses the paletted textures extension to do color
	/// mapping. Requires support for 3D textures
	class PalettedImpl : public UnshadedBase
	{
	public:
		PalettedImpl();
		virtual ~PalettedImpl();

		// Initializes the renderer.  Should be called again if the renderer is
		// moved to a different openGL context.  If this returns false, do not try
		// to use it to do volumeRendering
		virtual bool initRenderer();

		// Makes the check necessary to determine if this renderer is 
		// compatible with the hardware its running on
		virtual bool checkCompatibility() const;

		// Uploads colormapped data
		virtual bool uploadColormappedData(const GLubyte* data, int width, int height, int depth);

		// Tests to see if the given parameters would return an error
		virtual bool testColormappedData(int width, int height, int depth);

		// Uploads the transfer function for the colormapped data
		virtual bool uploadColorMap(const GLubyte* colorMap);

		// Performs the actual rendering.
		virtual bool renderVolume();

		virtual bool isShadedRenderingAvailable(){ return false; }

	protected:
		// Remembers the uploaded width height and depth
		int m_Width, m_Height, m_Depth;

		// The opengl texture ID
		GLuint m_DataTextureName;

		// Flag indicating if we were successfully initialized
		bool m_Initialized;

		// Initializes the necessary extensions.
		virtual bool initExtensions();

		// Gets the opengl texture IDs
		bool initTextureNames();

		// Render the actual triangles
		void renderTriangles();
	};

};

#endif // !defined(AFX_PALETTEDIMPL_H__0A257040_7C0E_44A2_BF71_58D906390F88__INCLUDED_)
