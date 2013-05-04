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

// Renderer.h: interface for the Renderer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RENDERER_H__458FC25A_6D79_4B38_AF34_B758C4C43596__INCLUDED_)
#define AFX_RENDERER_H__458FC25A_6D79_4B38_AF34_B758C4C43596__INCLUDED_

#include <glew/glew.h>

#include <VolumeRenderer/Plane.h>
#include <VolumeRenderer/UnshadedBase.h>
#include <VolumeRenderer/RGBABase.h>

namespace OpenGLVolumeRendering {

	/// Automatically initializes a colormapped and non-colormapped volume renderer.
	class Renderer  
	{
	public:
		Renderer();
		virtual ~Renderer();

		bool initRenderer();
		
		bool setAspectRatio(double ratioX, double ratioY, double ratioZ);
		bool setTextureSubCube(double minX, double minY, double minZ, double maxX, double maxY, double maxZ);

		bool setQuality(double quality);
		double getQuality() const;

		bool setMaxPlanes(int maxPlanes);
		int getMaxPlanes() const;

		bool setNearPlane(double nearPlane);
		double getNearPlane();

		bool isShadedRenderingAvailable() const;
		bool enableShadedRendering();
		bool disableShadedRendering();

		bool uploadColorMappedData(const GLubyte* data, int width, int height, int depth);
		bool uploadColorMappedDataWithBorder(const GLubyte* data, int width, int height, int depth);
		bool testColorMappedData(int width, int height, int depth);
		bool testColorMappedDataWithBorder(int width, int height, int depth);
		bool uploadRGBAData(const GLubyte* data, int width, int height, int depth);
		bool uploadGradients(const GLubyte* data, int width, int height, int depth);
		bool calculateGradientsFromDensities(const GLubyte* data, int width, int height, int depth);

		bool uploadColorMap(const GLubyte* colorMap);
                bool uploadColorMap(const GLfloat* colorMap);

		// For Shading
		void setLight(float *lightf);
		void setView(float *viewf);

		int getNumberOfPlanesRendered() const;

		bool renderVolume();


	protected:
		bool initColormappedRenderer();
		bool initRGBARenderer();

		bool m_UseColormapped;
                bool m_UseRGBA;
		bool m_DataLoaded;

		bool initFlags();

		UnshadedBase* m_Colormapped;
		RGBABase* m_RGBA;

	};

};

#endif // !defined(AFX_RENDERER_H__458FC25A_6D79_4B38_AF34_B758C4C43596__INCLUDED_)
