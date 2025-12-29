
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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// OpenGLVolumeCGRGBAImpl.h: interface for the OpenGLVolumeCGRGBAImpl class.
//
//////////////////////////////////////////////////////////////////////
#if !defined(                                                                \
    AFX_OPENGLVOLUMECGRGBAIMPL_H__D437055A_0A8E_5755_A9AD_3475D43DB261__INCLUDED_)
#define AFX_OPENGLVOLUMECGRGBAIMPL_H__D437055A_0A8E_5755_A9AD_3475D43DB261__INCLUDED_

#include <VolumeRenderer/CG_Programs.h>
#include <VolumeRenderer/RGBABase.h>

namespace OpenGLVolumeRendering {

class CGRGBAImpl : public RGBABase, public CG_Programs {
public:
  CGRGBAImpl();
  virtual ~CGRGBAImpl();

  // Initializes the renderer.  Should be called again if the renderer is
  // moved to a different openGL context.  If this returns false, do not try
  // to use it to do volumeRendering
  virtual bool initRenderer();

  // Makes the check necessary to determine if this renderer is
  // compatible with the hardware its running on
  virtual bool checkCompatibility() const;

  // Uploads colormapped data
  virtual bool uploadRGBAData(const GLubyte *data, int width, int height,
                              int depth);

  // Tests to see if the given parameters would return an error
  virtual bool testRGBAData(int width, int height, int depth);

  // Performs the actual rendering.
  virtual bool renderVolume();

  // For Shading Uploads Normal data
  virtual bool uploadGradients(const GLubyte *data, int width, int height,
                               int depth);
  virtual void setLight(float *lightf);
  virtual void setView(float *viewf);

  bool isShadedRenderingAvailable() { return true; }
  bool enableShadedRendering() {
    m_ShadeFlag = true;
    return true;
  }
  bool disableShadedRendering() {
    m_ShadeFlag = false;
    return true;
  }

protected:
  // Remembers the uploaded width height and depth
  int m_Width, m_Height, m_Depth;

  // The opengl texture ID
  GLuint m_DataTextureName;
  bool m_GL_VERSION_1_2;

  // Holds the pointers to the extension functions
  // MyExtensions m_Extensions;

  // Flag indicating if we were successfully initialized
  bool m_Initialized;

  // Initializes the necessary extensions.
  virtual bool initExtensions();

  // Gets the opengl texture IDs
  bool initTextureNames();

  // Render the actual triangles
  void renderTriangles();

  // For Shading
  void shadeRenderTriangles();

  // For Unshading - can be substituted by CG,
  // but keep both version for convenience.
  // if Video card supports CG, it always support Register Combiner
  void NV_Unshade_Combiner_Setup(void);
};
}; // namespace OpenGLVolumeRendering

#endif // !defined(AFX_OPENGLVOLUMECGRGBAIMPL_H__D437055A_0A8E_5755_A9AD_3475D43DB261__INCLUDED_)
