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

// CGRGBAImpl.cpp: implementation of the CGRGBAImpl class.
//
//////////////////////////////////////////////////////////////////////

// #define CGRGBA_DEBUG
// #define TEXTURE_COMPRESSION

#include <VolumeRenderer/CGRGBAImpl.h>
#include <VolumeRenderer/CG_RenderDef.h>
#include <assert.h>
#include <fstream>
#include <math.h>
#include <stdio.h>

using namespace OpenGLVolumeRendering;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CGRGBAImpl::CGRGBAImpl() {
  m_Width = -1;
  m_Height = -1;
  m_Depth = -1;
  m_Initialized = false;

  m_ShadeFlag = false;
}

CGRGBAImpl::~CGRGBAImpl() {}

// Initializes the renderer.  Should be called again if the renderer is
// moved to a different openGL context.  If this returns false, do not try
// to use it to do volumeRendering
bool CGRGBAImpl::initRenderer() {
  m_GL_VERSION_1_2 = glewIsSupported("GL_VERSION_1_2");
  m_Initialized = false;
  m_Width = -1;
  m_Height = -1;
  m_Depth = -1;

  if (!RGBABase::initRenderer()) {
    printf("Error: UnshadedBase::initRenderer()\n");
    return false;
  }
  if (!initExtensions()) {
    printf("Error: initExtensions()\n");
    return false;
  }
  if (!initTextureNames()) {
    printf("Error: initTextureNames()\n");
    return false;
  }
  if (!initCG(DEF_CG_RGBA)) {
    printf("Error: initCG()\n");
    //	cgDestroyProgram(_vertexProgramCg);
    //	cgDestroyContext(_contextCg);
    return false;
  }
#ifdef CG_DEBUG
  printf("Init renderer() is done\n");
#endif
  m_Initialized = true;
  return true;
}

// Initializes the necessary extensions.
bool CGRGBAImpl::initExtensions() {

  if (!glewIsSupported("GL_VERSION_1_2") &&
      !glewIsSupported("GL_SGIS_texture_edge_clamp")) {
// if neither is available, we have to bail
#ifdef CG_DEBUG
    fprintf(stderr, "initExtensions false\n");
    fflush(stderr);
#endif
    return false;
  }

#ifdef CG_DEBUG
  fprintf(stderr, "initExtensions true\n");
  fflush(stderr);
#endif

  // arand: 8-19-2011, added the first argument because I think that case is
  // ok...
  if (!glewIsSupported("GL_VERSION_1_2") &&
      !glewIsSupported("GL_EXT_texture3D")) {
#ifdef CG_DEBUG
    fprintf(stderr, "GL_EXT_texture3D is not supported\n");
    fflush(stderr);
#endif
    return false;
  }

  return true;
}

// Gets the opengl texture IDs
bool CGRGBAImpl::initTextureNames() {
  // clear previous errors
  GLenum error = glGetError();

  // get the names
  glGenTextures(1, &m_DataTextureName);
  glGenTextures(1, &m_RGB_normals_3DT);

  // test for error
  error = glGetError();
  if (error == GL_NO_ERROR)
    return true;
  else
    return false;
}

// Makes the check necessary to determine if this renderer is
// compatible with the hardware its running on
bool CGRGBAImpl::checkCompatibility() const {
  if (!glewIsSupported("GL_VERSION_1_2") &&
      !glewIsSupported("GL_SGIS_texture_edge_clamp")) {
// if neither is available, we have to bail
#ifdef CG_DEBUG
    fprintf(stderr, "initExtensions false\n");
    fflush(stderr);
#endif
    return false;
  }

#ifdef CG_DEBUG
  fprintf(stderr, "initExtensions true\n");
  fflush(stderr);
#endif
  if (!glewIsSupported("GL_EXT_texture3D")) {
#ifdef CG_DEBUG
    fprintf(stderr, "GL_EXT_texture3D is not supported\n");
    fflush(stderr);
#endif
    return false;
  }

  return true;
}

// Uploads colormapped data
bool CGRGBAImpl::uploadRGBAData(const GLubyte *data, int width, int height,
                                int depth) {
  // bail if we haven't been initialized properly
  if (!m_Initialized)
    return false;

  // clear previous errors
  GLenum error = glGetError();

  if (m_GL_VERSION_1_2)
    glBindTexture(GL_TEXTURE_3D, m_DataTextureName);
  else
    glBindTexture(GL_TEXTURE_3D_EXT, m_DataTextureName);

#ifdef CG_DEBUG
  fprintf(stderr, "glGetError() = %d\n", glGetError());
  fflush(stderr);
#endif

  if (width != m_Width || height != m_Height || depth != m_Depth) {
    if (m_GL_VERSION_1_2) {
#ifdef CG_DEBUG
      printf("GL_VERSION_1_2\n");
#endif
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, width, height, depth, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, data);
    } else
      glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA, width, height, depth, 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, data);
  } else {
    if (m_GL_VERSION_1_2) {
#ifdef CG_DEBUG
      printf("GL_VERSION_1_2\n");
#endif
      glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, width, height, depth,
                      GL_RGBA, GL_UNSIGNED_BYTE, data);
    } else
      glTexSubImage3DEXT(GL_TEXTURE_3D_EXT, 0, 0, 0, 0, width, height, depth,
                         GL_RGBA, GL_UNSIGNED_BYTE, data);
  }
#ifdef CG_DEBUG
  fprintf(stderr, "glGetError() = %d\n", glGetError());
  fflush(stderr);
#endif

  if (m_GL_VERSION_1_2) {
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  } else {
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }
#ifdef CG_DEBUG
  fprintf(stderr, "glGetError() = %d\n", glGetError());
  fflush(stderr);
#endif

  // save the width height and depth
  m_Width = width;
  m_HintDimX = width;
  m_Height = height;
  m_HintDimY = height;
  m_Depth = depth;
  m_HintDimZ = depth;

  // test for error
  error = glGetError();
  if (error == GL_NO_ERROR) {
    return true;
  } else {
    return false;
  }
}

// Tests to see if the given parameters would return an error
bool CGRGBAImpl::testRGBAData(int width, int height, int depth) {
  // bail if we haven't been initialized properly
  if (!m_Initialized)
    return false;

  // nothing above 512
  if (width > 512 || height > 512 || depth > 512) {
    return false;
  }

  // clear previous errors
  GLenum error;
  int c = 0;
  while (glGetError() != GL_NO_ERROR && c < 10)
    c++;

  glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, GL_RGBA, width, height, depth,
                  0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  // test for error
  error = glGetError();
  if (error == GL_NO_ERROR) {
    return true;
  } else {
    return false;
  }
}

// For Shading Uploads Normal data
bool CGRGBAImpl::uploadGradients(const GLubyte *data, int width, int height,
                                 int depth) {
  assert(data != NULL);

  m_Width = -1;
  m_Height = -1;
  m_Depth = -1;

  m_ShadeFlag = true;

  // clear previous errors
  GLenum error = glGetError();
  if (m_GL_VERSION_1_2) {
    glBindTexture(GL_TEXTURE_3D, m_RGB_normals_3DT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  } else {
    glBindTexture(GL_TEXTURE_3D_EXT, m_RGB_normals_3DT);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }

  if (width != m_Width || height != m_Height || depth != m_Depth) {
    if (m_GL_VERSION_1_2) {
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, width, height, depth, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, data);
    } else {
#ifdef TEXTURE_COMPRESSION
      glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                      width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                      data);
#else
      glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA, width, height, depth, 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, data);
#endif
    }
  } else {
    if (m_GL_VERSION_1_2)
      glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, width, height, depth,
                      GL_RGBA, GL_UNSIGNED_BYTE, data);
    else
      glTexSubImage3DEXT(GL_TEXTURE_3D_EXT, 0, 0, 0, 0, width, height, depth,
                         GL_RGBA, GL_UNSIGNED_BYTE, data);
  }

#ifdef CG_DEBUG
  fprintf(stderr,
          "In TEXTURE_COMPRESSION ELSE glGetError() = %d\n m_RGB_normals_3DT "
          "= %d\n",
          glGetError(), m_RGB_normals_3DT);
  fflush(stderr);
#endif

  // save the width height and depth
  m_Width = width;
  m_Height = height;
  m_Depth = depth;

  // test for error
  error = glGetError();
  if (error == GL_NO_ERROR) {
    return true;
  } else {
    return false;
  }
}

void CGRGBAImpl::setLight(float *lightf) {
  for (int i = 0; i < 3; i++)
    m_Light3f[i] = lightf[i];
}

void CGRGBAImpl::setView(float *viewf) {
  for (int i = 0; i < 3; i++)
    m_View3f[i] = viewf[i];
}

// Performs the actual rendering.
bool CGRGBAImpl::renderVolume() {
  // bail if we haven't been initialized properly
  if (!m_Initialized)
    return false;

  // set up the state
  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  // glBlendFunc( GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
  // glBlendFunc( GL_SRC_ALPHA, GL_ONE );
  // glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);

  // bind the data texture
  if (m_GL_VERSION_1_2) {
    glEnable(GL_TEXTURE_3D);
    glActiveTextureARB(GL_TEXTURE0_ARB);
    glBindTexture(GL_TEXTURE_3D, m_DataTextureName);
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glBindTexture(GL_TEXTURE_3D, m_RGB_normals_3DT);
  } else {
    glEnable(GL_TEXTURE_3D_EXT);
    glActiveTextureARB(GL_TEXTURE0_ARB);
    glBindTexture(GL_TEXTURE_3D_EXT, m_DataTextureName);
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glBindTexture(GL_TEXTURE_3D_EXT, m_RGB_normals_3DT);
  }

  computePolygons();

  convertToTriangles();

  if (m_ShadeFlag)
    shadeRenderTriangles();
  else
    renderTriangles();

  // restore the state
  glPopAttrib();

  return true;
}

// Render the actual triangles
void CGRGBAImpl::renderTriangles() {
  //	NV_Unshade_Combiner_Setup();

  // Cg Binding
  cgGLBindProgram(_vertexProgramCg[_cgVertexProgramId]);
  cgGLEnableProfile(_vertexProfileCg);
  cgGLBindProgram(_fragmentProgramCgUnshaded[_cgFragmentProgramId]);
  cgGLEnableProfile(_fragmentProfileCg);

  cgGLSetStateMatrixParameter(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId],
                          "ModelViewProj"),
      CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);

  CGparameter param;
  //-------------------------------------------------------------------------
  // Vertex Program
  // Set Parameter Pointers for the Vertex Program
  param =
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "Position");
  cgGLSetParameterPointer(param, 3, GL_FLOAT, 0, m_VertexArray.get());
  param =
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "TexCoor");
  cgGLSetParameterPointer(param, 3, GL_FLOAT, 0, m_TextureArray.get());

  // Enable the variables
  cgGLEnableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "Position"));
  cgGLEnableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "TexCoor"));

  // set up the client render state
  //	glClientActiveTextureARB(GL_TEXTURE0_ARB);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glEnableClientState(GL_VERTEX_ARRAY);

  glTexCoordPointer(3, GL_FLOAT, 0, m_TextureArray.get());
  glVertexPointer(3, GL_FLOAT, 0, m_VertexArray.get());

  //-------------------------------------------------------------------------
  // Fragment Program
  // Set Parameter Pointers for the Fragment Program
  // Enable the texture parameter as well.
  CGparameter texParam = cgGetNamedParameter(
      _fragmentProgramCgUnshaded[_cgFragmentProgramId], "DataMap");
  cgGLSetTextureParameter(texParam, m_DataTextureName);
  cgGLEnableTextureParameter(texParam);
  //-------------------------------------------------------------------------

  // render the triangles
  glDrawElements(GL_TRIANGLES, m_NumTriangles * 3, GL_UNSIGNED_INT,
                 m_TriangleArray.get());

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);

  //-------------------------------------------------------------------------
  // Vertex Program
  // Disable the variables
  cgGLDisableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "Position"));
  cgGLDisableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "TexCoor"));
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  // Fragment Program
  cgGLDisableTextureParameter(cgGetNamedParameter(
      _fragmentProgramCgUnshaded[_cgFragmentProgramId], "DataMap"));
  //-------------------------------------------------------------------------

  cgGLDisableProfile(_vertexProfileCg);
  cgGLDisableProfile(_fragmentProfileCg);

  /*
          // CG Unshading - also need to change fragment.cg
          // Cg Binding
          cgGLBindProgram(_vertexProgramCg);
          cgGLBindProgram(_fragmentProgramCg);

          cgGLSetStateMatrixParameter(cgGetNamedParameter(_vertexProgramCg,
     "ModelViewProj"), CG_GL_MODELVIEW_PROJECTION_MATRIX,
                                  CG_GL_MATRIX_IDENTITY);

          CGparameter param;
          //-------------------------------------------------------------------------
          // Vertex Program
          // Set Parameter Pointers for the Vertex Program
          param = cgGetNamedParameter(_vertexProgramCg, "Position");
          cgGLSetParameterPointer(param, 3, GL_FLOAT, 0, m_VertexArray);
          param = cgGetNamedParameter(_vertexProgramCg, "TexCoor");
          cgGLSetParameterPointer(param, 3, GL_FLOAT, 0, m_TextureArray);

          // Enable the variables
          cgGLEnableClientState(cgGetNamedParameter(_vertexProgramCg,
     "Position")); cgGLEnableClientState(cgGetNamedParameter(_vertexProgramCg,
     "TexCoor"));
          //-------------------------------------------------------------------------

          //-------------------------------------------------------------------------
          // Fragment Program
          // Set Parameter Pointers for the Fragment Program
          cgGLEnableTextureParameter(cgGetNamedParameter(_fragmentProgramCg,
     "ColorIndexMap"));

          // Drawing Triangle Elements
          glDrawElements(GL_TRIANGLES, m_NumTriangles*3, GL_UNSIGNED_INT,
     m_TriangleArray);

          // Vertex Program
          // Disable the variables
          cgGLDisableClientState(cgGetNamedParameter(_vertexProgramCg,
     "Position"));
          cgGLDisableClientState(cgGetNamedParameter(_vertexProgramCg,
     "TexCoor"));
          //-------------------------------------------------------------------------
  */
}

static inline double m3_det(double *mat) {
  double det;

  det = mat[0] * (mat[4] * mat[8] - mat[7] * mat[5]) -
        mat[1] * (mat[3] * mat[8] - mat[6] * mat[5]) +
        mat[2] * (mat[3] * mat[7] - mat[6] * mat[4]);

  return det;
}

static inline void m4_submat(double *mr, double *mb, int i, int j) {
  int ti, tj, idst, jdst;

  for (ti = 0; ti < 4; ti++) {
    if (ti < i)
      idst = ti;
    else if (ti > i)
      idst = ti - 1;

    for (tj = 0; tj < 4; tj++) {
      if (tj < j)
        jdst = tj;
      else if (tj > j)
        jdst = tj - 1;

      if (ti != i && tj != j)
        mb[idst * 3 + jdst] = mr[ti * 4 + tj];
    }
  }
}
static inline double m4_det(double *mr) {
  double det, result = 0.0, i = 1.0, msub3[9];
  int n;

  for (n = 0; n < 4; n++, i *= -1.0) {
    m4_submat(mr, msub3, 0, n);

    det = m3_det(msub3);
    result += mr[n] * det * i;
  }

  return result;
}

static inline int m4_inverse(double *mr, double *ma) {
  double mtemp[9], mdet = m4_det(ma);
  int i, j, sign;

  if (fabs(mdet) == 0.0)
    return 0;

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      sign = 1 - ((i + j) % 2) * 2;

      m4_submat(ma, mtemp, i, j);

      mr[i + j * 4] = (m3_det(mtemp) * sign) / mdet;
    }
  }

  return 1;
}

static inline void mv_mult(double m[16], float vin[3], float vout[4]) {
  vout[0] = vin[0] * m[0] + vin[1] * m[4] + vin[2] * m[8] + m[12];
  vout[1] = vin[0] * m[1] + vin[1] * m[5] + vin[2] * m[9] + m[13];
  vout[2] = vin[0] * m[2] + vin[1] * m[6] + vin[2] * m[10] + m[14];
  vout[3] = vin[0] * m[3] + vin[1] * m[7] + vin[2] * m[11] + m[15];
}
static inline void normalize(float vin[3]) {
  float div = sqrt(vin[0] * vin[0] + vin[1] * vin[1] + vin[2] * vin[2]);

  if (div != 0.0) {
    vin[0] /= div;
    vin[1] /= div;
    vin[2] /= div;
  }
}

// Render the actual triangles
void CGRGBAImpl::shadeRenderTriangles() {
#ifdef CG_DEBUG
  fprintf(stderr, "start shadeRenderTriangles()\n");
  fflush(stderr);
#endif
  // Cg Binding
  cgGLBindProgram(_vertexProgramCg[_cgVertexProgramId]);
  cgGLEnableProfile(_vertexProfileCg);
  cgGLBindProgram(_fragmentProgramCgShaded[_cgFragmentProgramId]);
  cgGLEnableProfile(_fragmentProfileCg);

  //---------------------------------------------------
#ifdef CG_DEBUG
  fprintf(stderr, "display Sub0: Cg Binding is done\n");
  fflush(stderr);
#endif

  cgGLSetStateMatrixParameter(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId],
                          "ModelViewProj"),
      CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);

  CGparameter param;
  //-------------------------------------------------------------------------
  // Vertex Program
  // Set Parameter Pointers for the Vertex Program
  param =
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "Position");
  cgGLSetParameterPointer(param, 3, GL_FLOAT, 0, m_VertexArray.get());
  param =
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "TexCoor");
  cgGLSetParameterPointer(param, 3, GL_FLOAT, 0, m_TextureArray.get());

  // Enable the variables
  cgGLEnableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "Position"));
  cgGLEnableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "TexCoor"));
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  // Fragment Program
  // Set Parameter Pointers for the Fragment Program
  float lightPosEye[3] = {DEF_LIGHT_POS_EYECOOR_X, DEF_LIGHT_POS_EYECOOR_Y,
                          DEF_LIGHT_POS_EYECOOR_Z};
  float viewPosEye[3] = {0.0, 0.0, 0.0}, scratch[4];
  float lightColor[3] = {DEF_LIGHT_COLOR_R, DEF_LIGHT_COLOR_G,
                         DEF_LIGHT_COLOR_B};
  GLdouble modelview[16], inverse[16];

  // get the modelview matrix
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  // invert it
  m4_inverse(inverse, modelview);
  // set light position and view position in object space.
  mv_mult(inverse, lightPosEye, scratch);
  cgGLSetParameter3fv(
      cgGetNamedParameter(_fragmentProgramCgShaded[_cgFragmentProgramId],
                          "LightPosObj"),
      scratch);
  mv_mult(inverse, viewPosEye, scratch);
  cgGLSetParameter3fv(
      cgGetNamedParameter(_fragmentProgramCgShaded[_cgFragmentProgramId],
                          "ViewPosObj"),
      scratch);
  cgGLSetParameter3fv(
      cgGetNamedParameter(_fragmentProgramCgShaded[_cgFragmentProgramId],
                          "LightColor"),
      lightColor);

  // Enable the texture parameter as well.
  CGparameter texParam = cgGetNamedParameter(
      _fragmentProgramCgShaded[_cgFragmentProgramId], "DataMap");
  cgGLSetTextureParameter(texParam, m_DataTextureName);
  cgGLEnableTextureParameter(texParam);

  texParam = cgGetNamedParameter(
      _fragmentProgramCgShaded[_cgFragmentProgramId], "RGB_NormalMap");
  cgGLSetTextureParameter(texParam, m_RGB_normals_3DT);
  cgGLEnableTextureParameter(texParam);
  //-------------------------------------------------------------------------

  //-----------------------------------------------------------------------
  // Drawing Triangle Elements
  glDrawElements(GL_TRIANGLES, m_NumTriangles * 3, GL_UNSIGNED_INT,
                 m_TriangleArray.get());
  //-----------------------------------------------------------------------

  //-------------------------------------------------------------------------
  // Vertex Program
  // Disable the variables
  cgGLDisableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "Position"));
  cgGLDisableClientState(
      cgGetNamedParameter(_vertexProgramCg[_cgVertexProgramId], "TexCoor"));
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  // Fragment Program
  cgGLDisableTextureParameter(cgGetNamedParameter(
      _fragmentProgramCgShaded[_cgFragmentProgramId], "DataMap"));
  cgGLDisableTextureParameter(cgGetNamedParameter(
      _fragmentProgramCgShaded[_cgFragmentProgramId], "RGB_NormalMap"));
  //-------------------------------------------------------------------------

  cgGLDisableProfile(_vertexProfileCg);
  cgGLDisableProfile(_fragmentProfileCg);
}

void CGRGBAImpl::NV_Unshade_Combiner_Setup(void) {
  // Texture 3D Lookup
  glActiveTextureARB(
      GL_TEXTURE0_ARB); // Index values are transfered to RGBA colors
  glBindTexture(GL_TEXTURE_3D_EXT, m_DataTextureName); // look up image

  glEnable(GL_REGISTER_COMBINERS_NV);

  glCombinerParameteriNV(GL_NUM_GENERAL_COMBINERS_NV, 1);

  glCombinerInputNV(GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_A_NV,
                    GL_TEXTURE0_ARB, GL_UNSIGNED_IDENTITY_NV,
                    GL_RGB); // Indexed Color
  glCombinerInputNV(GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_B_NV, GL_ZERO,
                    GL_UNSIGNED_INVERT_NV, GL_RGB);
  glCombinerInputNV(GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_C_NV, GL_ZERO,
                    GL_UNSIGNED_IDENTITY_NV, GL_RGB);
  glCombinerInputNV(GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_D_NV, GL_ZERO,
                    GL_UNSIGNED_IDENTITY_NV, GL_RGB);

  glCombinerInputNV(GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_A_NV,
                    GL_TEXTURE0_ARB, GL_UNSIGNED_IDENTITY_NV, GL_ALPHA);
  glCombinerInputNV(GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_B_NV, GL_ZERO,
                    GL_UNSIGNED_INVERT_NV, GL_ALPHA);
  glCombinerInputNV(GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_C_NV, GL_ZERO,
                    GL_UNSIGNED_IDENTITY_NV, GL_ALPHA);
  glCombinerInputNV(GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_D_NV, GL_ZERO,
                    GL_UNSIGNED_IDENTITY_NV, GL_ALPHA);

  //								ab
  //cd			sum
  glCombinerOutputNV(GL_COMBINER0_NV, GL_RGB, GL_DISCARD_NV, GL_DISCARD_NV,
                     GL_SPARE1_NV, GL_NONE, GL_NONE, GL_FALSE, GL_FALSE,
                     GL_FALSE); // GL_TRUE means a1 or a2
  glCombinerOutputNV(GL_COMBINER0_NV, GL_ALPHA, GL_DISCARD_NV, GL_DISCARD_NV,
                     GL_SPARE1_NV, GL_NONE, GL_NONE, GL_FALSE, GL_FALSE,
                     GL_FALSE); // GL_FALSE means a1+a2
  // muxSum = GL_TRUE means a1 or a2; return (SPARE0_NV<0.5) ? a1 : a2

  // Final Combiner Formula = A*B + (1-A)*C +D
  // Pre-multiply source-alpha value for final composition
  glFinalCombinerInputNV(GL_VARIABLE_A_NV, GL_SPARE1_NV,
                         GL_UNSIGNED_IDENTITY_NV, GL_ALPHA); // source alpha
  glFinalCombinerInputNV(GL_VARIABLE_B_NV, GL_SPARE1_NV,
                         GL_UNSIGNED_IDENTITY_NV, GL_RGB); // source color
  glFinalCombinerInputNV(GL_VARIABLE_C_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV,
                         GL_RGB); // Zero
  glFinalCombinerInputNV(GL_VARIABLE_D_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV,
                         GL_RGB); // Zero
  glFinalCombinerInputNV(GL_VARIABLE_G_NV, GL_SPARE1_NV,
                         GL_UNSIGNED_IDENTITY_NV, GL_ALPHA);
}
