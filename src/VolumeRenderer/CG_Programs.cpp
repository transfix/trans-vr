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

#ifndef __VOLUME_LIBRARY_CG_PROGRAMS_CPP__
#define __VOLUME_LIBRARY_CG_PROGRAMS_CPP__

#include <VolumeRenderer/CG_Programs.h>
#include <assert.h>
#include <cstdio>

// #define CG_DEBUG

bool _hasCgContext;

using namespace std;

static const char *fragment_cg_shaded =
    "float3 Expand(float3 Vec)"
    "{ return (Vec - 0.5)*2.0; }"
    "float Unsigned_Invert(float Value)"
    "{ return (1.0 - min(max(Value,0), 1.0)); }"
    "void main(	float3 TexCoor			: TEXCOORD0,"
    "             float3 PosObj                   : TEXCOORD1,"
    "             out float4 oColor               : COLOR,"
    "		uniform sampler3D DataMap,"
    "		uniform sampler1D TransferMap,"
    "		uniform sampler3D RGB_NormalMap,"
    "		uniform float3 LightPosObj,"
    "		uniform float3 ViewPosObj,"
    "             uniform float3 LightColor)"
    "{"
    // "// Normalize surface normal, vector to light source, and vector to the
    // viewer"
    "	float DataTex = tex3D(DataMap, TexCoor);"
    "     float4 ColorTex = tex1D(TransferMap, DataTex);"
    "	float4 NormalTex = tex3D(RGB_NormalMap, TexCoor);"
    "     if(NormalTex.w < 0.5) {"
    //         Normal is not defined: force unshaded render
    "        oColor = ColorTex;"
    "     }"
    "     else {"
    "	   float3 NormalObj = normalize(Expand(NormalTex.xyz));"
    "        float3 l = normalize(LightPosObj - PosObj);"
    "        float3 v = normalize(ViewPosObj - PosObj);"
    "        float3 h = normalize(l + v);"
    "        if(dot(NormalObj,v)<0.0f)"
    "           NormalObj = -NormalObj;"
    "	   float Diffuse = clamp(dot(NormalObj, l), 0.0, 1.0);"
    "	   float Specular0 = clamp(dot(NormalObj, h), 0.0, 1.0);"
    "	   float Specular1 = Specular0*Specular0;" //^2
    "        Specular0 = Specular1*Specular1;"     //^4
    "        Specular1 = Specular0*Specular0;"     //^8
    "        Specular0 = Specular1*Specular1;"     //^16
    "        Specular1 = Specular0*Specular0;"     //^32
    "	   float3 SpecularColor = float3(Specular1, Specular1, Specular1);"
    "        oColor.xyz = (Diffuse*ColorTex.xyz + SpecularColor)*LightColor;"
    "	   oColor.w = ColorTex.w;"
    "     }"
    "}";

static const char *fragment_cg_unshaded =
    "void main(	float3 TexCoor			: TEXCOORD0,"
    "             out float4 oColor               : COLOR,"
    "		uniform sampler3D DataMap,"
    "		uniform sampler1D TransferMap)"
    "{"
    // "// Normalize surface normal, vector to light source, and vector to the
    // viewer"
    "	float DataTex = tex3D(DataMap, TexCoor);"
    "     float4 ColorTex = tex1D(TransferMap, DataTex);"
    "     oColor = ColorTex;"
    "}";

static const char *fragment_cg_rgba_shaded =
    "float3 Expand(float3 Vec)"
    "{ return (Vec - 0.5)*2.0; }"
    "float Unsigned_Invert(float Value)"
    "{ return (1.0 - min(max(Value,0), 1.0)); }"
    "void main(	float3 TexCoor			: TEXCOORD0,"
    "             float3 PosObj                   : TEXCOORD1,"
    "             out float4 oColor               : COLOR,"
    "		uniform sampler3D DataMap,"
    "		uniform sampler3D RGB_NormalMap,"
    "		uniform float3 LightPosObj,"
    "		uniform float3 ViewPosObj,"
    "             uniform float3 LightColor)"
    "{"
    // "// Normalize surface normal, vector to light source, and vector to the
    // viewer"
    "	float4 ColorTex = tex3D(DataMap, TexCoor);"
    "	float4 NormalTex = tex3D(RGB_NormalMap, TexCoor);"
    "     if(NormalTex.w < 0.5) {"
    //         Normal is not defined: force unshaded render
    "        oColor = ColorTex;"
    "     }"
    "     else {"
    "	   float3 NormalObj = normalize(Expand(NormalTex.xyz));"
    "        float3 l = normalize(LightPosObj - PosObj);"
    "        float3 v = normalize(ViewPosObj - PosObj);"
    "        float3 h = normalize(l + v);"
    "        if(dot(NormalObj,v)<0.0f)"
    "           NormalObj = -NormalObj;"
    "	   float Diffuse = clamp(dot(NormalObj, l), 0.0, 1.0);"
    "	   float Specular0 = clamp(dot(NormalObj, h), 0.0, 1.0);"
    "	   float Specular1 = Specular0*Specular0;" //^2
    "        Specular0 = Specular1*Specular1;"     //^4
    "        Specular1 = Specular0*Specular0;"     //^8
    "        Specular0 = Specular1*Specular1;"     //^16
    "        Specular1 = Specular0*Specular0;"     //^32
    "	   float3 SpecularColor = float3(Specular1, Specular1, Specular1);"
    "        oColor.xyz = (Diffuse*ColorTex.xyz + SpecularColor)*LightColor;"
    "	   oColor.w = ColorTex.w;"
    "     }"
    "}";

static const char *fragment_cg_rgba_unshaded =
    "void main(	float3 TexCoor			: TEXCOORD0,"
    "             out float4 oColor               : COLOR,"
    "		uniform sampler3D DataMap)"
    "{"
    "	float4 ColorTex = tex3D(DataMap, TexCoor);"
    "     oColor = ColorTex;"
    "}";

static const char *vertex_cg =
    "void main(	float4 Position		: POSITION,"
    "		float3 TexCoor		: TEXCOORD0,"
    "		uniform float4x4 ModelViewProj,"
    "		out float4 oPosition	:POSITION,"
    "		out float3 oTexCoor	:TEXCOORD0,"
    "             out float3 oPosObj      :TEXCOORD1)"
    "{"
    "	oPosition = mul(ModelViewProj, Position);"
    "	oTexCoor = TexCoor;"
    "     oPosObj = Position.xyz;"
    "}";

bool CG_Programs::_cgErrorFlag = false;

// 05/11/2012 - transfix - initialzing id's to avoid a crash
CG_Programs::CG_Programs() : _cgVertexProgramId(0), _cgFragmentProgramId(0) {}

CG_Programs::~CG_Programs() {
  // it seems that VolumeRover uses multi-threads, so, destroying class might
  // be considered.
  //  cgDestroyProgram(_vertexProgramCg[_cgVertexProgramId]);
  //  cgDestroyProgram(_fragmentProgramCg[_cgFragmentProgramId]);
  if (_hasCgContext) {
    cgDestroyContext(_contextCg);
    _hasCgContext = false;
  }
}

void CG_Programs::handleCgError() {
  fprintf(stderr, "%s\n", cgGetErrorString(cgGetError()));
  _cgErrorFlag = false;
}

bool CG_Programs::initCG(int renderType) {
  //-----------------------------------------------------------------
  // Basic Cg setup; register a callback function for any errors
  // and create an initial context
  cgSetErrorCallback(CG_Programs::handleCgError);

  static bool _cgInitialized = false;
  if (_cgInitialized == false) {
    _vertexProgramCgCounter = 0;
    _fragmentProgramCgCounter = 0;
    _cgVertexProgramId = _cgFragmentProgramId =
        0; // making sure these are zero

    _vertexProfileCg = CG_PROFILE_VP30;
    _fragmentProfileCg = CG_PROFILE_FP30;

    if (!(_contextCg = cgCreateContext())) {
      cgDestroyContext(_contextCg);
      return false;
    }
    _hasCgContext = true;

#ifdef CG_DEBUG
    fprintf(stderr, "context created\n");
#endif

    if (!ChooseProfiles()) {
      fprintf(stderr, "Error in ChooseProfiles()\n");
      return false;
    }
#ifdef CG_DEBUG
    fprintf(stderr, "profile intialized\n");
#endif
    _cgInitialized = true;
  }

  if (_vertexProgramCgCounter >= MAX_CG_PROGRAM) {
    fprintf(stderr, "Too many vertex programs\n");
    return false;
  } else
    _cgVertexProgramId = _vertexProgramCgCounter++;

  if (_fragmentProgramCgCounter >= MAX_CG_PROGRAM) {
    fprintf(stderr, "Too many fragment programs\n");
    return false;
  } else
    _cgFragmentProgramId = _fragmentProgramCgCounter++;

  // Do one-time setup only once; setup Cg programs and textures
  // and set up OpenGL state.
  if (!LoadCgPrograms(renderType)) {
    fprintf(stderr, "Error in LoadCgPrograms()\n");
    // cgDestroyProgram(_vertexProgramCg);
    return false;
  } else {
#ifdef CG_DEBUG
    fprintf(stderr, "Program loaded\n");
#endif
    return true;
  }
}

bool CG_Programs::ChooseProfiles() {

#ifdef CG_DEBUG
  fprintf(stderr, "ChooseProfiles:\n");
  fflush(stderr);
#endif

  // Make sure that the appropriate profiles are available on the
  // user's system.
  if (cgGLIsProfileSupported(CG_PROFILE_VP30)) {
    _vertexProfileCg = CG_PROFILE_VP30;
#ifdef CG_DEBUG
    fprintf(stderr, "CG_PROFILE_VP30 is selected for the vertex profile\n");
    fflush(stderr);
#endif
  } else {
    // try VP30
    if (cgGLIsProfileSupported(CG_PROFILE_ARBVP1)) {
      _vertexProfileCg = CG_PROFILE_ARBVP1;
#ifdef CG_DEBUG
      fprintf(stderr,
              "CG_PROFILE_ARBVP1 is selected for the vertex profile\n");
      fflush(stderr);
#endif
    } else {
#ifdef CG_DEBUG
      fprintf(stderr, "Neither arbvp1 or vp30 vertex profiles supported on "
                      "this system\n");
#endif
      return false;
    }
  }

  if (cgGLIsProfileSupported(CG_PROFILE_FP30)) {
    _fragmentProfileCg = CG_PROFILE_FP30;
#ifdef CG_DEBUG
    fprintf(stderr, "CG_PROFILE_FP30 is selected for the fragment profile\n");
    fflush(stderr);
#endif
  } else {
    // try FP30
    if (cgGLIsProfileSupported(CG_PROFILE_ARBFP1)) {
      _fragmentProfileCg = CG_PROFILE_ARBFP1;
#ifdef CG_DEBUG
      fprintf(stderr,
              "CG_PROFILE_ARBFP1 is selected for the fragment profile\n");
      fflush(stderr);
#endif
    } else {
#ifdef CG_DEBUG
      fprintf(stderr, "Neither arbfp1 or fp30 vertex profiles supported on "
                      "this system.\n");
#endif
      return false;
    }
  }

  return true;
}

bool CG_Programs::LoadCgPrograms(int renderType) {

#ifdef CG_DEBUG
  fprintf(stderr, "Load Cg Programs:\n");
  fflush(stderr);
#endif

  assert(cgIsContext(_contextCg));

#ifdef CG_DEBUG
  fprintf(stderr, "assert(cgIsContext(_contextCg))\n");
  fflush(stderr);
#endif

  // Load and compile the vertex program from demo_vert.cg; hold on to the
  // handle to it that is returned.
#if 0
  if(!(_vertexProgramCg = 
       cgCreateProgramFromFile(_contextCg, CG_SOURCE, 
                               "./CG/vertex.cg", _vertexProfileCg, NULL, NULL)) ) {
    return false;			
  }
#endif
  if (!(_vertexProgramCg[_cgVertexProgramId] =
            cgCreateProgram(_contextCg, CG_SOURCE, vertex_cg,
                            _vertexProfileCg, NULL, NULL))) {
    return false;
  }

#ifdef CG_DEBUG
  fprintf(stderr, "_vertexProgramCg created\n");
  fflush(stderr);
#endif

  if (!cgIsProgramCompiled(_vertexProgramCg[_cgVertexProgramId]))
    cgCompileProgram(_vertexProgramCg[_cgVertexProgramId]);

#ifdef CG_DEBUG
  fprintf(stderr, "Load Cg Programs: Compile Vertex Program\n");
  fflush(stderr);
#endif

  // Enable the appropriate vertex profile and load the vertex program.
  cgGLEnableProfile(_vertexProfileCg);
  cgGLLoadProgram(_vertexProgramCg[_cgVertexProgramId]);
  cgGLDisableProfile(_vertexProfileCg);

#ifdef CG_DEBUG
  fprintf(stderr, "Load Cg Programs: Load Vertex Program\n");
  fflush(stderr);
#endif

  // And similarly set things up for the fragment program.
#if 0
  if(!(_fragmentProgramCg = 
       cgCreateProgramFromFile(_contextCg, CG_SOURCE, 
                               "./CG/fragment.cg", _fragmentProfileCg, NULL, NULL)) ) {
    return false;			
  }
#endif
  if (renderType == DEF_CG_RGBA) {
    if (!(_fragmentProgramCgShaded[_cgFragmentProgramId] =
              cgCreateProgram(_contextCg, CG_SOURCE, fragment_cg_rgba_shaded,
                              _fragmentProfileCg, NULL, NULL))) {
      return false;
    }
    if (!(_fragmentProgramCgUnshaded[_cgFragmentProgramId] = cgCreateProgram(
              _contextCg, CG_SOURCE, fragment_cg_rgba_unshaded,
              _fragmentProfileCg, NULL, NULL))) {
      return false;
    }

  } else {
    if (!(_fragmentProgramCgShaded[_cgFragmentProgramId] =
              cgCreateProgram(_contextCg, CG_SOURCE, fragment_cg_shaded,
                              _fragmentProfileCg, NULL, NULL))) {
      return false;
    }
    if (!(_fragmentProgramCgUnshaded[_cgFragmentProgramId] =
              cgCreateProgram(_contextCg, CG_SOURCE, fragment_cg_unshaded,
                              _fragmentProfileCg, NULL, NULL))) {
      return false;
    }
  }

#ifdef CG_DEBUG
  fprintf(stderr, "_fragmentProgramCg created\n");
  fflush(stderr);
#endif

  if (!cgIsProgramCompiled(_fragmentProgramCgShaded[_cgFragmentProgramId]))
    cgCompileProgram(_fragmentProgramCgShaded[_cgFragmentProgramId]);

  if (!cgIsProgramCompiled(_fragmentProgramCgUnshaded[_cgFragmentProgramId]))
    cgCompileProgram(_fragmentProgramCgUnshaded[_cgFragmentProgramId]);

#ifdef CG_DEBUG
  fprintf(stderr, "Load Cg Programs: Compile Fragment Program\n");
  fflush(stderr);
#endif

  cgGLEnableProfile(_fragmentProfileCg);
  cgGLLoadProgram(_fragmentProgramCgShaded[_cgFragmentProgramId]);
  cgGLLoadProgram(_fragmentProgramCgUnshaded[_cgFragmentProgramId]);
  cgGLDisableProfile(_fragmentProfileCg);

#ifdef CG_DEBUG
  fprintf(stderr, "Load Cg Programs: Load Fragment Program\n");
  fflush(stderr);
#endif

  return true;
}

#endif //__VOLUME_LIBRARY_CG_PROGRAMS_CPP__
