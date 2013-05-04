/*
  Copyright 2002-2003,2011 The University of Texas at Austin

	Authors: Anthony Thane <thanea@ices.utexas.edu>
                 Jose Rivera   <transfix@ices.utexas.edu>
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

#ifndef __FRAGMENTPROGRAMGLSLIMPL_H__
#define __FRAGMENTPROGRAMGLSLIMPL_H__

#include <stdexcept>
#include <vector>
#include <map>
#include <string>
#include <boost/format.hpp>
#include <boost/array.hpp>

#include <glew/glew.h>

#include <VolumeRenderer/UnshadedBase.h>

namespace OpenGLVolumeRendering {
	
  /// A volume renderer which uses GLSL fragment programs to perform color mapping
  // 10/07/2011 -- transfix -- removed setShaders/unsetShaders
  class FragmentProgramGLSLImpl : public UnshadedBase  
  {
  public:

#define DEF_EXCEPTION(name) \
  class name : public std::exception      \
  { \
  public: \
    name () : _msg("Viewer::"#name) {} \
    name (const std::string& msg) : \
      _msg(boost::str(boost::format("Viewer::" #name " exception: %1%") % msg)) {} \
    virtual ~name() throw() {} \
    virtual const std::string& what_str() const throw() { return _msg; } \
    virtual const char * what() const throw () { return _msg.c_str(); } \
  private: \
    std::string _msg; \
  }

  DEF_EXCEPTION(UninitializedException);
  DEF_EXCEPTION(ShaderCompileException);
  DEF_EXCEPTION(ShaderLinkException);
  DEF_EXCEPTION(InvalidProgramException);
  DEF_EXCEPTION(InsufficientHardwareException);

#undef DEF_EXCEPTION

    FragmentProgramGLSLImpl();
    virtual ~FragmentProgramGLSLImpl();
		
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
    void setVertexShader(const char *prog);
    void setFragmentShader(const char *prog);
    
    void unsetVertexShader();
    void unsetFragmentShader();  
    
    //returns true only after init() is called
    bool initialized() const { return _initialized; }
    
    //update info passed to shaders
    void updateUniforms();

    void printUniformInfo(const std::string& uniformName);
    std::string getGLSLLogString() const;

    // Initializes the necessary extensions.
    virtual bool initExtensions();

    // Gets the opengl texture IDs
    bool initTextureNames();
		
    // Gets the fragment program ready
    bool initFragmentProgram();

    // Render the actual triangles
    void renderTriangles();

  bool _initialized;
    
  bool _isGL2_0;
  bool _glslAvailable;

  GLuint _program;
  GLuint _vertexShader;
  GLuint _fragmentShader;

  //for debugging... mapping uniform type enums to strings
  std::map<GLenum,std::string> _GLSL_UNIFORM_TYPE_MAP;
    
  //data used by shaders
  typedef boost::array<float,3> vec3;
  std::vector<vec3> _lights; //vector of directional lights
  GLuint _lightsTex;
  std::vector<vec3> _lightColors;
  GLuint _lightColorsTex;

    // Remembers the uploaded width height and depth
    int m_Width, m_Height, m_Depth;

    // The opengl texture ID
    GLuint m_DataTextureName;

    // The transfer function texture ID
    GLuint m_TransferTextureName;
  };
};

#endif

