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

// MyExtensions.cpp: implementation of the MyExtensions class.
//
//////////////////////////////////////////////////////////////////////

#define DECLARE_STATIC_POINTERS
#include <VolumeRenderer/MyExtensions.h>

#include <string.h>

#define STATIC_POINTER_PREFIX staticPointer

#define CREATESTATICNAME2(a,b) a##b
#define CREATESTATICNAME(a,b) CREATESTATICNAME2(a,b)

// define a macro for getting the address of a function
#if defined(_WIN32)
// use the windows technique
#define INIT_PROC_POINTER(pointerType,procName) procName = (pointerType) wglGetProcAddress(#procName) ; if (procName==0) return false
#elif defined(GLX_ARB_get_proc_address)
// use glXGetProcAddress
#define INIT_PROC_POINTER(pointerType,procName) procName = (pointerType) glXGetProcAddressARB( (const GLubyte *) #procName) ; if (procName==0) return false
#else
// try to get the static location of the function
#define INIT_PROC_POINTER(pointerType,procName) procName = (pointerType) CREATESTATICNAME(STATIC_POINTER_PREFIX,procName) ; if (procName==0) return false
#endif

// define a macro for getting addresses of wgl functions
#if defined(_WIN32)
#define INIT_WGL_PROC_POINTER(pointerType,procName) procName = (pointerType) wglGetProcAddress(#procName) ; if (procName==0) return false
#else
// wgl functions are only supported in windows
#define INIT_WGL_PROC_POINTER(pointerType,procName) procName = 0 ; return false
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MyExtensions::MyExtensions()
{
	initPointersToZero();
	initStaticPointers();
}

MyExtensions::~MyExtensions()
{

}

bool MyExtensions::extensionExists(const char* extension, const char* allExtensions)
{
	int extLen = strlen(extension);
	char* padExtName = new char[extLen + 2];
	strcpy(padExtName, extension);
	padExtName[extLen] = ' ';
	padExtName[extLen+1] = 0;

    if (0 == strcmp(extension, "GL_VERSION_1_2")) {
        const char *version = (const char*)glGetString(GL_VERSION);
        if (strstr(version, "1.0") == version || strstr(version, "1.1") == version) {
            return false;
        } else {
            return true;
        }
    }
    if (0 == strcmp(extension, "GL_VERSION_1_3")) {
        const char *version = (const char*)glGetString(GL_VERSION);
        if (strstr(version, "1.0") == version ||
            strstr(version, "1.1") == version ||
            strstr(version, "1.2") == version) {
            return false;
        } else {
            return true;
        }
    }
    if (0 == strcmp(extension, "GL_VERSION_1_4")) {
        const char *version = (const char*)glGetString(GL_VERSION);
        if (strstr(version, "1.0") == version ||
            strstr(version, "1.1") == version ||
            strstr(version, "1.2") == version ||
            strstr(version, "1.3") == version) {
            return false;
        } else {
            return true;
        }
    }
    if (strstr(allExtensions, padExtName)) {
        delete [] padExtName;
        return true;
    } 
	else {
        delete [] padExtName;
        return false;
    }
}

bool MyExtensions::checkExtensions(const char* requestedExtensions)
{
	MyExtensions myExtensions;
	return myExtensions.initExtensions(requestedExtensions);
}

bool MyExtensions::initExtensions(const char* requestedExtensions)
{
	if (!requestedExtensions) {
		return false;
	}

	
	// get the extensions string and pad it
	const char *extensions = getExtensionStringPrivate();
	int normalExtsLen = strlen(extensions);

	const char* systemExtensions = getSystemExtensions();
	int systemExtsLen; 

	char* paddedExtensions;
	if (systemExtensions) {
		systemExtsLen = strlen(systemExtensions);
		paddedExtensions = new char[normalExtsLen + 1 + systemExtsLen + 2];
		strcpy(paddedExtensions, extensions);
		paddedExtensions[normalExtsLen] = ' ';
		strcpy(paddedExtensions+normalExtsLen+1, systemExtensions);
		paddedExtensions[normalExtsLen + 1 + systemExtsLen] = ' ';
		paddedExtensions[normalExtsLen + 1 + systemExtsLen+1] = 0;
	}
	else {
		systemExtsLen = 0;
		paddedExtensions = new char[normalExtsLen+2];
		strcpy(paddedExtensions, extensions);
		paddedExtensions[normalExtsLen] = ' ';
		paddedExtensions[normalExtsLen+1] = 0;
	}

		
	// duplicate the requested string
    int reqExtsLen = strlen(requestedExtensions);
    char* reqExts = new char[reqExtsLen+1];
	strcpy(reqExts, requestedExtensions);

	char* currentExt;
	bool success = true;
	
    // Parse requested extension list
    for (currentExt = reqExts; (currentExt = (char*)EatWhiteSpace(currentExt)) && *currentExt; currentExt = (char*)EatNonWhiteSpace(currentExt))
	{
        char *extEnd = (char*)EatNonWhiteSpace(currentExt);
        char saveChar = *extEnd;
        *extEnd = (char)0;

		if (!extensionExists(currentExt, paddedExtensions) || !initExtension(currentExt)) {
			// failed
			success = false;
		}

		*extEnd = saveChar;
	}
	delete [] paddedExtensions;
	delete [] reqExts;
	return success;
}

bool MyExtensions::initExtension(const char* extension)
{
	if (0 == extension) {
		return false;
	}
#ifdef GL_VERSION_1_2
	else if (0 == strcmp(extension, "GL_VERSION_1_2")) {
		INIT_PROC_POINTER(MYPFNGLBLENDCOLORPROC, glBlendColor);
		INIT_PROC_POINTER(MYPFNGLBLENDEQUATIONPROC, glBlendEquation);
		INIT_PROC_POINTER(MYPFNGLDRAWRANGEELEMENTSPROC, glDrawRangeElements);
		INIT_PROC_POINTER(MYPFNGLCOLORTABLEPROC, glColorTable);
		INIT_PROC_POINTER(MYPFNGLCOLORTABLEPARAMETERFVPROC, glColorTableParameterfv);
		INIT_PROC_POINTER(MYPFNGLCOLORTABLEPARAMETERIVPROC, glColorTableParameteriv);
		INIT_PROC_POINTER(MYPFNGLCOPYCOLORTABLEPROC, glCopyColorTable);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEPROC, glGetColorTable);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEPARAMETERFVPROC, glGetColorTableParameterfv);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEPARAMETERIVPROC, glGetColorTableParameteriv);
		INIT_PROC_POINTER(MYPFNGLCOLORSUBTABLEPROC, glColorSubTable);
		INIT_PROC_POINTER(MYPFNGLCOPYCOLORSUBTABLEPROC, glCopyColorSubTable);
		INIT_PROC_POINTER(MYPFNGLCONVOLUTIONFILTER1DPROC, glConvolutionFilter1D);
		INIT_PROC_POINTER(MYPFNGLCONVOLUTIONFILTER2DPROC, glConvolutionFilter2D);
		INIT_PROC_POINTER(MYPFNGLCONVOLUTIONPARAMETERFPROC, glConvolutionParameterf);
		INIT_PROC_POINTER(MYPFNGLCONVOLUTIONPARAMETERFVPROC, glConvolutionParameterfv);
		INIT_PROC_POINTER(MYPFNGLCONVOLUTIONPARAMETERIPROC, glConvolutionParameteri);
		INIT_PROC_POINTER(MYPFNGLCONVOLUTIONPARAMETERIVPROC, glConvolutionParameteriv);
		INIT_PROC_POINTER(MYPFNGLCOPYCONVOLUTIONFILTER1DPROC, glCopyConvolutionFilter1D);
		INIT_PROC_POINTER(MYPFNGLCOPYCONVOLUTIONFILTER2DPROC, glCopyConvolutionFilter2D);
		INIT_PROC_POINTER(MYPFNGLGETCONVOLUTIONFILTERPROC, glGetConvolutionFilter);
		INIT_PROC_POINTER(MYPFNGLGETCONVOLUTIONPARAMETERFVPROC, glGetConvolutionParameterfv);
		INIT_PROC_POINTER(MYPFNGLGETCONVOLUTIONPARAMETERIVPROC, glGetConvolutionParameteriv);
		INIT_PROC_POINTER(MYPFNGLGETSEPARABLEFILTERPROC, glGetSeparableFilter);
		INIT_PROC_POINTER(MYPFNGLSEPARABLEFILTER2DPROC, glSeparableFilter2D);
		INIT_PROC_POINTER(MYPFNGLGETHISTOGRAMPROC, glGetHistogram);
		INIT_PROC_POINTER(MYPFNGLGETHISTOGRAMPARAMETERFVPROC, glGetHistogramParameterfv);
		INIT_PROC_POINTER(MYPFNGLGETHISTOGRAMPARAMETERIVPROC, glGetHistogramParameteriv);
		INIT_PROC_POINTER(MYPFNGLGETMINMAXPROC, glGetMinmax);
		INIT_PROC_POINTER(MYPFNGLGETMINMAXPARAMETERFVPROC, glGetMinmaxParameterfv);
		INIT_PROC_POINTER(MYPFNGLGETMINMAXPARAMETERIVPROC, glGetMinmaxParameteriv);
		INIT_PROC_POINTER(MYPFNGLHISTOGRAMPROC, glHistogram);
		INIT_PROC_POINTER(MYPFNGLMINMAXPROC, glMinmax);
		INIT_PROC_POINTER(MYPFNGLRESETHISTOGRAMPROC, glResetHistogram);
		INIT_PROC_POINTER(MYPFNGLRESETMINMAXPROC, glResetMinmax);
		INIT_PROC_POINTER(MYPFNGLTEXIMAGE3DPROC, glTexImage3D);
		INIT_PROC_POINTER(MYPFNGLTEXSUBIMAGE3DPROC, glTexSubImage3D);
		INIT_PROC_POINTER(MYPFNGLCOPYTEXSUBIMAGE3DPROC, glCopyTexSubImage3D);
		return true;
	}
#endif // GL_VERSION_1_2

#ifdef GL_VERSION_1_3
	else if (0 == strcmp(extension, "GL_VERSION_1_3")) {
		INIT_PROC_POINTER(MYPFNGLACTIVETEXTUREPROC, glActiveTexture);
		INIT_PROC_POINTER(MYPFNGLCLIENTACTIVETEXTUREPROC, glClientActiveTexture);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1DPROC, glMultiTexCoord1d);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1DVPROC, glMultiTexCoord1dv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1FPROC, glMultiTexCoord1f);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1FVPROC, glMultiTexCoord1fv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1IPROC, glMultiTexCoord1i);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1IVPROC, glMultiTexCoord1iv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1SPROC, glMultiTexCoord1s);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1SVPROC, glMultiTexCoord1sv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2DPROC, glMultiTexCoord2d);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2DVPROC, glMultiTexCoord2dv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2FPROC, glMultiTexCoord2f);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2FVPROC, glMultiTexCoord2fv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2IPROC, glMultiTexCoord2i);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2IVPROC, glMultiTexCoord2iv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2SPROC, glMultiTexCoord2s);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2SVPROC, glMultiTexCoord2sv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3DPROC, glMultiTexCoord3d);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3DVPROC, glMultiTexCoord3dv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3FPROC, glMultiTexCoord3f);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3FVPROC, glMultiTexCoord3fv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3IPROC, glMultiTexCoord3i);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3IVPROC, glMultiTexCoord3iv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3SPROC, glMultiTexCoord3s);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3SVPROC, glMultiTexCoord3sv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4DPROC, glMultiTexCoord4d);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4DVPROC, glMultiTexCoord4dv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4FPROC, glMultiTexCoord4f);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4FVPROC, glMultiTexCoord4fv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4IPROC, glMultiTexCoord4i);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4IVPROC, glMultiTexCoord4iv);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4SPROC, glMultiTexCoord4s);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4SVPROC, glMultiTexCoord4sv);
		INIT_PROC_POINTER(MYPFNGLLOADTRANSPOSEMATRIXFPROC, glLoadTransposeMatrixf);
		INIT_PROC_POINTER(MYPFNGLLOADTRANSPOSEMATRIXDPROC, glLoadTransposeMatrixd);
		INIT_PROC_POINTER(MYPFNGLMULTTRANSPOSEMATRIXFPROC, glMultTransposeMatrixf);
		INIT_PROC_POINTER(MYPFNGLMULTTRANSPOSEMATRIXDPROC, glMultTransposeMatrixd);
		INIT_PROC_POINTER(MYPFNGLSAMPLECOVERAGEPROC, glSampleCoverage);
		INIT_PROC_POINTER(MYPFNGLCOMPRESSEDTEXIMAGE3DPROC, glCompressedTexImage3D);
		INIT_PROC_POINTER(MYPFNGLCOMPRESSEDTEXIMAGE2DPROC, glCompressedTexImage2D);
		INIT_PROC_POINTER(MYPFNGLCOMPRESSEDTEXIMAGE1DPROC, glCompressedTexImage1D);
		INIT_PROC_POINTER(MYPFNGLCOMPRESSEDTEXSUBIMAGE3DPROC, glCompressedTexSubImage3D);
		INIT_PROC_POINTER(MYPFNGLCOMPRESSEDTEXSUBIMAGE2DPROC, glCompressedTexSubImage2D);
		INIT_PROC_POINTER(MYPFNGLCOMPRESSEDTEXSUBIMAGE1DPROC, glCompressedTexSubImage1D);
		INIT_PROC_POINTER(MYPFNGLGETCOMPRESSEDTEXIMAGEPROC, glGetCompressedTexImage);
		return initExtension("GL_VERSION_1_2");
	}
#endif // GL_VERSION_1_3

#ifdef GL_VERSION_1_4
	else if (0 == strcmp(extension, "GL_VERSION_1_4")) {
		INIT_PROC_POINTER(MYPFNGLBLENDFUNCSEPARATEPROC, glBlendFuncSeparate);
		INIT_PROC_POINTER(MYPFNGLFOGCOORDFPROC, glFogCoordf);
		INIT_PROC_POINTER(MYPFNGLFOGCOORDFVPROC, glFogCoordfv);
		INIT_PROC_POINTER(MYPFNGLFOGCOORDDPROC, glFogCoordd);
		INIT_PROC_POINTER(MYPFNGLFOGCOORDDVPROC, glFogCoorddv);
		INIT_PROC_POINTER(MYPFNGLFOGCOORDPOINTERPROC, glFogCoordPointer);
		INIT_PROC_POINTER(MYPFNGLMULTIDRAWARRAYSPROC, glMultiDrawArrays);
		INIT_PROC_POINTER(MYPFNGLMULTIDRAWELEMENTSPROC, glMultiDrawElements);
		INIT_PROC_POINTER(MYPFNGLPOINTPARAMETERFPROC, glPointParameterf);
		INIT_PROC_POINTER(MYPFNGLPOINTPARAMETERFVPROC, glPointParameterfv);
		INIT_PROC_POINTER(MYPFNGLPOINTPARAMETERIPROC, glPointParameteri);
		INIT_PROC_POINTER(MYPFNGLPOINTPARAMETERIVPROC, glPointParameteriv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3BPROC, glSecondaryColor3b);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3BVPROC, glSecondaryColor3bv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3DPROC, glSecondaryColor3d);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3DVPROC, glSecondaryColor3dv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3FPROC, glSecondaryColor3f);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3FVPROC, glSecondaryColor3fv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3IPROC, glSecondaryColor3i);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3IVPROC, glSecondaryColor3iv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3SPROC, glSecondaryColor3s);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3SVPROC, glSecondaryColor3sv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3UBPROC, glSecondaryColor3ub);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3UBVPROC, glSecondaryColor3ubv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3UIPROC, glSecondaryColor3ui);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3UIVPROC, glSecondaryColor3uiv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3USPROC, glSecondaryColor3us);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLOR3USVPROC, glSecondaryColor3usv);
		INIT_PROC_POINTER(MYPFNGLSECONDARYCOLORPOINTERPROC, glSecondaryColorPointer);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2DPROC, glWindowPos2d);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2DVPROC, glWindowPos2dv);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2FPROC, glWindowPos2f);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2FVPROC, glWindowPos2fv);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2IPROC, glWindowPos2i);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2IVPROC, glWindowPos2iv);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2SPROC, glWindowPos2s);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS2SVPROC, glWindowPos2sv);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3DPROC, glWindowPos3d);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3DVPROC, glWindowPos3dv);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3FPROC, glWindowPos3f);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3FVPROC, glWindowPos3fv);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3IPROC, glWindowPos3i);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3IVPROC, glWindowPos3iv);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3SPROC, glWindowPos3s);
		INIT_PROC_POINTER(MYPFNGLWINDOWPOS3SVPROC, glWindowPos3sv);
		return initExtension("GL_VERSION_1_3");
	}
#endif // GL_VERSION_1_4

#ifdef GL_EXT_paletted_texture
	else if (0 == strcmp(extension, "GL_EXT_paletted_texture")) {
		INIT_PROC_POINTER(MYPFNGLCOLORTABLEEXTPROC, glColorTableEXT);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEEXTPROC, glGetColorTableEXT);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEPARAMETERIVEXTPROC, glGetColorTableParameterivEXT);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEPARAMETERFVEXTPROC, glGetColorTableParameterfvEXT);
		return true;
	}
#endif // GL_EXT_paletted_texture

#ifdef GL_ARB_multitexture
	else if (0 == strcmp(extension, "GL_ARB_multitexture")) {
		INIT_PROC_POINTER(MYPFNGLACTIVETEXTUREARBPROC, glActiveTextureARB);
		INIT_PROC_POINTER(MYPFNGLCLIENTACTIVETEXTUREARBPROC, glClientActiveTextureARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1DARBPROC, glMultiTexCoord1dARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1DVARBPROC, glMultiTexCoord1dvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1FARBPROC, glMultiTexCoord1fARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1FVARBPROC, glMultiTexCoord1fvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1IARBPROC, glMultiTexCoord1iARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1IVARBPROC, glMultiTexCoord1ivARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1SARBPROC, glMultiTexCoord1sARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD1SVARBPROC, glMultiTexCoord1svARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2DARBPROC, glMultiTexCoord2dARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2DVARBPROC, glMultiTexCoord2dvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2FARBPROC, glMultiTexCoord2fARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2FVARBPROC, glMultiTexCoord2fvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2IARBPROC, glMultiTexCoord2iARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2IVARBPROC, glMultiTexCoord2ivARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2SARBPROC, glMultiTexCoord2sARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD2SVARBPROC, glMultiTexCoord2svARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3DARBPROC, glMultiTexCoord3dARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3DVARBPROC, glMultiTexCoord3dvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3FARBPROC, glMultiTexCoord3fARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3FVARBPROC, glMultiTexCoord3fvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3IARBPROC, glMultiTexCoord3iARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3IVARBPROC, glMultiTexCoord3ivARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3SARBPROC, glMultiTexCoord3sARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD3SVARBPROC, glMultiTexCoord3svARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4DARBPROC, glMultiTexCoord4dARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4DVARBPROC, glMultiTexCoord4dvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4FARBPROC, glMultiTexCoord4fARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4FVARBPROC, glMultiTexCoord4fvARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4IARBPROC, glMultiTexCoord4iARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4IVARBPROC, glMultiTexCoord4ivARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4SARBPROC, glMultiTexCoord4sARB);
		INIT_PROC_POINTER(MYPFNGLMULTITEXCOORD4SVARBPROC, glMultiTexCoord4svARB);
		return true;
	}
#endif // GL_ARB_multitexture

#ifdef GL_SGI_texture_color_table
	else if (0 == strcmp(extension, "GL_SGI_texture_color_table")) {
		return true;
	}
#endif // GL_SGI_texture_color_table

#ifdef GL_SGI_color_table
	else if (0 == strcmp(extension, "GL_SGI_color_table")) {
		INIT_PROC_POINTER(MYPFNGLCOLORTABLESGIPROC, glColorTableSGI);
		INIT_PROC_POINTER(MYPFNGLCOLORTABLEPARAMETERFVSGIPROC, glColorTableParameterfvSGI);
		INIT_PROC_POINTER(MYPFNGLCOLORTABLEPARAMETERIVSGIPROC, glColorTableParameterivSGI);
		INIT_PROC_POINTER(MYPFNGLCOPYCOLORTABLESGIPROC, glCopyColorTableSGI);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLESGIPROC, glGetColorTableSGI);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEPARAMETERFVSGIPROC, glGetColorTableParameterfvSGI);
		INIT_PROC_POINTER(MYPFNGLGETCOLORTABLEPARAMETERIVSGIPROC, glGetColorTableParameterivSGI);
		return true;
	}
#endif // GL_SGI_color_table

#ifdef GL_SGIS_texture_edge_clamp
	else if (0 == strcmp(extension, "GL_SGIS_texture_edge_clamp")) {
		return true;
	}
#endif // GL_SGIS_texture_edge_clamp

#ifdef GL_EXT_texture3D
	else if (0 == strcmp(extension, "GL_EXT_texture3D")) {
		INIT_PROC_POINTER(MYPFNGLTEXIMAGE3DEXTPROC, glTexImage3DEXT);
		INIT_PROC_POINTER(MYPFNGLTEXSUBIMAGE3DEXTPROC, glTexSubImage3DEXT);
		return true;
	}
#endif // GL_EXT_texture3D

#ifdef GL_NV_fragment_program
	else if (0 == strcmp(extension, "GL_NV_fragment_program")) {
		INIT_PROC_POINTER(MYPFNGLPROGRAMNAMEDPARAMETER4FNVPROC, glProgramNamedParameter4fNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMNAMEDPARAMETER4DNVPROC, glProgramNamedParameter4dNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMNAMEDPARAMETER4FVNVPROC, glProgramNamedParameter4fvNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMNAMEDPARAMETER4DVNVPROC, glProgramNamedParameter4dvNV);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC, glGetProgramNamedParameterfvNV);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC, glGetProgramNamedParameterdvNV);
		return true;
	}
#endif // GL_NV_fragment_program

#ifdef GL_NV_vertex_program
	else if (0 == strcmp(extension, "GL_NV_vertex_program")) {
		INIT_PROC_POINTER(MYPFNGLAREPROGRAMSRESIDENTNVPROC, glAreProgramsResidentNV);
		INIT_PROC_POINTER(MYPFNGLBINDPROGRAMNVPROC, glBindProgramNV);
		INIT_PROC_POINTER(MYPFNGLDELETEPROGRAMSNVPROC, glDeleteProgramsNV);
		INIT_PROC_POINTER(MYPFNGLEXECUTEPROGRAMNVPROC, glExecuteProgramNV);
		INIT_PROC_POINTER(MYPFNGLGENPROGRAMSNVPROC, glGenProgramsNV);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMPARAMETERDVNVPROC, glGetProgramParameterdvNV);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMPARAMETERFVNVPROC, glGetProgramParameterfvNV);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMIVNVPROC, glGetProgramivNV);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMSTRINGNVPROC, glGetProgramStringNV);
		INIT_PROC_POINTER(MYPFNGLGETTRACKMATRIXIVNVPROC, glGetTrackMatrixivNV);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBDVNVPROC, glGetVertexAttribdvNV);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBFVNVPROC, glGetVertexAttribfvNV);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBIVNVPROC, glGetVertexAttribivNV);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBPOINTERVNVPROC, glGetVertexAttribPointervNV);
		INIT_PROC_POINTER(MYPFNGLISPROGRAMNVPROC, glIsProgramNV);
		INIT_PROC_POINTER(MYPFNGLLOADPROGRAMNVPROC, glLoadProgramNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMPARAMETER4DNVPROC, glProgramParameter4dNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMPARAMETER4DVNVPROC, glProgramParameter4dvNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMPARAMETER4FNVPROC, glProgramParameter4fNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMPARAMETER4FVNVPROC, glProgramParameter4fvNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMPARAMETERS4DVNVPROC, glProgramParameters4dvNV);
		INIT_PROC_POINTER(MYPFNGLPROGRAMPARAMETERS4FVNVPROC, glProgramParameters4fvNV);
		INIT_PROC_POINTER(MYPFNGLREQUESTRESIDENTPROGRAMSNVPROC, glRequestResidentProgramsNV);
		INIT_PROC_POINTER(MYPFNGLTRACKMATRIXNVPROC, glTrackMatrixNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBPOINTERNVPROC, glVertexAttribPointerNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1DNVPROC, glVertexAttrib1dNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1DVNVPROC, glVertexAttrib1dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1FNVPROC, glVertexAttrib1fNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1FVNVPROC, glVertexAttrib1fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1SNVPROC, glVertexAttrib1sNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1SVNVPROC, glVertexAttrib1svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2DNVPROC, glVertexAttrib2dNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2DVNVPROC, glVertexAttrib2dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2FNVPROC, glVertexAttrib2fNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2FVNVPROC, glVertexAttrib2fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2SNVPROC, glVertexAttrib2sNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2SVNVPROC, glVertexAttrib2svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3DNVPROC, glVertexAttrib3dNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3DVNVPROC, glVertexAttrib3dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3FNVPROC, glVertexAttrib3fNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3FVNVPROC, glVertexAttrib3fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3SNVPROC, glVertexAttrib3sNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3SVNVPROC, glVertexAttrib3svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4DNVPROC, glVertexAttrib4dNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4DVNVPROC, glVertexAttrib4dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4FNVPROC, glVertexAttrib4fNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4FVNVPROC, glVertexAttrib4fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4SNVPROC, glVertexAttrib4sNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4SVNVPROC, glVertexAttrib4svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4UBNVPROC, glVertexAttrib4ubNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4UBVNVPROC, glVertexAttrib4ubvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS1DVNVPROC, glVertexAttribs1dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS1FVNVPROC, glVertexAttribs1fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS1SVNVPROC, glVertexAttribs1svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS2DVNVPROC, glVertexAttribs2dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS2FVNVPROC, glVertexAttribs2fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS2SVNVPROC, glVertexAttribs2svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS3DVNVPROC, glVertexAttribs3dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS3FVNVPROC, glVertexAttribs3fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS3SVNVPROC, glVertexAttribs3svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS4DVNVPROC, glVertexAttribs4dvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS4FVNVPROC, glVertexAttribs4fvNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS4SVNVPROC, glVertexAttribs4svNV);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBS4UBVNVPROC, glVertexAttribs4ubvNV);
		return true;
	}
#endif // GL_NV_vertex_program

#ifdef GL_ARB_vertex_program
	else if (0 == strcmp(extension, "GL_ARB_vertex_program")) {
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1DARBPROC, glVertexAttrib1dARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1DVARBPROC, glVertexAttrib1dvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1FARBPROC, glVertexAttrib1fARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1FVARBPROC, glVertexAttrib1fvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1SARBPROC, glVertexAttrib1sARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB1SVARBPROC, glVertexAttrib1svARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2DARBPROC, glVertexAttrib2dARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2DVARBPROC, glVertexAttrib2dvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2FARBPROC, glVertexAttrib2fARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2FVARBPROC, glVertexAttrib2fvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2SARBPROC, glVertexAttrib2sARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB2SVARBPROC, glVertexAttrib2svARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3DARBPROC, glVertexAttrib3dARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3DVARBPROC, glVertexAttrib3dvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3FARBPROC, glVertexAttrib3fARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3FVARBPROC, glVertexAttrib3fvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3SARBPROC, glVertexAttrib3sARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB3SVARBPROC, glVertexAttrib3svARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4NBVARBPROC, glVertexAttrib4NbvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4NIVARBPROC, glVertexAttrib4NivARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4NSVARBPROC, glVertexAttrib4NsvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4NUBARBPROC, glVertexAttrib4NubARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4NUBVARBPROC, glVertexAttrib4NubvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4NUIVARBPROC, glVertexAttrib4NuivARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4NUSVARBPROC, glVertexAttrib4NusvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4BVARBPROC, glVertexAttrib4bvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4DARBPROC, glVertexAttrib4dARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4DVARBPROC, glVertexAttrib4dvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4FARBPROC, glVertexAttrib4fARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4FVARBPROC, glVertexAttrib4fvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4IVARBPROC, glVertexAttrib4ivARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4SARBPROC, glVertexAttrib4sARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4SVARBPROC, glVertexAttrib4svARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4UBVARBPROC, glVertexAttrib4ubvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4UIVARBPROC, glVertexAttrib4uivARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIB4USVARBPROC, glVertexAttrib4usvARB);
		INIT_PROC_POINTER(MYPFNGLVERTEXATTRIBPOINTERARBPROC, glVertexAttribPointerARB);
		INIT_PROC_POINTER(MYPFNGLENABLEVERTEXATTRIBARRAYARBPROC, glEnableVertexAttribArrayARB);
		INIT_PROC_POINTER(MYPFNGLDISABLEVERTEXATTRIBARRAYARBPROC, glDisableVertexAttribArrayARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMSTRINGARBPROC, glProgramStringARB);
		INIT_PROC_POINTER(MYPFNGLBINDPROGRAMARBPROC, glBindProgramARB);
		INIT_PROC_POINTER(MYPFNGLDELETEPROGRAMSARBPROC, glDeleteProgramsARB);
		INIT_PROC_POINTER(MYPFNGLGENPROGRAMSARBPROC, glGenProgramsARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMENVPARAMETER4DARBPROC, glProgramEnvParameter4dARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMENVPARAMETER4DVARBPROC, glProgramEnvParameter4dvARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMENVPARAMETER4FARBPROC, glProgramEnvParameter4fARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMENVPARAMETER4FVARBPROC, glProgramEnvParameter4fvARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMLOCALPARAMETER4DARBPROC, glProgramLocalParameter4dARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMLOCALPARAMETER4DVARBPROC, glProgramLocalParameter4dvARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMLOCALPARAMETER4FARBPROC, glProgramLocalParameter4fARB);
		INIT_PROC_POINTER(MYPFNGLPROGRAMLOCALPARAMETER4FVARBPROC, glProgramLocalParameter4fvARB);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMENVPARAMETERDVARBPROC, glGetProgramEnvParameterdvARB);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMENVPARAMETERFVARBPROC, glGetProgramEnvParameterfvARB);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMLOCALPARAMETERDVARBPROC, glGetProgramLocalParameterdvARB);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMLOCALPARAMETERFVARBPROC, glGetProgramLocalParameterfvARB);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMIVARBPROC, glGetProgramivARB);
		INIT_PROC_POINTER(MYPFNGLGETPROGRAMSTRINGARBPROC, glGetProgramStringARB);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBDVARBPROC, glGetVertexAttribdvARB);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBFVARBPROC, glGetVertexAttribfvARB);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBIVARBPROC, glGetVertexAttribivARB);
		INIT_PROC_POINTER(MYPFNGLGETVERTEXATTRIBPOINTERVARBPROC, glGetVertexAttribPointervARB);
		INIT_PROC_POINTER(MYPFNGLISPROGRAMARBPROC, glIsProgramARB);
		return true;
	}
#endif // GL_ARB_vertex_program

#ifdef GL_ARB_fragment_program
	else if (0 == strcmp(extension, "GL_ARB_fragment_program")) {
		return true;
	}
#endif // GL_ARB_fragment_program


	else {
		return false;
	}
}

void MyExtensions::initPointersToZero()
{
	// OpenGL version 1.2
	glBlendColor = 0;
	glBlendEquation = 0;
	glDrawRangeElements = 0;
	glColorTable = 0;
	glColorTableParameterfv = 0;
	glColorTableParameteriv = 0;
	glCopyColorTable = 0;
	glGetColorTable = 0;
	glGetColorTableParameterfv = 0;
	glGetColorTableParameteriv = 0;
	glColorSubTable = 0;
	glCopyColorSubTable = 0;
	glConvolutionFilter1D = 0;
	glConvolutionFilter2D = 0;
	glConvolutionParameterf = 0;
	glConvolutionParameterfv = 0;
	glConvolutionParameteri = 0;
	glConvolutionParameteriv = 0;
	glCopyConvolutionFilter1D = 0;
	glCopyConvolutionFilter2D = 0;
	glGetConvolutionFilter = 0;
	glGetConvolutionParameterfv = 0;
	glGetConvolutionParameteriv = 0;
	glGetSeparableFilter = 0;
	glSeparableFilter2D = 0;
	glGetHistogram = 0;
	glGetHistogramParameterfv = 0;
	glGetHistogramParameteriv = 0;
	glGetMinmax = 0;
	glGetMinmaxParameterfv = 0;
	glGetMinmaxParameteriv = 0;
	glHistogram = 0;
	glMinmax = 0;
	glResetHistogram = 0;
	glResetMinmax = 0;
	glTexImage3D = 0;
	glTexSubImage3D = 0;
	glCopyTexSubImage3D = 0;
	// End OpenGL version 1.2

	// OpenGL version 1.3
	glActiveTexture = 0;
	glClientActiveTexture = 0;
	glMultiTexCoord1d = 0;
	glMultiTexCoord1dv = 0;
	glMultiTexCoord1f = 0;
	glMultiTexCoord1fv = 0;
	glMultiTexCoord1i = 0;
	glMultiTexCoord1iv = 0;
	glMultiTexCoord1s = 0;
	glMultiTexCoord1sv = 0;
	glMultiTexCoord2d = 0;
	glMultiTexCoord2dv = 0;
	glMultiTexCoord2f = 0;
	glMultiTexCoord2fv = 0;
	glMultiTexCoord2i = 0;
	glMultiTexCoord2iv = 0;
	glMultiTexCoord2s = 0;
	glMultiTexCoord2sv = 0;
	glMultiTexCoord3d = 0;
	glMultiTexCoord3dv = 0;
	glMultiTexCoord3f = 0;
	glMultiTexCoord3fv = 0;
	glMultiTexCoord3i = 0;
	glMultiTexCoord3iv = 0;
	glMultiTexCoord3s = 0;
	glMultiTexCoord3sv = 0;
	glMultiTexCoord4d = 0;
	glMultiTexCoord4dv = 0;
	glMultiTexCoord4f = 0;
	glMultiTexCoord4fv = 0;
	glMultiTexCoord4i = 0;
	glMultiTexCoord4iv = 0;
	glMultiTexCoord4s = 0;
	glMultiTexCoord4sv = 0;
	glLoadTransposeMatrixf = 0;
	glLoadTransposeMatrixd = 0;
	glMultTransposeMatrixf = 0;
	glMultTransposeMatrixd = 0;
	glSampleCoverage = 0;
	glCompressedTexImage3D = 0;
	glCompressedTexImage2D = 0;
	glCompressedTexImage1D = 0;
	glCompressedTexSubImage3D = 0;
	glCompressedTexSubImage2D = 0;
	glCompressedTexSubImage1D = 0;
	glGetCompressedTexImage = 0;
	// End OpenGL version 1.3

	// OpenGL version 1.4
	glBlendFuncSeparate = 0;
	glFogCoordf = 0;
	glFogCoordfv = 0;
	glFogCoordd = 0;
	glFogCoorddv = 0;
	glFogCoordPointer = 0;
	glMultiDrawArrays = 0;
	glMultiDrawElements = 0;
	glPointParameterf = 0;
	glPointParameterfv = 0;
	glPointParameteri = 0;
	glPointParameteriv = 0;
	glSecondaryColor3b = 0;
	glSecondaryColor3bv = 0;
	glSecondaryColor3d = 0;
	glSecondaryColor3dv = 0;
	glSecondaryColor3f = 0;
	glSecondaryColor3fv = 0;
	glSecondaryColor3i = 0;
	glSecondaryColor3iv = 0;
	glSecondaryColor3s = 0;
	glSecondaryColor3sv = 0;
	glSecondaryColor3ub = 0;
	glSecondaryColor3ubv = 0;
	glSecondaryColor3ui = 0;
	glSecondaryColor3uiv = 0;
	glSecondaryColor3us = 0;
	glSecondaryColor3usv = 0;
	glSecondaryColorPointer = 0;
	glWindowPos2d = 0;
	glWindowPos2dv = 0;
	glWindowPos2f = 0;
	glWindowPos2fv = 0;
	glWindowPos2i = 0;
	glWindowPos2iv = 0;
	glWindowPos2s = 0;
	glWindowPos2sv = 0;
	glWindowPos3d = 0;
	glWindowPos3dv = 0;
	glWindowPos3f = 0;
	glWindowPos3fv = 0;
	glWindowPos3i = 0;
	glWindowPos3iv = 0;
	glWindowPos3s = 0;
	glWindowPos3sv = 0;
	// End OpenGL version 1.4

	// GL_EXT_paletted_texture
	glColorTableEXT = 0;
	glGetColorTableEXT = 0;
	glGetColorTableParameterivEXT = 0;
	glGetColorTableParameterfvEXT = 0;
	// End GL_EXT_paletted_texture

	// GL_ARB_multitexture
	glActiveTextureARB = 0;
	glClientActiveTextureARB = 0;
	glMultiTexCoord1dARB = 0;
	glMultiTexCoord1dvARB = 0;
	glMultiTexCoord1fARB = 0;
	glMultiTexCoord1fvARB = 0;
	glMultiTexCoord1iARB = 0;
	glMultiTexCoord1ivARB = 0;
	glMultiTexCoord1sARB = 0;
	glMultiTexCoord1svARB = 0;
	glMultiTexCoord2dARB = 0;
	glMultiTexCoord2dvARB = 0;
	glMultiTexCoord2fARB = 0;
	glMultiTexCoord2fvARB = 0;
	glMultiTexCoord2iARB = 0;
	glMultiTexCoord2ivARB = 0;
	glMultiTexCoord2sARB = 0;
	glMultiTexCoord2svARB = 0;
	glMultiTexCoord3dARB = 0;
	glMultiTexCoord3dvARB = 0;
	glMultiTexCoord3fARB = 0;
	glMultiTexCoord3fvARB = 0;
	glMultiTexCoord3iARB = 0;
	glMultiTexCoord3ivARB = 0;
	glMultiTexCoord3sARB = 0;
	glMultiTexCoord3svARB = 0;
	glMultiTexCoord4dARB = 0;
	glMultiTexCoord4dvARB = 0;
	glMultiTexCoord4fARB = 0;
	glMultiTexCoord4fvARB = 0;
	glMultiTexCoord4iARB = 0;
	glMultiTexCoord4ivARB = 0;
	glMultiTexCoord4sARB = 0;
	glMultiTexCoord4svARB = 0;
	// End GL_ARB_multitexture

	// GL_NV_fragment_program
	glProgramNamedParameter4fNV = 0;
	glProgramNamedParameter4dNV = 0;
	glProgramNamedParameter4fvNV = 0;
	glProgramNamedParameter4dvNV = 0;
	glGetProgramNamedParameterfvNV = 0;
	glGetProgramNamedParameterdvNV = 0;
	// End GL_NV_fragment_program

	// GL_NV_vertex_program
	glAreProgramsResidentNV = 0;
	glBindProgramNV = 0;
	glDeleteProgramsNV = 0;
	glExecuteProgramNV = 0;
	glGenProgramsNV = 0;
	glGetProgramParameterdvNV = 0;
	glGetProgramParameterfvNV = 0;
	glGetProgramivNV = 0;
	glGetProgramStringNV = 0;
	glGetTrackMatrixivNV = 0;
	glGetVertexAttribdvNV = 0;
	glGetVertexAttribfvNV = 0;
	glGetVertexAttribivNV = 0;
	glGetVertexAttribPointervNV = 0;
	glIsProgramNV = 0;
	glLoadProgramNV = 0;
	glProgramParameter4dNV = 0;
	glProgramParameter4dvNV = 0;
	glProgramParameter4fNV = 0;
	glProgramParameter4fvNV = 0;
	glProgramParameters4dvNV = 0;
	glProgramParameters4fvNV = 0;
	glRequestResidentProgramsNV = 0;
	glTrackMatrixNV = 0;
	glVertexAttribPointerNV = 0;
	glVertexAttrib1dNV = 0;
	glVertexAttrib1dvNV = 0;
	glVertexAttrib1fNV = 0;
	glVertexAttrib1fvNV = 0;
	glVertexAttrib1sNV = 0;
	glVertexAttrib1svNV = 0;
	glVertexAttrib2dNV = 0;
	glVertexAttrib2dvNV = 0;
	glVertexAttrib2fNV = 0;
	glVertexAttrib2fvNV = 0;
	glVertexAttrib2sNV = 0;
	glVertexAttrib2svNV = 0;
	glVertexAttrib3dNV = 0;
	glVertexAttrib3dvNV = 0;
	glVertexAttrib3fNV = 0;
	glVertexAttrib3fvNV = 0;
	glVertexAttrib3sNV = 0;
	glVertexAttrib3svNV = 0;
	glVertexAttrib4dNV = 0;
	glVertexAttrib4dvNV = 0;
	glVertexAttrib4fNV = 0;
	glVertexAttrib4fvNV = 0;
	glVertexAttrib4sNV = 0;
	glVertexAttrib4svNV = 0;
	glVertexAttrib4ubNV = 0;
	glVertexAttrib4ubvNV = 0;
	glVertexAttribs1dvNV = 0;
	glVertexAttribs1fvNV = 0;
	glVertexAttribs1svNV = 0;
	glVertexAttribs2dvNV = 0;
	glVertexAttribs2fvNV = 0;
	glVertexAttribs2svNV = 0;
	glVertexAttribs3dvNV = 0;
	glVertexAttribs3fvNV = 0;
	glVertexAttribs3svNV = 0;
	glVertexAttribs4dvNV = 0;
	glVertexAttribs4fvNV = 0;
	glVertexAttribs4svNV = 0;
	glVertexAttribs4ubvNV = 0;
	// End GL_NV_vertex_program

	// GL_ARB_vertex_program
	glVertexAttrib1dARB = 0;
	glVertexAttrib1dvARB = 0;
	glVertexAttrib1fARB = 0;
	glVertexAttrib1fvARB = 0;
	glVertexAttrib1sARB = 0;
	glVertexAttrib1svARB = 0;
	glVertexAttrib2dARB = 0;
	glVertexAttrib2dvARB = 0;
	glVertexAttrib2fARB = 0;
	glVertexAttrib2fvARB = 0;
	glVertexAttrib2sARB = 0;
	glVertexAttrib2svARB = 0;
	glVertexAttrib3dARB = 0;
	glVertexAttrib3dvARB = 0;
	glVertexAttrib3fARB = 0;
	glVertexAttrib3fvARB = 0;
	glVertexAttrib3sARB = 0;
	glVertexAttrib3svARB = 0;
	glVertexAttrib4NbvARB = 0;
	glVertexAttrib4NivARB = 0;
	glVertexAttrib4NsvARB = 0;
	glVertexAttrib4NubARB = 0;
	glVertexAttrib4NubvARB = 0;
	glVertexAttrib4NuivARB = 0;
	glVertexAttrib4NusvARB = 0;
	glVertexAttrib4bvARB = 0;
	glVertexAttrib4dARB = 0;
	glVertexAttrib4dvARB = 0;
	glVertexAttrib4fARB = 0;
	glVertexAttrib4fvARB = 0;
	glVertexAttrib4ivARB = 0;
	glVertexAttrib4sARB = 0;
	glVertexAttrib4svARB = 0;
	glVertexAttrib4ubvARB = 0;
	glVertexAttrib4uivARB = 0;
	glVertexAttrib4usvARB = 0;
	glVertexAttribPointerARB = 0;
	glEnableVertexAttribArrayARB = 0;
	glDisableVertexAttribArrayARB = 0;
	glProgramStringARB = 0;
	glBindProgramARB = 0;
	glDeleteProgramsARB = 0;
	glGenProgramsARB = 0;
	glProgramEnvParameter4dARB = 0;
	glProgramEnvParameter4dvARB = 0;
	glProgramEnvParameter4fARB = 0;
	glProgramEnvParameter4fvARB = 0;
	glProgramLocalParameter4dARB = 0;
	glProgramLocalParameter4dvARB = 0;
	glProgramLocalParameter4fARB = 0;
	glProgramLocalParameter4fvARB = 0;
	glGetProgramEnvParameterdvARB = 0;
	glGetProgramEnvParameterfvARB = 0;
	glGetProgramLocalParameterdvARB = 0;
	glGetProgramLocalParameterfvARB = 0;
	glGetProgramivARB = 0;
	glGetProgramStringARB = 0;
	glGetVertexAttribdvARB = 0;
	glGetVertexAttribfvARB = 0;
	glGetVertexAttribivARB = 0;
	glGetVertexAttribPointervARB = 0;
	glIsProgramARB = 0;
	// End GL_ARB_vertex_program

	// GL_ARB_fragment_program
	// No new functions.
	// End GL_ARB_fragment_program
}

const char* MyExtensions::EatWhiteSpace(const char* str)
{
    for (; *str && (' ' == *str || '\t' == *str || '\n' == *str); str++);
    return str;
}

const char* MyExtensions::EatNonWhiteSpace(const char* str)
{
    for (; *str && (' ' != *str && '\t' != *str && '\n' != *str); str++);
    return str;
}

const char* MyExtensions::getExtensionStringPrivate()
{
	const char* normalExtensions = (const char*)glGetString(GL_EXTENSIONS);
	return normalExtensions;
}

const char* MyExtensions::getSystemExtensions()
{
	// for now, we only do wgl extensions, not glx extensions
//#if defined(_WIN32)
#if 0
	MYPFNWGLGETEXTENSIONSSTRINGARBPROC mywglGetExtensionsStringARB = 0;
	mywglGetExtensionsStringARB = (MYPFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
	if(mywglGetExtensionsStringARB)
	{
		return mywglGetExtensionsStringARB(wglGetCurrentDC());
	}
	else {
		return 0;
	}
#else
	return 0;
#endif
}

