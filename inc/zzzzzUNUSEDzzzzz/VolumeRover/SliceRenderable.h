/******************************************************************************

        Authors: Jose Rivera <transfix@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

/* $Id: SliceRenderable.h 1527 2010-03-12 22:10:16Z transfix $ */

#ifndef __VOLUME__SLICERENDERABLE_H__
#define __VOLUME__SLICERENDERABLE_H__

#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <VolumeWidget/Renderable.h>
#include <VolumeFileTypes/VolumeBufferManager.h>
#include <VolumeGridRover/VolumeGridRover.h>
#include <VolumeGridRover/SurfRecon.h>
#include <glew/glew.h>

class VolumeBufferManager;

class SliceRenderable : public Renderable
{
public:
  enum SliceAxis { XY, XZ, ZY };
  
  SliceRenderable();
  virtual ~SliceRenderable();
  
  void setVolume(const VolMagick::VolumeFileInfo& vfi);
  void unsetVolume();
  
  void setDrawSlice(SliceAxis a, bool b);
  bool drawSlice(SliceAxis a);
  
  void setDepth(SliceAxis a, unsigned int d);
  void setGrayscale(SliceAxis a, bool set);
  void setCurrentVariable(unsigned int var);
  void setCurrentTimestep(unsigned int time);
  unsigned int currentVariable() const { return m_XYSlice.currentVariable(); }
  unsigned int currentTimestep() const { return m_XYSlice.currentTimestep(); }
  
  virtual bool render();
  
  bool isGrayscale(SliceAxis a);
  
  void setSecondarySliceOffset(SliceAxis a, int o);
  void setDrawSecondarySlice(SliceAxis a, bool b);
  bool drawSecondarySlice(SliceAxis a);
  int secondarySliceOffset(SliceAxis a);
  
  void setColorTable(unsigned char *palette);

  void setDraw2DContours(SliceAxis a, bool b);
  bool draw2DContours(SliceAxis a);
  void set2DContours(const SurfRecon::ContourPtrArray* p);
  
  void setSubVolume(const VolMagick::BoundingBox& subvolbox);

  void setClipGeometry(bool b);

 protected:
  class SliceRenderer;
  class Slice
  {
  public:
    Slice(SliceAxis a);
    ~Slice();
    
    void setVolume(const VolMagick::VolumeFileInfo& vfi);
    void unsetVolume();
    
    void setDepth(unsigned int d);
    void setSecondarySliceOffset(int o);
    void setCurrentVariable(unsigned int var);
    void setCurrentTimestep(unsigned int time);
    unsigned int currentVariable() const { return m_Variable; }
    unsigned int currentTimestep() const { return m_Timestep; }
    unsigned int depth() const { return m_Depth; }
    int secondarySliceOffset() const { return m_SecondarySliceOffset; }
    void setDrawSlice(bool b);
    bool drawSlice() const { return m_DrawSlice; }
    void setDrawSecondarySlice(bool b);
    bool drawSecondarySlice() const { return m_DrawSecondarySlice; }
    
    bool render();
    
    void setGrayscale(bool set);
    bool isGrayscale() const { return m_Palette == m_GrayMap; }
    
    SliceAxis sliceAxis() const { return m_SliceAxis; }

    void setColorTable(unsigned char *palette);
    
    void setDraw2DContours(bool b);
    bool draw2DContours() const { return m_Draw2DContours; }
    void set2DContours(const SurfRecon::ContourPtrArray* p) { m_Contours = p; }

    void setSubVolume(const VolMagick::BoundingBox& subvolbox);
    const VolMagick::BoundingBox& subVolume() const { return m_SubVolumeBoundingBox; }

    bool clipGeometry() const { return m_ClipGeometry; }
    void setClipGeometry(bool b) { m_ClipGeometry = b; }

  private:
    bool initRenderer();
    void updateSliceBuffers();
    void updateColorTable();
    void updateTextureCoordinates();
    void updateVertexCoordinates();
    void doDraw2DContours();
    void setClipPlanes();
    void disableClipPlanes();
        
    SliceAxis m_SliceAxis;
    
    bool m_Drawable;
    
    unsigned int m_Depth;
    unsigned int m_Variable;
    unsigned int m_Timestep;
    int m_SecondarySliceOffset;
    
    bool m_DrawSlice;
    bool m_DrawSecondarySlice;
    
    unsigned char m_ByteMap[256*4]; /* The transfer function */
    unsigned char m_GrayMap[256*4]; /* a grayscale map */
    unsigned char *m_Palette; /* a pointer to one of the above maps */
    VolMagick::VolumeFileInfo m_VolumeFileInfo; // entire volume info
    VolMagick::Volume m_VolumeSlice; // contains current slice data
    VolMagick::Volume m_SecondaryVolumeSlice; // contains the current secondary slice data
    VolMagick::BoundingBox m_SubVolumeBoundingBox; // defines the subvolume from which to extract slices
    
    float m_TexCoords[8];
    float m_VertCoords[12];
    float m_SecondaryVertCoords[12];
    
    //boost::shared_array<unsigned char> m_Slice; // the slice as a unsigned char uploadable texture
    boost::scoped_ptr<SliceRenderer> m_SliceRenderer;

    bool m_InitRenderer; //if this is true when calling render(), call initRenderer()
    bool m_UpdateSlice; //if this is true when calling render(), call updateSliceBuffers()
    bool m_UpdateColorTable; //if this is true when calling render(), call updateColorTable()

    bool m_Draw2DContours;
    const SurfRecon::ContourPtrArray* m_Contours; //a pointer to VGR's 2D contour set

    bool m_ClipGeometry;
  };

  class SliceRenderer
  {
  public:
    SliceRenderer() {}
    virtual ~SliceRenderer() {}
    
    virtual bool init() = 0;
    virtual void draw(float *vert_coords, float *tex_coords) = 0;
    virtual void drawSecondary(float *vert_coords, float *tex_coords) = 0;
    virtual void uploadSlice(unsigned char *, unsigned int, unsigned int) = 0;
    virtual void uploadSecondarySlice(unsigned char *, unsigned int, unsigned int) = 0;
    virtual void uploadColorTable(unsigned char *) = 0;
  };
  
  class ARBFragmentProgramSliceRenderer : public SliceRenderer
  {
  public:
    ARBFragmentProgramSliceRenderer() 
      : m_FragmentProgram(0), m_PaletteTexture(0), m_SliceTexture(0), m_SecondarySliceTexture(0) {}
    virtual ~ARBFragmentProgramSliceRenderer() {}
    
    virtual bool init();
    virtual void draw(float *vert_coords, float *tex_coords);
    virtual void drawSecondary(float *vert_coords, float *tex_coords);
    virtual void uploadSlice(unsigned char *slice, unsigned int dimx, unsigned int dimy);
    virtual void uploadSecondarySlice(unsigned char *slice, unsigned int dimx, unsigned int dimy);
    virtual void uploadColorTable(unsigned char *palette);
    
    GLuint m_FragmentProgram;
    GLuint m_PaletteTexture; /* texture id for the current palette */
    GLuint m_SliceTexture; /* texture id for the slice texture */
    GLuint m_SecondarySliceTexture; /* texture id for the secondary slice texture */
  };

  Slice m_XYSlice, m_XZSlice, m_ZYSlice;
};

#endif
