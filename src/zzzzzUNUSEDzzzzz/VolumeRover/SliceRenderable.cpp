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

#ifdef VOLUMEGRIDROVER

#include <boost/array.hpp>

// using GSL library for 2D contour interpolation
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include <VolumeRover/SliceRenderable.h>

static inline void mv_mult(double m[16], double vin[3], double vout[4])
{
  vout[0] = vin[0]*m[0] + vin[1]*m[4] + vin[2]*m[8] + m[12];
  vout[1] = vin[0]*m[1] + vin[1]*m[5] + vin[2]*m[9] + m[13];
  vout[2] = vin[0]*m[2] + vin[1]*m[6] + vin[2]*m[10] + m[14];
  vout[3] = vin[0]*m[3] + vin[1]*m[7] + vin[2]*m[11] + m[15];
}

static inline unsigned int upToPowerOfTwo(unsigned int value)
{
  unsigned int c = 0;
  unsigned int v = value;

  // round down to nearest power of two 
  while (v>1) {
  v = v>>1;
  c++;
  }

  // if that isn't exactly the original value 
  if ((v<<c)!=value) { 
  // return the next power of two 
  return (v<<(c+1));
  }
  else {
  // return this power of two 
  return (v<<c);
  }
}

SliceRenderable::SliceRenderable()
  : m_XYSlice(XY), m_XZSlice(XZ), m_ZYSlice(ZY)
{
}
 
SliceRenderable::~SliceRenderable() {}

void SliceRenderable::setVolume(const VolMagick::VolumeFileInfo& vfi)
{
  m_XYSlice.setVolume(vfi);
  m_XZSlice.setVolume(vfi);
  m_ZYSlice.setVolume(vfi);
}

void SliceRenderable::unsetVolume()
{
  m_XYSlice.unsetVolume();
  m_XZSlice.unsetVolume();
  m_ZYSlice.unsetVolume();
}

void SliceRenderable::setDrawSlice(SliceRenderable::SliceAxis a, bool b)
{
  switch(a)
    {
    case XY: m_XYSlice.setDrawSlice(b); break;
    case XZ: m_XZSlice.setDrawSlice(b); break;
    case ZY: m_ZYSlice.setDrawSlice(b); break;
    default: break;
    }
}

bool SliceRenderable::drawSlice(SliceRenderable::SliceAxis a)
{
  switch(a)
    {
    case XY: return m_XYSlice.drawSlice(); break;
    case XZ: return m_XZSlice.drawSlice(); break;
    case ZY: return m_ZYSlice.drawSlice(); break;
    default: break;
    }

  return false;
}

void SliceRenderable::setDepth(SliceRenderable::SliceAxis a, unsigned int d)
{
  switch(a)
    {
    case XY: m_XYSlice.setDepth(d); break;
    case XZ: m_XZSlice.setDepth(d); break;
    case ZY: m_ZYSlice.setDepth(d); break;
    default: break;
    }
}

void SliceRenderable::setGrayscale(SliceRenderable::SliceAxis a, bool set)
{
  switch(a)
    {
    case XY: m_XYSlice.setGrayscale(set); break;
    case XZ: m_XZSlice.setGrayscale(set); break;
    case ZY: m_ZYSlice.setGrayscale(set); break;
    default: break;
    }
}

bool SliceRenderable::isGrayscale(SliceRenderable::SliceAxis a)
{
  switch(a)
    {
    case XY: return m_XYSlice.isGrayscale(); break;
    case XZ: return m_XZSlice.isGrayscale(); break;
    case ZY: return m_ZYSlice.isGrayscale(); break;
    default: break;
    }

  return false;
}

void SliceRenderable::setCurrentVariable(unsigned int var)
{
  m_XYSlice.setCurrentVariable(var);
  m_XZSlice.setCurrentVariable(var);
  m_ZYSlice.setCurrentVariable(var);
}

void SliceRenderable::setCurrentTimestep(unsigned int var)
{
  m_XYSlice.setCurrentTimestep(var);
  m_XZSlice.setCurrentTimestep(var);
  m_ZYSlice.setCurrentTimestep(var);
}

bool SliceRenderable::render()
{
  m_XYSlice.render();
  m_XZSlice.render();
  m_ZYSlice.render();

  return true;
}

void SliceRenderable::setSecondarySliceOffset(SliceAxis a, int o)
{
  switch(a)
    {
    case XY: m_XYSlice.setSecondarySliceOffset(o); break;
    case XZ: m_XZSlice.setSecondarySliceOffset(o); break;
    case ZY: m_ZYSlice.setSecondarySliceOffset(o); break;
    default: break;
    }
}

void SliceRenderable::setDrawSecondarySlice(SliceAxis a, bool b)
{
  switch(a)
    {
    case XY: m_XYSlice.setDrawSecondarySlice(b); break;
    case XZ: m_XZSlice.setDrawSecondarySlice(b); break;
    case ZY: m_ZYSlice.setDrawSecondarySlice(b); break;
    default: break;
    }
}

bool SliceRenderable::drawSecondarySlice(SliceAxis a)
{
  switch(a)
    {
    case XY: return m_XYSlice.drawSecondarySlice(); break;
    case XZ: return m_XZSlice.drawSecondarySlice(); break;
    case ZY: return m_ZYSlice.drawSecondarySlice(); break;
    default: break;
    }

  return false;
}

int SliceRenderable::secondarySliceOffset(SliceAxis a)
{
  switch(a)
    {
    case XY: return m_XYSlice.secondarySliceOffset(); break;
    case XZ: return m_XZSlice.secondarySliceOffset(); break;
    case ZY: return m_ZYSlice.secondarySliceOffset(); break;
    default: break;
    }

  return 0;
}

void SliceRenderable::setColorTable(unsigned char *palette)
{
  m_XYSlice.setColorTable(palette);
  m_XZSlice.setColorTable(palette);
  m_ZYSlice.setColorTable(palette);  
}

void SliceRenderable::setDraw2DContours(SliceAxis a, bool b)
{
  switch(a)
    {
    case XY: m_XYSlice.setDraw2DContours(b); break;
    case XZ: m_XZSlice.setDraw2DContours(b); break;
    case ZY: m_ZYSlice.setDraw2DContours(b); break;
    default: break;
    }
}

bool SliceRenderable::draw2DContours(SliceAxis a)
{
  switch(a)
    {
    case XY: return m_XYSlice.draw2DContours();
    case XZ: return m_XZSlice.draw2DContours();
    case ZY: return m_ZYSlice.draw2DContours();
    default: return false;
    }
}

void SliceRenderable::set2DContours(const SurfRecon::ContourPtrArray* p)
{
  m_XYSlice.set2DContours(p);
  m_XZSlice.set2DContours(p);
  m_ZYSlice.set2DContours(p);
}

void SliceRenderable::setSubVolume(const VolMagick::BoundingBox& subvolbox)
{
  qDebug("SliceRenderable::setSubVolume(): (%f,%f,%f) (%f,%f,%f)",
	 subvolbox.minx,subvolbox.miny,subvolbox.minz,
	 subvolbox.maxx,subvolbox.maxy,subvolbox.maxz);

  m_XYSlice.setSubVolume(subvolbox);
  m_XZSlice.setSubVolume(subvolbox);
  m_ZYSlice.setSubVolume(subvolbox);
}

void SliceRenderable::setClipGeometry(bool b)
{
  m_XYSlice.setClipGeometry(b);
  m_XZSlice.setClipGeometry(b);
  m_ZYSlice.setClipGeometry(b);
}

SliceRenderable::Slice::Slice(SliceAxis a)
  : m_SliceAxis(a), m_Drawable(false),
    m_Depth(0), m_Variable(0), m_Timestep(0),
    m_SecondarySliceOffset(0), m_DrawSlice(false), m_DrawSecondarySlice(false),
    m_InitRenderer(true), m_UpdateSlice(true), m_UpdateColorTable(true),
    m_Draw2DContours(false), m_Contours(NULL), m_ClipGeometry(true)
{
  /* initialize the color maps */
  m_Palette = m_ByteMap;
  memset(m_ByteMap,0,256*4);
  for(int i=0; i<256; i++)
    {
      m_GrayMap[i*4+0] = i;
      m_GrayMap[i*4+1] = i;
      m_GrayMap[i*4+2] = i;
      m_GrayMap[i*4+3] = 255;
    }
}

SliceRenderable::Slice::~Slice()
{
}

void SliceRenderable::Slice::setVolume(const VolMagick::VolumeFileInfo& vfi)
{
  m_VolumeFileInfo = vfi;
  setDepth(0);
  setCurrentVariable(0);
  setCurrentTimestep(0);
  setSecondarySliceOffset(0);
  setSubVolume(vfi.boundingBox());
}

void SliceRenderable::Slice::unsetVolume()
{
  m_VolumeFileInfo = VolMagick::VolumeFileInfo();
  m_UpdateSlice = true;
}

void SliceRenderable::Slice::setDepth(unsigned int d)
{
  //m_Depth = d;
  m_Depth = std::min((unsigned int)((m_SliceAxis == XY ? m_VolumeFileInfo.ZDim() : 
				     m_SliceAxis == XZ ? m_VolumeFileInfo.YDim() : 
				     m_SliceAxis == ZY ? m_VolumeFileInfo.XDim() : 0)-1),d);
  m_UpdateSlice = true;
}

void SliceRenderable::Slice::setSecondarySliceOffset(int o)
{
  //make sure we're within bounds
  //i.e. 0 <= m_Depth + m_SecondarySliceOffset < m_VolumeFileInfo.[X-Z]Dim()
  int d = m_Depth;
  m_SecondarySliceOffset = std::max(-d,
				    std::min(int((m_SliceAxis == XY ? m_VolumeFileInfo.ZDim() : 
						  m_SliceAxis == XZ ? m_VolumeFileInfo.YDim() : 
						  m_SliceAxis == ZY ? m_VolumeFileInfo.XDim() : 0)-1) - int(m_Depth), o));
  m_UpdateSlice = true;
}

void SliceRenderable::Slice::setCurrentVariable(unsigned int var)
{
  m_Variable = var;
  m_UpdateSlice = true;
}

void SliceRenderable::Slice::setCurrentTimestep(unsigned int time)
{
  m_Timestep = time;
  m_UpdateSlice = true;
}

void SliceRenderable::Slice::setDrawSlice(bool b)
{
  m_DrawSlice = b;
  m_UpdateSlice = true;
}

void SliceRenderable::Slice::setDrawSecondarySlice(bool b)
{
  m_DrawSecondarySlice = b;
  m_UpdateSlice = true;
}

bool SliceRenderable::Slice::render()
{
  glDisable(GL_LIGHTING);
  if(m_InitRenderer) initRenderer();
  if(m_UpdateSlice) updateSliceBuffers();
  if(m_UpdateColorTable) updateColorTable();
  if(m_SliceRenderer && m_VolumeFileInfo.isSet()) //actually do the drawing
    {
      if(clipGeometry())
	setClipPlanes();
      
      if(m_DrawSlice) m_SliceRenderer->draw(m_VertCoords,m_TexCoords);
      if(m_DrawSecondarySlice) m_SliceRenderer->drawSecondary(m_SecondaryVertCoords,m_TexCoords);
      if(m_Draw2DContours && m_Contours) doDraw2DContours();

      if(clipGeometry())
	disableClipPlanes();
    }
  return true;
}

void SliceRenderable::Slice::setGrayscale(bool set)
{
  //if(!m_Drawable) return;

  if(set)
    m_Palette = m_GrayMap;
  else
    m_Palette = m_ByteMap;
	
  m_UpdateColorTable = true;

  //makeCurrent();

  //uploadColorTable();
  //if(m_SliceRenderer) m_SliceRenderer->uploadColorTable(m_ByteMap);

  //updateGL();
}

void SliceRenderable::Slice::setColorTable(unsigned char *palette)
{
  memcpy(m_ByteMap,palette,256*4);
  m_UpdateColorTable = true;
}

void SliceRenderable::Slice::setDraw2DContours(bool b)
{
  m_Draw2DContours = b;
}

void SliceRenderable::Slice::setSubVolume(const VolMagick::BoundingBox& subvolbox)
{
  if(subvolbox.isWithin(m_VolumeFileInfo.boundingBox()))
    m_SubVolumeBoundingBox = subvolbox;
  else
    m_SubVolumeBoundingBox = m_VolumeFileInfo.boundingBox();

  updateTextureCoordinates();
  updateVertexCoordinates();
}

bool SliceRenderable::Slice::initRenderer()
{
  m_SliceRenderer.reset(new SliceRenderable::ARBFragmentProgramSliceRenderer());
  if(m_SliceRenderer->init())
    {
      m_InitRenderer = false;
      m_Drawable = true;
      return true;
    }

  m_Drawable = false;
  m_SliceRenderer.reset();
  return false;
}

void SliceRenderable::Slice::updateSliceBuffers()
{
  unsigned int i,j,imgx,imgy,imgz;

  if(!m_VolumeFileInfo.isSet() || !m_Drawable) 
    {
      m_UpdateSlice = false;
      return;
    }

  /* uploadable buffer's dimensions must be == 2^n */
  imgx = upToPowerOfTwo(m_VolumeFileInfo.XDim());
  imgy = upToPowerOfTwo(m_VolumeFileInfo.YDim());
  imgz = upToPowerOfTwo(m_VolumeFileInfo.ZDim());

  boost::scoped_array<unsigned char> slice_data;
  boost::scoped_array<unsigned char> secondary_slice_data;

  switch(m_SliceAxis)
    {
    case XY:
      {
	m_Depth = m_Depth <= m_VolumeFileInfo.ZDim()-1 ? m_Depth : m_VolumeFileInfo.ZDim()-1;
	
	/* read a single slice into m_VolumeSlice */
	VolMagick::readVolumeFile(m_VolumeSlice,
				  m_VolumeFileInfo.filename(),
				  m_Variable,m_Timestep,
				  0,0,m_Depth,
				  VolMagick::Dimension(m_VolumeFileInfo.XDim(),m_VolumeFileInfo.YDim(),1));

	/* duplicate the slice and map it to unsigned char range */
	VolMagick::Volume mappedSlice(m_VolumeSlice);
	if(mappedSlice.voxelType() != VolMagick::UChar)
	  {
	    /* get min and max from overall file, else it will be calculated from subvolume */
	    mappedSlice.min(m_VolumeFileInfo.min(m_Variable,m_Timestep));
	    mappedSlice.max(m_VolumeFileInfo.max(m_Variable,m_Timestep));
	    mappedSlice.map(0.0,255.0);
	    mappedSlice.voxelType(VolMagick::UChar);
	  }
      
	/* now copy it to the uploadable slice buffer */
	slice_data.reset(new unsigned char[imgx*imgy]);
	for(i=0; i<m_VolumeSlice.XDim(); i++)
	  for(j=0; j<m_VolumeSlice.YDim(); j++)
	    slice_data[i + imgx*j] = mappedSlice(i,j,0);

	if(m_DrawSecondarySlice)
	  {
	    /* read a single slice into m_SecondaryVolumeSlice */
	    VolMagick::readVolumeFile(m_SecondaryVolumeSlice,
				      m_VolumeFileInfo.filename(),
				      m_Variable,m_Timestep,
				      0,0,m_Depth+m_SecondarySliceOffset,
				      VolMagick::Dimension(m_VolumeFileInfo.XDim(),m_VolumeFileInfo.YDim(),1));
	    
	    /* duplicate the slice and map it to unsigned char range */
	    VolMagick::Volume secondaryMappedSlice(m_SecondaryVolumeSlice);
	    if(secondaryMappedSlice.voxelType() != VolMagick::UChar)
	      {
		/* get min and max from overall file, else it will be calculated from subvolume */
		secondaryMappedSlice.min(m_VolumeFileInfo.min(m_Variable,m_Timestep));
		secondaryMappedSlice.max(m_VolumeFileInfo.max(m_Variable,m_Timestep));
		secondaryMappedSlice.map(0.0,255.0);
		secondaryMappedSlice.voxelType(VolMagick::UChar);
	      }
	    
	    /* now copy it to the uploadable slice buffer */
	    secondary_slice_data.reset(new unsigned char[imgx*imgy]);
	    for(i=0; i<m_VolumeSlice.XDim(); i++)
	      for(j=0; j<m_VolumeSlice.YDim(); j++)
		secondary_slice_data[i + imgx*j] = secondaryMappedSlice(i,j,0);
	  }
      }
      break;
    case XZ:
      {
	m_Depth = m_Depth <= m_VolumeFileInfo.YDim()-1 ? m_Depth : m_VolumeFileInfo.YDim()-1;

	/* read a single slice into m_VolumeSlice */
	VolMagick::readVolumeFile(m_VolumeSlice,
				  m_VolumeFileInfo.filename(),
				  m_Variable,m_Timestep,
				  0,m_Depth,0,
				  VolMagick::Dimension(m_VolumeFileInfo.XDim(),1,m_VolumeFileInfo.ZDim()));
      
	/* duplicate the slice and map it to the unsigned char range */
	VolMagick::Volume mappedSlice(m_VolumeSlice);
	if(mappedSlice.voxelType() != VolMagick::UChar)
	  {
	    /* get min and max from overall file, else it will be calculated from subvolume */
	    mappedSlice.min(m_VolumeFileInfo.min(m_Variable,m_Timestep));
	    mappedSlice.max(m_VolumeFileInfo.max(m_Variable,m_Timestep));
	    mappedSlice.map(0.0,255.0);
	    mappedSlice.voxelType(VolMagick::UChar);
	  }
      
	/* now copy it to the uploadable slice buffer */
	slice_data.reset(new unsigned char[imgx*imgz]);
	for(i=0; i<m_VolumeSlice.XDim(); i++)
	  for(j=0; j<m_VolumeSlice.ZDim(); j++)
	    slice_data[i + imgx*j] = mappedSlice(i,0,j);

	if(m_DrawSecondarySlice)
	  {
	    /* read a single slice into m_SecondaryVolumeSlice */
	    VolMagick::readVolumeFile(m_SecondaryVolumeSlice,
				      m_VolumeFileInfo.filename(),
				      m_Variable,m_Timestep,
				      0,m_Depth+m_SecondarySliceOffset,0,
				      VolMagick::Dimension(m_VolumeFileInfo.XDim(),1,m_VolumeFileInfo.ZDim()));
	    
	    /* duplicate the slice and map it to the unsigned char range */
	    VolMagick::Volume secondaryMappedSlice(m_SecondaryVolumeSlice);
	    if(secondaryMappedSlice.voxelType() != VolMagick::UChar)
	      {
		/* get min and max from overall file, else it will be calculated from subvolume */
		secondaryMappedSlice.min(m_VolumeFileInfo.min(m_Variable,m_Timestep));
		secondaryMappedSlice.max(m_VolumeFileInfo.max(m_Variable,m_Timestep));
		secondaryMappedSlice.map(0.0,255.0);
		secondaryMappedSlice.voxelType(VolMagick::UChar);
	      }
	    
	    /* now copy it to the uploadable slice buffer */
	    secondary_slice_data.reset(new unsigned char[imgx*imgz]);
	    for(i=0; i<m_VolumeSlice.XDim(); i++)
	      for(j=0; j<m_VolumeSlice.ZDim(); j++)
		secondary_slice_data[i + imgx*j] = secondaryMappedSlice(i,0,j);
	  }
      }
      break;
    case ZY:
      {
	m_Depth = m_Depth <= m_VolumeFileInfo.XDim()-1 ? m_Depth : m_VolumeFileInfo.XDim()-1;
      
	/* read a single slice into m_VolumeSlice */
	VolMagick::readVolumeFile(m_VolumeSlice,
				  m_VolumeFileInfo.filename(),
				  m_Variable,m_Timestep,
				  m_Depth,0,0,
				  VolMagick::Dimension(1,m_VolumeFileInfo.YDim(),m_VolumeFileInfo.ZDim()));
      
	/* duplicate the slice and map it to the unsigned char range */
	VolMagick::Volume mappedSlice(m_VolumeSlice);
	if(mappedSlice.voxelType() != VolMagick::UChar)
	  {
	    /* get min and max from overall file, else it will be calculated from subvolume */
	    mappedSlice.min(m_VolumeFileInfo.min(m_Variable,m_Timestep));
	    mappedSlice.max(m_VolumeFileInfo.max(m_Variable,m_Timestep));
	    mappedSlice.map(0.0,255.0);
	    mappedSlice.voxelType(VolMagick::UChar);
	  }
      
	/* now copy it to the uploadable slice buffer */
	slice_data.reset(new unsigned char[imgz*imgy]);
	for(i=0; i<m_VolumeSlice.ZDim(); i++)
	  for(j=0; j<m_VolumeSlice.YDim(); j++)
	    slice_data[i + imgz*j] = mappedSlice(0,j,i);

	if(m_DrawSecondarySlice)
	  {
	    /* read a single slice into m_SecondaryVolumeSlice */
	    VolMagick::readVolumeFile(m_SecondaryVolumeSlice,
				      m_VolumeFileInfo.filename(),
				      m_Variable,m_Timestep,
				      m_Depth+m_SecondarySliceOffset,0,0,
				      VolMagick::Dimension(1,m_VolumeFileInfo.YDim(),m_VolumeFileInfo.ZDim()));
	    
	    /* duplicate the slice and map it to the unsigned char range */
	    VolMagick::Volume secondaryMappedSlice(m_SecondaryVolumeSlice);
	    if(secondaryMappedSlice.voxelType() != VolMagick::UChar)
	      {
		/* get min and max from overall file, else it will be calculated from subvolume */
		secondaryMappedSlice.min(m_VolumeFileInfo.min(m_Variable,m_Timestep));
		secondaryMappedSlice.max(m_VolumeFileInfo.max(m_Variable,m_Timestep));
		secondaryMappedSlice.map(0.0,255.0);
		secondaryMappedSlice.voxelType(VolMagick::UChar);
	      }
	    
	    /* now copy it to the uploadable slice buffer */
	    secondary_slice_data.reset(new unsigned char[imgz*imgy]);
	    for(i=0; i<m_VolumeSlice.ZDim(); i++)
	      for(j=0; j<m_VolumeSlice.YDim(); j++)
		secondary_slice_data[i + imgz*j] = secondaryMappedSlice(0,j,i);
	  }
      }
      break;
    default: m_UpdateSlice = false; return;
    }

  updateVertexCoordinates();

  if(m_SliceRenderer)
    {
      m_SliceRenderer->uploadSlice(slice_data.get(), 
				   m_SliceAxis == XY ? imgx : 
				   m_SliceAxis == XZ ? imgx : 
				   m_SliceAxis == ZY ? imgz : 0,
				   m_SliceAxis == XY ? imgy : 
				   m_SliceAxis == XZ ? imgz : 
				   m_SliceAxis == ZY ? imgy : 0);

      if(m_DrawSecondarySlice)
	m_SliceRenderer->uploadSecondarySlice(secondary_slice_data.get(), 
					      m_SliceAxis == XY ? imgx : 
					      m_SliceAxis == XZ ? imgx : 
					      m_SliceAxis == ZY ? imgz : 0,
					      m_SliceAxis == XY ? imgy : 
					      m_SliceAxis == XZ ? imgz : 
					      m_SliceAxis == ZY ? imgy : 0);
    }

  m_UpdateSlice = false;
}

void SliceRenderable::Slice::updateColorTable()
{
  if(!m_Drawable) 
    {
      m_UpdateColorTable = false;
      return;
    }

  if(m_SliceRenderer) m_SliceRenderer->uploadColorTable(m_Palette);
  m_UpdateColorTable = false;
}

void SliceRenderable::Slice::updateTextureCoordinates()
{
  unsigned int imgx,imgy,imgz;
  double tex_minx=0.0,tex_miny=0.0,tex_maxx=0.0,tex_maxy=0.0;
  double final_tex_minx=0.0,final_tex_miny=0.0,final_tex_maxx=0.0,final_tex_maxy=0.0;
  double scale_x=0.0,scale_y=0.0,scale_z=0.0;
  double trans_x=0.0,trans_y=0.0,trans_z=0.0;
  
  imgx = upToPowerOfTwo(m_VolumeFileInfo.XDim());
  imgy = upToPowerOfTwo(m_VolumeFileInfo.YDim());
  imgz = upToPowerOfTwo(m_VolumeFileInfo.ZDim());

  //Converting from object coordinates (defined by the subvolume bounding box) to texture coordinates.
  //I suppose this could be done using the GL_TEXTURE matrix but since it's only calculated infrequently
  //  (i.e. only when volumes are loaded and the subvolume box changes), it's simpler to just do it in software.
  switch(m_SliceAxis)
    {
    case XY:
      {
	tex_minx = 0.0;
	tex_miny = 0.0;
	tex_maxx = double(m_VolumeFileInfo.XDim())/double(imgx);
	tex_maxy = double(m_VolumeFileInfo.YDim())/double(imgy);
	
	scale_x = (tex_maxx-tex_minx)/(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin());
	trans_x = (-m_VolumeFileInfo.XMin())*scale_x + tex_minx;
	scale_y = (tex_maxy-tex_miny)/(m_VolumeFileInfo.YMax()-m_VolumeFileInfo.YMin());
	trans_y = (-m_VolumeFileInfo.YMin())*scale_y + tex_miny;
	final_tex_minx = m_SubVolumeBoundingBox.XMin()*scale_x + trans_x;
	final_tex_maxx = m_SubVolumeBoundingBox.XMax()*scale_x + trans_x;
	final_tex_miny = m_SubVolumeBoundingBox.YMin()*scale_y + trans_y;
	final_tex_maxy = m_SubVolumeBoundingBox.YMax()*scale_y + trans_y;
      }
      break;
    case XZ:
      {
	tex_minx = 0.0;
	tex_miny = 0.0;
	tex_maxx = double(m_VolumeFileInfo.XDim())/double(imgx);
	tex_maxy = double(m_VolumeFileInfo.ZDim())/double(imgz);
	
	scale_x = (tex_maxx-tex_minx)/(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin());
	trans_x = (-m_VolumeFileInfo.XMin())*scale_x + tex_minx;
	scale_z = (tex_maxy-tex_miny)/(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin());
	trans_z = (-m_VolumeFileInfo.ZMin())*scale_z + tex_miny;
	final_tex_minx = m_SubVolumeBoundingBox.XMin()*scale_x + trans_x;
	final_tex_maxx = m_SubVolumeBoundingBox.XMax()*scale_x + trans_x;
	final_tex_miny = m_SubVolumeBoundingBox.ZMin()*scale_z + trans_z;
	final_tex_maxy = m_SubVolumeBoundingBox.ZMax()*scale_z + trans_z;
      }
      break;
    case ZY:
      {
	tex_minx = 0.0;
	tex_miny = 0.0;
	tex_maxx = double(m_VolumeFileInfo.ZDim())/double(imgz);
	tex_maxy = double(m_VolumeFileInfo.YDim())/double(imgy);
	
	scale_z = (tex_maxx-tex_minx)/(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin());
	trans_z = (-m_VolumeFileInfo.ZMin())*scale_z + tex_minx;
	scale_y = (tex_maxy-tex_miny)/(m_VolumeFileInfo.YMax()-m_VolumeFileInfo.YMin());
	trans_y = (-m_VolumeFileInfo.YMin())*scale_y + tex_miny;
	final_tex_minx = m_SubVolumeBoundingBox.ZMin()*scale_z + trans_z;
	final_tex_maxx = m_SubVolumeBoundingBox.ZMax()*scale_z + trans_z;
	final_tex_miny = m_SubVolumeBoundingBox.YMin()*scale_y + trans_y;
	final_tex_maxy = m_SubVolumeBoundingBox.YMax()*scale_y + trans_y;
      }
      break;
    default: break;
    }

  m_TexCoords[0] = final_tex_minx; m_TexCoords[1] = final_tex_miny;
  m_TexCoords[2] = final_tex_maxx; m_TexCoords[3] = final_tex_miny;
  m_TexCoords[4] = final_tex_maxx; m_TexCoords[5] = final_tex_maxy;
  m_TexCoords[6] = final_tex_minx; m_TexCoords[7] = final_tex_maxy;
}

void SliceRenderable::Slice::updateVertexCoordinates()
{
  //the 3D widget renders the volume within a 1.0^3 box centered at the world origin.
  //So we must fit our slices within that box as well!
  VolMagick::BoundingBox renderbox = m_SubVolumeBoundingBox;
  
  double dx = renderbox.maxx - renderbox.minx;
  double dy = renderbox.maxy - renderbox.miny;
  double dz = renderbox.maxz - renderbox.minz;

  boost::array<double,3> dV = { dx, dy, dz };

  double aspectx = dx/(*std::max_element(dV.begin(),dV.end()));
  double aspecty = dy/(*std::max_element(dV.begin(),dV.end()));
  double aspectz = dz/(*std::max_element(dV.begin(),dV.end()));

  //calculate the whole volume's bounding box coordinates in "aspect" coordinates
  double min_x = m_VolumeFileInfo.XMin()*(aspectx/dx) + (-renderbox.minx)*(aspectx/dx) - aspectx/2.0;
  double max_x = m_VolumeFileInfo.XMax()*(aspectx/dx) + (-renderbox.minx)*(aspectx/dx) - aspectx/2.0;
  double span_x = (max_x - min_x)/(m_VolumeFileInfo.XDim()-1);

  double min_y = m_VolumeFileInfo.YMin()*(aspecty/dy) + (-renderbox.miny)*(aspecty/dy) - aspecty/2.0;
  double max_y = m_VolumeFileInfo.YMax()*(aspecty/dy) + (-renderbox.miny)*(aspecty/dy) - aspecty/2.0;
  double span_y = (max_y - min_y)/(m_VolumeFileInfo.YDim()-1);

  double min_z = m_VolumeFileInfo.ZMin()*(aspectz/dz) + (-renderbox.minz)*(aspectz/dz) - aspectz/2.0;
  double max_z = m_VolumeFileInfo.ZMax()*(aspectz/dz) + (-renderbox.minz)*(aspectz/dz) - aspectz/2.0;
  double span_z = (max_z - min_z)/(m_VolumeFileInfo.ZDim()-1);

  renderbox.minx = -aspectx/2.0; renderbox.maxx = aspectx/2.0;
  renderbox.miny = -aspecty/2.0; renderbox.maxy = aspecty/2.0;
  renderbox.minz = -aspectz/2.0; renderbox.maxz = aspectz/2.0;

  switch(m_SliceAxis)
    {
    case XY:
      {
	m_VertCoords[0] = renderbox.minx;
	m_VertCoords[1] = renderbox.miny;
	m_VertCoords[2] = min_z + m_Depth*span_z; //the depth value is relative to the entire bounding box, not the subvolbox
	m_VertCoords[3] = renderbox.maxx;
	m_VertCoords[4] = renderbox.miny;
	m_VertCoords[5] = min_z + m_Depth*span_z;
	m_VertCoords[6] = renderbox.maxx;
	m_VertCoords[7] = renderbox.maxy;
	m_VertCoords[8] = min_z + m_Depth*span_z;
	m_VertCoords[9] = renderbox.minx;
	m_VertCoords[10] = renderbox.maxy;
	m_VertCoords[11] = min_z + m_Depth*span_z;

	m_SecondaryVertCoords[0] = renderbox.minx;
	m_SecondaryVertCoords[1] = renderbox.miny;
	m_SecondaryVertCoords[2] = min_z + (m_Depth+m_SecondarySliceOffset)*span_z;
	m_SecondaryVertCoords[3] = renderbox.maxx;
	m_SecondaryVertCoords[4] = renderbox.miny; 
	m_SecondaryVertCoords[5] = min_z + (m_Depth+m_SecondarySliceOffset)*span_z;
	m_SecondaryVertCoords[6] = renderbox.maxx;
	m_SecondaryVertCoords[7] = renderbox.maxy;
	m_SecondaryVertCoords[8] = min_z + (m_Depth+m_SecondarySliceOffset)*span_z;
	m_SecondaryVertCoords[9] = renderbox.minx;
	m_SecondaryVertCoords[10] = renderbox.maxy;
	m_SecondaryVertCoords[11] = min_z + (m_Depth+m_SecondarySliceOffset)*span_z;
      }
      break;
    case XZ:
      {
	m_VertCoords[0] = renderbox.minx;
	m_VertCoords[1] = min_y + m_Depth*span_y;
	m_VertCoords[2] = renderbox.minz;
	m_VertCoords[3] = renderbox.maxx;
	m_VertCoords[4] = min_y + m_Depth*span_y;
	m_VertCoords[5] = renderbox.minz;
	m_VertCoords[6] = renderbox.maxx;
	m_VertCoords[7] = min_y + m_Depth*span_y;
	m_VertCoords[8] = renderbox.maxz;
	m_VertCoords[9] = renderbox.minx;
	m_VertCoords[10] = min_y + m_Depth*span_y;
	m_VertCoords[11] = renderbox.maxz;

	m_SecondaryVertCoords[0] = renderbox.minx;
	m_SecondaryVertCoords[1] = min_y + (m_Depth+m_SecondarySliceOffset)*span_y;
	m_SecondaryVertCoords[2] = renderbox.minz;
	m_SecondaryVertCoords[3] = renderbox.maxx;
	m_SecondaryVertCoords[4] = min_y + (m_Depth+m_SecondarySliceOffset)*span_y;
	m_SecondaryVertCoords[5] = renderbox.minz;
	m_SecondaryVertCoords[6] = renderbox.maxx;
	m_SecondaryVertCoords[7] = min_y + (m_Depth+m_SecondarySliceOffset)*span_y;
	m_SecondaryVertCoords[8] = renderbox.maxz;
	m_SecondaryVertCoords[9] = renderbox.minx;
	m_SecondaryVertCoords[10] = min_y + (m_Depth+m_SecondarySliceOffset)*span_y;
	m_SecondaryVertCoords[11] = renderbox.maxz;
      }
      break;
    case ZY:
      {
	m_VertCoords[0] = min_x + m_Depth*span_x;
	m_VertCoords[1] = renderbox.miny;
	m_VertCoords[2] = renderbox.minz;
	m_VertCoords[3] = min_x + m_Depth*span_x;
	m_VertCoords[4] = renderbox.miny;
	m_VertCoords[5] = renderbox.maxz;
	m_VertCoords[6] = min_x + m_Depth*span_x;
	m_VertCoords[7] = renderbox.maxy;
	m_VertCoords[8] = renderbox.maxz;
	m_VertCoords[9] = min_x + m_Depth*span_x;
	m_VertCoords[10] = renderbox.maxy;
	m_VertCoords[11] = renderbox.minz;

	m_SecondaryVertCoords[0] = min_x + (m_Depth+m_SecondarySliceOffset)*span_x;
	m_SecondaryVertCoords[1] = renderbox.miny;
	m_SecondaryVertCoords[2] = renderbox.minz;
	m_SecondaryVertCoords[3] = min_x + (m_Depth+m_SecondarySliceOffset)*span_x;
	m_SecondaryVertCoords[4] = renderbox.miny;
	m_SecondaryVertCoords[5] = renderbox.maxz;
	m_SecondaryVertCoords[6] = min_x + (m_Depth+m_SecondarySliceOffset)*span_x;
	m_SecondaryVertCoords[7] = renderbox.maxy;
	m_SecondaryVertCoords[8] = renderbox.maxz;
	m_SecondaryVertCoords[9] = min_x + (m_Depth+m_SecondarySliceOffset)*span_x;
	m_SecondaryVertCoords[10] = renderbox.maxy;
	m_SecondaryVertCoords[11] = renderbox.minz;
      }
      break;
    default: break;
    }
}

void SliceRenderable::Slice::doDraw2DContours()
{
  double scale_x,scale_y,scale_z;
  double trans_x,trans_y,trans_z;

  const gsl_interp_type *interp;//gsl_interp_cspline; /* cubic spline with natural boundary conditions */
  gsl_interp_accel *acc_x, *acc_y, *acc_z;
  gsl_spline *spline_x, *spline_y, *spline_z;
  std::vector<double> vec_x, vec_y, vec_z, vec_t; //used to build the input arrays for gsl_spline_init()
  double t = 0.0;
  double interval = 0.5;
  int contourName = 0;

  VolMagick::BoundingBox renderbox = m_SubVolumeBoundingBox;
  
  double dx = renderbox.maxx - renderbox.minx;
  double dy = renderbox.maxy - renderbox.miny;
  double dz = renderbox.maxz - renderbox.minz;

  boost::array<double,3> dV = { dx, dy, dz };

  double aspectx = dx/(*std::max_element(dV.begin(),dV.end()));
  double aspecty = dy/(*std::max_element(dV.begin(),dV.end()));
  double aspectz = dz/(*std::max_element(dV.begin(),dV.end()));

  renderbox.minx = -aspectx/2.0; renderbox.maxx = aspectx/2.0;
  renderbox.miny = -aspecty/2.0; renderbox.maxy = aspecty/2.0;
  renderbox.minz = -aspectz/2.0; renderbox.maxz = aspectz/2.0;

  if(!m_Contours->empty())
    {
      for(SurfRecon::ContourPtrMap::const_iterator i = (*m_Contours)[m_Variable][m_Timestep].begin();
	  i != (*m_Contours)[m_Variable][m_Timestep].end();
	  i++)
	{
	  if((*i).second == NULL) continue;
	  for(std::vector<SurfRecon::CurvePtr>::const_iterator cur = (*i).second->curves().begin();
	      cur != (*i).second->curves().end();
	      cur++)
	    {
	      //if the curve lies on the current slices
	      if((boost::get<0>(**cur) == m_Depth || 
		  (boost::get<0>(**cur) == (m_Depth+m_SecondarySliceOffset) && m_SecondarySliceOffset!=0)) &&
		 boost::get<1>(**cur) == SurfRecon::Orientation(m_SliceAxis))
		{
		  vec_x.clear(); vec_y.clear(); vec_z.clear(); vec_t.clear(); t = 0.0;
		  
		  //collect the control points
		  for(SurfRecon::PointPtrList::const_iterator pcur = boost::get<2>(**cur).begin();
		      pcur != boost::get<2>(**cur).end();
		      pcur++, contourName++)
		    {
		      SurfRecon::Point p;
		      qglviewer::Vec v, screenv, discv;
		      
		      p = **pcur;
		      
		      //collect the point in the arrays
		      vec_x.push_back(p.x());
		      vec_y.push_back(p.y());
		      vec_z.push_back(p.z());
		      vec_t.push_back(t); t+=1.0;
		    }
	
		  //render the curve
		  /* set interpolation type */
		  switch((*i).second->interpolationType())
		    {
		    case 0: interp = gsl_interp_linear; break;
		    case 1: interp = gsl_interp_polynomial; break;
		    case 2: interp = gsl_interp_cspline; break;
		    case 3: interp = gsl_interp_cspline_periodic; break;
		    case 4: interp = gsl_interp_akima; break;
		    case 5: interp = gsl_interp_akima_periodic; break;
			
		    default: interp = gsl_interp_cspline; break;
		    }
		    
		  gsl_spline *test_spline = gsl_spline_alloc(interp, 1000); //this is hackish but how else can i find the min size?
		  if(vec_t.size() >= gsl_spline_min_size(test_spline))
		    {
		      acc_x = gsl_interp_accel_alloc();
		      acc_y = gsl_interp_accel_alloc();
		      acc_z = gsl_interp_accel_alloc();
		      spline_x = gsl_spline_alloc(interp, vec_t.size());
		      spline_y = gsl_spline_alloc(interp, vec_t.size());
		      spline_z = gsl_spline_alloc(interp, vec_t.size());
		      gsl_spline_init(spline_x, &(vec_t[0]), &(vec_x[0]), vec_t.size());
		      gsl_spline_init(spline_y, &(vec_t[0]), &(vec_y[0]), vec_t.size());
		      gsl_spline_init(spline_z, &(vec_t[0]), &(vec_z[0]), vec_t.size());

		      SurfRecon::Point p;
		      
		      glColor3f((*i).second->color().get<0>(),
				   (*i).second->color().get<1>(),
				   (*i).second->color().get<2>());

		      glPushMatrix();

		      //do transformations to place points in the right spot relative to the volume

#if 0
		      //what am i doing wrong :(((((((((
		      glScalef(aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin()),
				  aspecty/float(m_VolumeFileInfo.YMax()-m_VolumeFileInfo.YMin()),
				  aspectz/float(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin()));
		      glTranslatef((-float(m_VolumeFileInfo.XMin())*
				       (aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin())))
				      -(aspectx/2.0),
				      (-float(m_VolumeFileInfo.YMin())*
				       (aspecty/float(m_VolumeFileInfo.YMax()-m_VolumeFileInfo.YMin())))
				      -(aspecty/2.0),
				      (-float(m_VolumeFileInfo.ZMin())*
				       (aspectz/float(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin())))
				      -(aspectz/2.0));
#endif
		      
		      double modelview[16];
		      glGetDoublev(GL_MODELVIEW_MATRIX,modelview);
		      double pin[3];
		      double pout[4];
		      
		      /*
		      qDebug("[ %5.4f %5.4f %5.4f %5.4f ]",modelview[0],modelview[4],modelview[8],modelview[12]);
		      qDebug("[ %5.4f %5.4f %5.4f %5.4f ]",modelview[1],modelview[5],modelview[9],modelview[13]);
		      qDebug("[ %5.4f %5.4f %5.4f %5.4f ]",modelview[2],modelview[6],modelview[10],modelview[14]);
		      qDebug("[ %5.4f %5.4f %5.4f %5.4f ]",modelview[3],modelview[7],modelview[11],modelview[15]);
		      */

		      glLineWidth(2.0);
		      //GLint depthfunc;
		      //glGetIntegerv(GL_DEPTH_FUNC,&depthfunc); //save the original depth func in case it's not GL_LESS

		      glEnable(GL_POLYGON_OFFSET_LINE);
		      glPolygonOffset(-1.5,-1.5);

		      //glDepthFunc(GL_LEQUAL); //draw lines on top of everything

		      scale_x = aspectx/(m_SubVolumeBoundingBox.XMax()-m_SubVolumeBoundingBox.XMin());
		      scale_y = aspecty/(m_SubVolumeBoundingBox.YMax()-m_SubVolumeBoundingBox.YMin());
		      scale_z = aspectz/(m_SubVolumeBoundingBox.ZMax()-m_SubVolumeBoundingBox.ZMin());
		      trans_x = (-m_SubVolumeBoundingBox.XMin())*scale_x - aspectx/2.0;
		      trans_y = (-m_SubVolumeBoundingBox.YMin())*scale_y - aspecty/2.0;
		      trans_z = (-m_SubVolumeBoundingBox.ZMin())*scale_z - aspectz/2.0;

		      glBegin(GL_LINE_STRIP);
		      interval = 1.0/(1+(*i).second->numberOfSamples());
		      for(double time_i = 0.0; time_i<=t-1.0; time_i+=interval)
			{
			  p = SurfRecon::Point(gsl_spline_eval(spline_x, time_i, acc_x),
					       gsl_spline_eval(spline_y, time_i, acc_y),
					       gsl_spline_eval(spline_z, time_i, acc_z));
			  
			  //#if 0
			  //qDebug("x = %f, y = %f, z = %f",p.x(),p.y(),p.z());
			  //glVertex3f(p.x(),p.y(),p.z()+0.001);

			  //pin[0] = p.x(); pin[1] = p.y(); pin[2] = p.z();
			  //mv_mult(modelview,pin,pout);
			  //qDebug("trans: x = %f, y = %f, z = %f",pout[0],pout[1],pout[2]);
			  //#endif

			  //#if 0
			 //  glVertex3f(p.x()*(aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin()))+
// 					(-float(m_VolumeFileInfo.XMin())*
// 					 (aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin())))
// 					-(aspectx/2.0),
// 					p.y()*(aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin()))+
// 					(-float(m_VolumeFileInfo.YMin())*
// 					 (aspecty/float(m_VolumeFileInfo.YMax()-m_VolumeFileInfo.YMin())))
// 					-(aspecty/2.0),
// 					p.z()*(aspectz/float(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin()))+
// 					(-float(m_VolumeFileInfo.ZMin())*
// 					 (aspectz/float(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin())))
// 					-(aspectz/2.0));
			  //#endif

			  glVertex3f(p.x()*scale_x+trans_x,
					p.y()*scale_y+trans_y,
					p.z()*scale_z+trans_z);
			}

		      //do the last point..
		      p = SurfRecon::Point(gsl_spline_eval(spline_x, t-1.0, acc_x),
					   gsl_spline_eval(spline_y, t-1.0, acc_y),
					   gsl_spline_eval(spline_z, t-1.0, acc_z));
		      //glVertex3f(p.x(),p.y(),p.z());
		      // glVertex3f(p.x()*(aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin()))+
// 				    (-float(m_VolumeFileInfo.XMin())*
// 				     (aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin())))
// 					-(aspectx/2.0),
// 				    p.y()*(aspectx/float(m_VolumeFileInfo.XMax()-m_VolumeFileInfo.XMin()))+
// 				    (-float(m_VolumeFileInfo.YMin())*
// 				     (aspecty/float(m_VolumeFileInfo.YMax()-m_VolumeFileInfo.YMin())))
// 				    -(aspecty/2.0),
// 				    p.z()*(aspectz/float(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin()))+
// 				    (-float(m_VolumeFileInfo.ZMin())*
// 				     (aspectz/float(m_VolumeFileInfo.ZMax()-m_VolumeFileInfo.ZMin())))
// 				    -(aspectz/2.0));
		      glVertex3f(p.x()*scale_x+trans_x,
				    p.y()*scale_y+trans_y,
				    p.z()*scale_z+trans_z);
		      glEnd();
		      			
		      //glDepthFunc(depthfunc);
		      glDisable(GL_POLYGON_OFFSET_LINE);

		      glPopMatrix();

		      /* clean up our mess */
		      gsl_spline_free(spline_x);
		      gsl_spline_free(spline_y);
		      gsl_spline_free(spline_z);
		      gsl_interp_accel_free(acc_x);
		      gsl_interp_accel_free(acc_y);
		      gsl_interp_accel_free(acc_z);
		    }
		  gsl_spline_free(test_spline);
		}
	    }
	}
    }
}

void SliceRenderable::Slice::setClipPlanes()
{
  VolMagick::BoundingBox renderbox = m_SubVolumeBoundingBox;
  
  double dx = renderbox.maxx - renderbox.minx;
  double dy = renderbox.maxy - renderbox.miny;
  double dz = renderbox.maxz - renderbox.minz;

  boost::array<double,3> dV = { dx, dy, dz };

  double aspectx = dx/(*std::max_element(dV.begin(),dV.end()));
  double aspecty = dy/(*std::max_element(dV.begin(),dV.end()));
  double aspectz = dz/(*std::max_element(dV.begin(),dV.end()));

  double plane0[] = { 0.0, 0.0, -1.0, aspectz/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE0, plane0);
  glEnable(GL_CLIP_PLANE0);
  
  double plane1[] = { 0.0, 0.0, 1.0, aspectz/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE1, plane1);
  glEnable(GL_CLIP_PLANE1);
  
  double plane2[] = { 0.0, -1.0, 0.0, aspecty/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE2, plane2);
  glEnable(GL_CLIP_PLANE2);

  double plane3[] = { 0.0, 1.0, 0.0, aspecty/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE3, plane3);
  glEnable(GL_CLIP_PLANE3);
  
  double plane4[] = { -1.0, 0.0, 0.0, aspectx/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE4, plane4);
  glEnable(GL_CLIP_PLANE4);
  
  double plane5[] = { 1.0, 0.0, 0.0, aspectx/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE5, plane5);
  glEnable(GL_CLIP_PLANE5);
}

void SliceRenderable::Slice::disableClipPlanes()
{
  glDisable(GL_CLIP_PLANE0);
  glDisable(GL_CLIP_PLANE1);
  glDisable(GL_CLIP_PLANE2);
  glDisable(GL_CLIP_PLANE3);
  glDisable(GL_CLIP_PLANE4);
  glDisable(GL_CLIP_PLANE5);
}

bool SliceRenderable::ARBFragmentProgramSliceRenderer::init()
{
  /* make sure the proper extensions are initialized */
  if(//glewIsSupported("GL_VERSION_1_3") &&
     glewIsSupported("GL_ARB_vertex_program") &&
     glewIsSupported("GL_ARB_fragment_program") &&
     glewIsSupported("GL_ARB_multitexture"))
    {
      /* Initialize the fragment program */
      const GLubyte program[] = 
        "!!ARBfp1.0\n"
        "PARAM c0 = {0.5, 1, 2.7182817, 0};\n"
        "TEMP R0;\n"
        "TEX R0.x, fragment.texcoord[0].xyzx, texture[0], 2D;\n"
        "TEX result.color, R0.x, texture[1], 1D;\n"
        "END\n";
      /* initialize the fragment program */
      glEnable(GL_FRAGMENT_PROGRAM_ARB);
      if(glIsProgramARB(m_FragmentProgram))
	glDeleteProgramsARB(1,&m_FragmentProgram);
      glGenProgramsARB(1,&(m_FragmentProgram));
      glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, m_FragmentProgram);
      glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB, strlen((const char *)program), program);
      glDisable(GL_FRAGMENT_PROGRAM_ARB);
    }
  else
    return false;
  
  return true;
}

void SliceRenderable::ARBFragmentProgramSliceRenderer::draw(float *vert_coords, float *tex_coords)
{
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);

  glColor3f(1.0,1.0,1.0);
  
  //glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, m_FragmentProgram);

  /* bind the transfer function */
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glEnable(GL_TEXTURE_1D);
  glBindTexture(GL_TEXTURE_1D, m_PaletteTexture);
	
  /* bind the data texture */
  glActiveTextureARB(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, m_SliceTexture);

  glBegin(GL_QUADS);
  for(unsigned int j=0; j<4; j++)
    {
      //glTexCoord2f(m_SliceTiles[i].tex_coords[j*2+0],m_SliceTiles[i].tex_coords[j*2+1]);
      glMultiTexCoord2fARB(GL_TEXTURE0_ARB,tex_coords[j*2+0],tex_coords[j*2+1]);
      glVertex3f(vert_coords[j*3+0],vert_coords[j*3+1],vert_coords[j*3+2]);
    }
  glEnd();

#if 0
  glBegin(GL_LINE_LOOP);
  for(unsigned int j=0; j<4; j++)
    {
      glColor3f(1.0,0.0,0.0);
      glVertex3f(vert_coords[j*3+0],vert_coords[j*3+1],vert_coords[j*3+2]);
    }
  glEnd();
#endif

  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, 0);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
 
  glDisable(GL_TEXTURE_1D);
  glDisable(GL_TEXTURE_2D);
  //glDisable(GL_BLEND);
   
  glPopAttrib();
}

void SliceRenderable::ARBFragmentProgramSliceRenderer::drawSecondary(float *vert_coords, float *tex_coords)
{
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);

  glColor3f(1.0,1.0,1.0);
  
  //glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, m_FragmentProgram);

  /* bind the transfer function */
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glEnable(GL_TEXTURE_1D);
  glBindTexture(GL_TEXTURE_1D, m_PaletteTexture);
	
  /* bind the data texture */
  glActiveTextureARB(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, m_SecondarySliceTexture);

  glBegin(GL_QUADS);
  for(unsigned int j=0; j<4; j++)
    {
      //glTexCoord2f(m_SliceTiles[i].tex_coords[j*2+0],m_SliceTiles[i].tex_coords[j*2+1]);
      glMultiTexCoord2fARB(GL_TEXTURE0_ARB,tex_coords[j*2+0],tex_coords[j*2+1]);
      glVertex3f(vert_coords[j*3+0],vert_coords[j*3+1],vert_coords[j*3+2]);
    }
  glEnd();

#if 0
  glBegin(GL_LINE_LOOP);
  for(unsigned int j=0; j<4; j++)
    {
      glColor3f(1.0,0.0,0.0);
      glVertex3f(vert_coords[j*3+0],vert_coords[j*3+1],vert_coords[j*3+2]);
    }
  glEnd();
#endif

  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, 0);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
 
  glDisable(GL_TEXTURE_1D);
  glDisable(GL_TEXTURE_2D);
  //glDisable(GL_BLEND);
   
  glPopAttrib();
}

void SliceRenderable::ARBFragmentProgramSliceRenderer::uploadSlice(unsigned char *slice, 
								   unsigned int dimx, 
								   unsigned int dimy)
{
  //if(glIsTexture(m_SliceTexture))
    glDeleteTextures(1,&m_SliceTexture);
  glGenTextures(1,&m_SliceTexture);
  glBindTexture(GL_TEXTURE_2D, m_SliceTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, dimx, dimy, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, slice);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void SliceRenderable::ARBFragmentProgramSliceRenderer::uploadSecondarySlice(unsigned char *slice, 
									    unsigned int dimx, 
									    unsigned int dimy)
{
  //if(glIsTexture(m_SecondarySliceTexture))
    glDeleteTextures(1,&m_SecondarySliceTexture);
  glGenTextures(1,&m_SecondarySliceTexture);
  glBindTexture(GL_TEXTURE_2D, m_SecondarySliceTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, dimx, dimy, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, slice);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void SliceRenderable::ARBFragmentProgramSliceRenderer::uploadColorTable(unsigned char *palette)
{
  //if(glIsTexture(m_PaletteTexture))
    glDeleteTextures(1,&m_PaletteTexture);
  glGenTextures(1,&m_PaletteTexture);
  glBindTexture(GL_TEXTURE_1D, m_PaletteTexture);
  glTexImage1D(GL_TEXTURE_1D, 0, 4, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, palette);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

#endif
