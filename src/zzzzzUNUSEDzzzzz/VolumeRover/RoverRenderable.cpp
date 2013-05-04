/******************************************************************************
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

// RoverRenderable.cpp: implementation of the RoverRenderable class.
//
//////////////////////////////////////////////////////////////////////

#include <stdio.h>

#include <VolumeRover/RoverRenderable.h>
#include <VolumeLibrary/CG_RenderDef.h>

#if 0
#ifdef Q_WS_MACX
#include <glu.h>
#else
#include <GL/glu.h>
#endif
#endif

#include <time.h>
#ifdef _WIN32
        #include <sys/types.h>
        #include <sys/timeb.h>
#else
        #include <sys/time.h>
#endif

static inline double getTime()
{
#ifdef _WIN32
        time_t ltime;
        _timeb tstruct;
        time( &ltime );
        _ftime( &tstruct );
        return (double) (ltime + 1e-3*(tstruct.millitm));
#else
    struct timeval t;
    gettimeofday( &t, NULL );
    return (double)(t.tv_sec + 1e-6*t.tv_usec);
#endif
}


static inline double m3_det(double *mat)
{
  double det;

  det = mat[0] * (mat[4]*mat[8] - mat[7]*mat[5])
      - mat[1] * (mat[3]*mat[8] - mat[6]*mat[5])
      + mat[2] * (mat[3]*mat[7] - mat[6]*mat[4]);

  return det;
}

static inline void m4_submat(double *mr, double *mb, int i, int j)
{
  int ti, tj, idst, jdst;

  for (ti = 0; ti < 4; ti++)
  {
    if (ti < i)
      idst = ti;
    else if (ti > i)
      idst = ti-1;

    for (tj = 0; tj < 4; tj++)
    {
      if (tj < j)
        jdst = tj;
      else if (tj > j)
        jdst = tj-1;

      if (ti != i && tj != j)
        mb[idst*3 + jdst] = mr[ti*4 + tj ];
    }
  }
}

static inline double m4_det(double *mr)
{
  double det, result = 0.0, i = 1.0, msub3[9];
  int n;

  for (n = 0; n < 4; n++, i *= -1.0)
  {
    m4_submat(mr, msub3, 0, n);

    det = m3_det(msub3);
    result += mr[n] * det * i;
  }

  return result;
}

static inline int m4_inverse(double *mr, double *ma)
{
  double mtemp[9], mdet = m4_det(ma);
  int i, j, sign;

  if (fabs(mdet) == 0.0)
    return 0;

  for (i = 0; i < 4; i++)
  {
    for (j = 0; j < 4; j++)
    {
      sign = 1 - ((i +j) % 2) * 2;

      m4_submat(ma, mtemp, i, j);

      mr[i+j*4] = (m3_det(mtemp) * sign) / mdet;
    }
  }

  return 1;
}

static inline void mv_mult(double m[16], float vin[3], float vout[4])
{
  vout[0] = vin[0]*m[0] + vin[1]*m[4] + vin[2]*m[8] + m[12];
  vout[1] = vin[0]*m[1] + vin[1]*m[5] + vin[2]*m[9] + m[13];
  vout[2] = vin[0]*m[2] + vin[1]*m[6] + vin[2]*m[10] + m[14];
  vout[3] = vin[0]*m[3] + vin[1]*m[7] + vin[2]*m[11] + m[15];
}

static inline void normalize(float vin[3])
{
  float div = sqrt(vin[0]*vin[0]+vin[1]*vin[1]+vin[2]*vin[2]);

  if (div != 0.0)
  {
    vin[0] /= div;
    vin[1] /= div;
    vin[2] /= div;
  }
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RoverRenderable::RoverRenderable(Extents* extent, RenderableArray* geometryArray) :
m_GeometryRenderer(geometryArray, extent), m_OpaqueRenderable(0),
m_SuperOpaqueRenderable(0)
{
	m_VolumeRenderer = 0;
	m_EnableVolumeRendering = true;
	m_EnableIsocontourRendering = true;
	m_EnableGeometryRendering = true;
	m_EnableShadedVolumes = false;
	m_EnableDepthCue = false;
#ifdef VOLUMEGRIDROVER
	m_EnableSliceRendering = false;
#endif
#ifdef USING_SKELETONIZATION
	m_EnableSkeletonRendering = true;
#endif

	setDepthCueColor(0.0, 0.0, 0.0);
}

RoverRenderable::~RoverRenderable()
{
	delete m_VolumeRenderer;
}

void RoverRenderable::setVolumeRenderer(VolumeRenderer* volumeRenderer)
{
	m_VolumeRenderer = volumeRenderer;
}

void RoverRenderable::setOpaqueRenderable(Renderable* renderable)
{
	m_OpaqueRenderable = renderable;
}

void RoverRenderable::setSuperOpaqueRenderable(Renderable* renderable)
{
	m_SuperOpaqueRenderable = renderable;
}

VolumeBufferManager* RoverRenderable::getVolumeBufferManager()
{
	return &m_VolumeBufferManager;
}

GeometryRenderer* RoverRenderable::getGeometryRenderer()
{
	return &m_GeometryRenderer;
}

MultiContour* RoverRenderable::getMultiContour()
{
	return &m_MultiContour;
}

VolumeRenderer* RoverRenderable::getVolumeRenderer()
{
	return m_VolumeRenderer;
}

bool RoverRenderable::initForContext()
{
	if (m_VolumeRenderer) {
		return m_VolumeRenderer->initRenderer();
	}
	else {
		return true;
	}
}

bool RoverRenderable::render()
{
	bool ret1 = true, ret2 = true, ret3 = true, 
	  ret4 = true, ret5 = true, ret6 = true, ret7 = true;

	// set up the fog function for depth cueing
	if (m_EnableDepthCue) { 
		glFogfv(GL_FOG_COLOR,m_DepthCueColor);
		glFogf(GL_FOG_DENSITY,0.0001f);
		glFogi(GL_FOG_MODE,GL_LINEAR);
		glFogf(GL_FOG_START,-3.); // 6.0 ~ the center of the bounding cube
		glFogf(GL_FOG_END,6.);
		glHint(GL_FOG_HINT, GL_FASTEST);
		glEnable(GL_FOG);
	}
	else {
		glDisable(GL_FOG);
	}
	
	// render the items
	glPushAttrib(GL_DEPTH_BUFFER_BIT);
	if (m_OpaqueRenderable) {
		glEnable(GL_DEPTH_TEST);
		ret1 = m_OpaqueRenderable->render();
	}
	if (m_EnableGeometryRendering) {
		glEnable(GL_DEPTH_TEST);
		ret2 = m_GeometryRenderer.render();
	}
	if (m_EnableIsocontourRendering) {
		glEnable(GL_DEPTH_TEST);
		ret3 = m_MultiContour.render();
	}
	
#ifdef VOLUMEGRIDROVER
	if(m_EnableSliceRendering) {
	  glEnable(GL_DEPTH_TEST);
		ret4 = m_SliceRenderable.render();
	}
#endif

#ifdef USING_SKELETONIZATION
	if(m_EnableSkeletonRendering) {
	  glEnable(GL_DEPTH_TEST);
	  ret5 = m_SkeletonRenderable.render();
	}
#endif

	if (m_EnableVolumeRendering && m_VolumeRenderer) {
	//	Need to Check - CHA
/*
  		GLdouble modelview[16], inverse[16];
  		// get the modelview matrix
		glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
		// invert it
		m4_inverse(inverse, modelview);
		// multiply and set the light and view vectors
		mv_mult(inverse, lightvec, scratch);
		normalize(scratch);
		m_VolumeRenderer->setLight(scratch);
		mv_mult(inverse, viewvec, scratch);
		normalize(scratch);
		m_VolumeRenderer->setView(scratch);
*/		
		ret6 = m_VolumeRenderer->renderVolume();
	}
	if (m_SuperOpaqueRenderable) {
		glDisable(GL_DEPTH_TEST);
		ret7 = m_SuperOpaqueRenderable->render();
	}
	glPopAttrib();

	// FPS computation
	{
	  static unsigned int fpsCounter_ = 0;
	  static float f_p_s_ = 0.0;
	  static double newtime=0.0,oldtime=0.0;

	  const unsigned int maxCounter = 50;
	  if (++fpsCounter_ == maxCounter)
	    {
	      newtime = getTime();
	      f_p_s_ = maxCounter / (newtime-oldtime);
	      //fpsString_ = QString("%1Hz").arg(f_p_s_, 0, 'f', ((f_p_s_ < 10.0)?1:0));
	      printf("%fHz\n",f_p_s_);
	      fpsCounter_ = 0;
	      oldtime = newtime;
	    }
	}
	  
	return ret1 && ret2 && ret3 && ret4 && ret5 && ret6 && ret7;
}

bool RoverRenderable::getShowIsosurface()
{
	return m_EnableIsocontourRendering;
}

void RoverRenderable::setShowIsosurface(bool value)
{
	m_EnableIsocontourRendering = value;
}

bool RoverRenderable::getShowVolumeRendering()
{
	return m_EnableVolumeRendering;
}

void RoverRenderable::setShowVolumeRendering(bool value)
{
	m_EnableVolumeRendering = value;
}

bool RoverRenderable::getShadedVolumeRendering()
{
	return m_EnableShadedVolumes;
}

void RoverRenderable::setShadedVolumeRendering(bool value)
{
	m_EnableShadedVolumes = value;
}

void RoverRenderable::toggleDepthCueing(bool state)
{
	m_EnableDepthCue = state;
}

void RoverRenderable::setDepthCueColor(GLfloat r, GLfloat g, GLfloat b)
{
	m_DepthCueColor[0] = r;
	m_DepthCueColor[1] = g;
	m_DepthCueColor[2] = b;
	m_DepthCueColor[3] = 0.25;
}

#ifdef VOLUMEGRIDROVER
void RoverRenderable::setSlice(SliceRenderable::SliceAxis a, int depth)
{
  //printf("SliceAxis: %s, Depth: %d\n",a == SliceRenderable::XY ? "XY" : a == SliceRenderable::XZ ? "XZ" : "ZY",depth);
  m_SliceRenderable.setDepth(a,depth);
}

void RoverRenderable::setSliceRendering(bool state)
{
  m_EnableSliceRendering = state;
}

SliceRenderable* RoverRenderable::getSliceRenderable()
{
  return &m_SliceRenderable;
}
#endif
#ifdef USING_SKELETONIZATION
void RoverRenderable::setSkeletonRendering(bool state)
{
  m_EnableSkeletonRendering = state;
}

SkeletonRenderable* RoverRenderable::getSkeletonRenderable()
{
  return &m_SkeletonRenderable;
}
#endif
