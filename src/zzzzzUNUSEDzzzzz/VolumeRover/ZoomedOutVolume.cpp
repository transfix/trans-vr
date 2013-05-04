/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

/* $Id: ZoomedOutVolume.cpp 3500 2011-01-25 16:51:16Z arand $ */

// ZoomedOutVolume.cpp: implementation of the ZoomedOutVolume class.
//
//////////////////////////////////////////////////////////////////////

#include <boost/scoped_ptr.hpp>
#include <VolumeRover/ZoomedOutVolume.h>
#include <VolumeRover/GeometryInteractor.h>
#include <VolumeRover/GeometryRenderer.h>
#include <VolumeWidget/ZoomInteractor.h>
#include <VolumeWidget/TrackballRotateInteractor.h>
#include <qobject.h>
#ifdef VOLUMEGRIDROVER
#include <VolumeGridRover/VolumeGridRover.h>
#endif
#include <VolumeLibrary/CG_RenderDef.h>

#if 0
#ifdef Q_WS_MACX
#include <glu.h>
#else
#include <GL/glu.h>
#endif
#endif

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

ZoomedOutVolume::ZoomedOutVolume(Extents* extent, RenderableArray* geometryArray) :
RoverRenderable(extent, geometryArray)
{
#ifdef VOLUMEGRIDROVER
	m_EnableSliceRendering = false;
#endif
	setOpaqueRenderable(m_Rover3DWidget.getWireCubes());
	setSuperOpaqueRenderable(m_Rover3DWidget.getAxes());
}

ZoomedOutVolume::~ZoomedOutVolume()
{

}

void ZoomedOutVolume::addToSimpleOpenGLWidget(SimpleOpenGLWidget& simpleOpenGLWidget, QObject* receiver, const char* member)
{
  //these objects are cloned within setMouseHandler()
  boost::scoped_ptr<ZoomInteractor> middleButtonHandler(new ZoomInteractor);
  boost::scoped_ptr<ZoomInteractor> wheelHandler(new ZoomInteractor);
  boost::scoped_ptr<TrackballRotateInteractor> rightButtonHandler(new TrackballRotateInteractor);
  boost::scoped_ptr<GeometryInteractor> specialHandler(new GeometryInteractor(&m_GeometryRenderer));

	simpleOpenGLWidget.initForContext(this);
	simpleOpenGLWidget.setMainRenderable(this);
	simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, m_Rover3DWidget.getHandler());

	MouseHandler* handler;
	handler = simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::MiddleButtonHandler, middleButtonHandler.get());
	//QObject::connect(handler, SIGNAL(ViewChanged()), receiver, member);
	handler = simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::WheelHandler, wheelHandler.get());
	//QObject::connect(handler, SIGNAL(ViewChanged()), receiver, member);
	handler = simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::RightButtonHandler, rightButtonHandler.get());
	QObject::connect(handler, SIGNAL(ViewChanged()), receiver, member);

	// the special handler
	simpleOpenGLWidget.setMouseHandler(SimpleOpenGLWidget::SpecialHandler, specialHandler.get());
}

void ZoomedOutVolume::setAspectRatio(double x, double y, double z)
{
	m_VolumeRenderer->setAspectRatio(x,y,z);
	m_Rover3DWidget.setAspectRatio(x,y,z);
}

void ZoomedOutVolume::connectRoverSignals(QObject* receiver, const char* exploring, const char* released)
{
	QObject::connect(&m_Rover3DWidget, SIGNAL(RoverExploring()), receiver, exploring);
	QObject::connect(&m_Rover3DWidget, SIGNAL(RoverReleased()), receiver, released);
}

void ZoomedOutVolume::toggleWireCubeDrawing(bool state)
{
	if (state) {
		setOpaqueRenderable(m_Rover3DWidget.getWireCubes());
		setSuperOpaqueRenderable(m_Rover3DWidget.getAxes());
	}
	else {
		setOpaqueRenderable(NULL);
		setSuperOpaqueRenderable(NULL);
	}
}
	
void ZoomedOutVolume::setRover3DWidgetColor(float r, float g, float b)
{
	m_Rover3DWidget.setColor(r,g,b);
}

Extents ZoomedOutVolume::getSubVolume() const
{
	return m_Rover3DWidget.getSubVolume();
}

Extents ZoomedOutVolume::getBoundary() const
{
	return m_Rover3DWidget.getBoundary();
}

Vector ZoomedOutVolume::getPreviewOrigin() const
{
	return m_Rover3DWidget.getSubVolume().getOrigin();
}

bool ZoomedOutVolume::render()
{
	bool ret1 = true, ret2 = true, ret3 = true, ret4 = true, ret5 = true, ret6 = true;

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
	// Need to check -CHA
	//	if (m_EnableShadedVolumes) {
			// set the light and view vectors
			float lightvec[4] = { 0.0, 0.5, 2.0, 0.0 };
			float viewvec[4] = { 0.0, 0.0, 1.0, 0.0 }, halfw[4];
			
			normalize(lightvec);
                        m_VolumeRenderer->setLight(lightvec);
			halfw[0] = viewvec[0] + lightvec[0];
			halfw[1] = viewvec[1] + lightvec[1];
			halfw[2] = viewvec[2] + lightvec[2];
			halfw[3] = viewvec[3] + lightvec[3];
                        normalize(halfw);
                        m_VolumeRenderer->setView(halfw);
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
			viewvec[0] += lightvec[0];
			viewvec[1] += lightvec[1];
			viewvec[2] += lightvec[2];
			mv_mult(inverse, viewvec, scratch);
			normalize(scratch);
			m_VolumeRenderer->setView(scratch);
*/
	//	}
		
		ret5 = m_VolumeRenderer->renderVolume();
	}
	if (m_SuperOpaqueRenderable) {
		glDisable(GL_DEPTH_TEST);
		ret6 = m_SuperOpaqueRenderable->render();
	}
	glPopAttrib();
	return ret1 && ret2 && ret3 && ret4 && ret5 && ret6;
}

