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

// GeometryRenderer.cpp: implementation of the GeometryRenderer class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRover/GeometryRenderer.h>
#include <glew/glew.h>
#include <VolumeWidget/Matrix.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

GeometryRenderer::GeometryRenderer(RenderableArray* geometryArray, Extents* extent) :
  m_GeometryArray(geometryArray), m_Extent(extent), m_ClipGeometry(true)
{
	m_TransX = 0.0;
	m_TransY = 0.0;
	m_TransZ = 0.0;
	m_ScaleFactor = 1.0;
}

GeometryRenderer::~GeometryRenderer()
{

}

void GeometryRenderer::translateBy(float x, float y, float z)
{
	m_TransX += x;
	m_TransY += y;
	m_TransZ += z;
}

void GeometryRenderer::scaleBy(float f)
{
	m_ScaleFactor += f;
}

void GeometryRenderer::rotateBy(float angle, float x, float y, float z)
{
	m_Orientation.rotate(angle, x,y,z);
}

void GeometryRenderer::clearTransformation()
{
	m_TransX = 0.0;
	m_TransY = 0.0;
	m_TransZ = 0.0;
	m_ScaleFactor = 1.0;
	m_Orientation.set(1.0,0.0,0.0,0.0);
}

void GeometryRenderer::setClipGeometry(bool clip)
{
  m_ClipGeometry = clip;
}

bool GeometryRenderer::render()
{
	if (!m_GeometryArray) {
		return false;
	}

	bool ret1; 
	glEnable(GL_LIGHTING);


	// determine the max dimension
	double aspectX,aspectY,aspectZ;
	double centerX,centerY,centerZ;
	aspectX = m_Extent->getXMax() - m_Extent->getXMin();
	aspectY = m_Extent->getYMax() - m_Extent->getYMin();
	aspectZ = m_Extent->getZMax() - m_Extent->getZMin();
	double max = (aspectX>aspectY?aspectX:aspectY);
	max = (max>aspectZ?max:aspectZ);

	// compute the center of the space
	centerX = (m_Extent->getXMax() + m_Extent->getXMin()) / 2.0;
	centerY = (m_Extent->getYMax() + m_Extent->getYMin()) / 2.0;
	centerZ = (m_Extent->getZMax() + m_Extent->getZMin()) / 2.0;
	aspectX/=max;
	aspectY/=max;
	aspectZ/=max;

	if(m_ClipGeometry)
	  setClipPlanes(aspectX, aspectY, aspectZ);

	Matrix matrix;
	// any extra translation
	matrix.preMultiplication( Matrix::translation(m_TransX,m_TransY,m_TransZ) );
	// center
	matrix.preMultiplication(Matrix::translation(
		(float)(-centerX),
		(float)(-centerY),
		(float)(-centerZ)
		));
	// any extra scaling
	matrix.preMultiplication(Matrix::scale(m_ScaleFactor,m_ScaleFactor,m_ScaleFactor));
	// scale to aspect ratio
	matrix.preMultiplication(Matrix::scale(
		(float)(1.0/max),
		(float)(1.0/max),
		(float)(1.0/max)
		));

	// any rotation
	matrix.preMultiplication(m_Orientation.buildMatrix());

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(matrix.getMatrix());

	ret1 = m_GeometryArray->render();
	glPopMatrix();

	if(m_ClipGeometry)
	  disableClipPlanes();

	return ret1;
}

void GeometryRenderer::setWireframeMode(bool state)
{
	m_GeometryArray->setWireframeMode(state);
}

void GeometryRenderer::setSurfWithWire(bool state)
{
	m_GeometryArray->setSurfWithWire(state);
}

void GeometryRenderer::setClipPlanes(double aspectX, double aspectY, double aspectZ) const
{
	double plane0[] = { 0.0, 0.0, -1.0, aspectZ/2.0 + 0.00001 };
	glClipPlane(GL_CLIP_PLANE0, plane0);
	glEnable(GL_CLIP_PLANE0);

	double plane1[] = { 0.0, 0.0, 1.0, aspectZ/2.0 + 0.00001 };
	glClipPlane(GL_CLIP_PLANE1, plane1);
	glEnable(GL_CLIP_PLANE1);

	double plane2[] = { 0.0, -1.0, 0.0, aspectY/2.0 + 0.00001 };
	glClipPlane(GL_CLIP_PLANE2, plane2);
	glEnable(GL_CLIP_PLANE2);

	double plane3[] = { 0.0, 1.0, 0.0, aspectY/2.0 + 0.00001 };
	glClipPlane(GL_CLIP_PLANE3, plane3);
	glEnable(GL_CLIP_PLANE3);

	double plane4[] = { -1.0, 0.0, 0.0, aspectX/2.0 + 0.00001 };
	glClipPlane(GL_CLIP_PLANE4, plane4);
	glEnable(GL_CLIP_PLANE4);

	double plane5[] = { 1.0, 0.0, 0.0, aspectX/2.0 + 0.00001 };
	glClipPlane(GL_CLIP_PLANE5, plane5);
	glEnable(GL_CLIP_PLANE5);
}

void GeometryRenderer::disableClipPlanes() const
{
	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE1);

	glDisable(GL_CLIP_PLANE2);
	glDisable(GL_CLIP_PLANE3);

	glDisable(GL_CLIP_PLANE4);
	glDisable(GL_CLIP_PLANE5);
}

