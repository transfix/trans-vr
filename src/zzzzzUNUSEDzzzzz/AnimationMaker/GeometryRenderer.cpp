// GeometryRenderer.cpp: implementation of the GeometryRenderer class.
//
//////////////////////////////////////////////////////////////////////

#include <AnimationMaker/GeometryRenderer.h>
#include <qgl.h>
#include <VolumeWidget/Matrix.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

GeometryRenderer::GeometryRenderer(RenderableArray* geometryArray, Extents* extent) :
m_GeometryArray(geometryArray), m_Extent(extent)
{
}

GeometryRenderer::~GeometryRenderer()
{

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

	setClipPlanes(aspectX, aspectY, aspectZ);

	Matrix matrix;
	// center
	matrix.preMultiplication(Matrix::translation(
		(float)(-centerX),
		(float)(-centerY),
		(float)(-centerZ)
		));
	// scale to aspect ratio
	matrix.preMultiplication(Matrix::scale(
		(float)(1.0/max),
		(float)(1.0/max),
		(float)(1.0/max)
		));
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(matrix.getMatrix());
	ret1 = m_GeometryArray->render();
	glPopMatrix();

	disableClipPlanes();

	return ret1;
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

