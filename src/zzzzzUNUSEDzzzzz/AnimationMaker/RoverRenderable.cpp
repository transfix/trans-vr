// RoverRenderable.cpp: implementation of the RoverRenderable class.
//
//////////////////////////////////////////////////////////////////////

#include <AnimationMaker/RoverRenderable.h>

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
	// render the items
	bool ret1 = true, ret2 = true, ret3 = true, ret4 = true, ret5 = true;
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
	if (m_EnableVolumeRendering && m_VolumeRenderer) {
		ret4 = m_VolumeRenderer->renderVolume();
	}
	if (m_SuperOpaqueRenderable) {
		glDisable(GL_DEPTH_TEST);
		ret5 = m_SuperOpaqueRenderable->render();
	}
	glPopAttrib();
	return ret1 && ret2 && ret3 && ret4 && ret5;
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

void RoverRenderable::setAspectRatio(double x, double y, double z)
{
	m_VolumeRenderer->setAspectRatio(x,y,z);
}

