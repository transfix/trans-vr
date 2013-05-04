// RoverRenderable.h: interface for the RoverRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ROVERRENDERABLE_H__1D35D9A5_1F64_4EBB_A87E_F7E212B92A7A__INCLUDED_)
#define AFX_ROVERRENDERABLE_H__1D35D9A5_1F64_4EBB_A87E_F7E212B92A7A__INCLUDED_

#include <VolumeWidget/Renderable.h>
#include <VolumeFileTypes/VolumeBufferManager.h>
#include <AnimationMaker/GeometryRenderer.h>
#include <Contouring/MultiContour.h>
#include <VolumeLibrary/VolumeRenderer.h>
#include <VolumeWidget/Extents.h>

class RoverRenderable : public Renderable  
{
public:
	RoverRenderable(Extents* extent, RenderableArray* geometryArray);
	virtual ~RoverRenderable();

	void setVolumeRenderer(VolumeRenderer* volumeRenderer);
	void setOpaqueRenderable(Renderable* renderable);
	void setSuperOpaqueRenderable(Renderable* renderable);


	VolumeBufferManager* getVolumeBufferManager();
	GeometryRenderer* getGeometryRenderer();
	MultiContour* getMultiContour();
	VolumeRenderer* getVolumeRenderer();

	virtual bool initForContext();
	virtual bool render();

	bool getShowIsosurface();
	void setShowIsosurface(bool value);
	bool getShowVolumeRendering();
	void setShowVolumeRendering(bool value);

	virtual void setAspectRatio(double x, double y, double z);

protected:
	VolumeBufferManager m_VolumeBufferManager;
	GeometryRenderer m_GeometryRenderer;
	MultiContour m_MultiContour;
	VolumeRenderer* m_VolumeRenderer;

	Renderable* m_OpaqueRenderable;
	Renderable* m_SuperOpaqueRenderable;

	bool m_EnableVolumeRendering;
	bool m_EnableIsocontourRendering;
	bool m_EnableGeometryRendering;
};

#endif // !defined(AFX_ROVERRENDERABLE_H__1D35D9A5_1F64_4EBB_A87E_F7E212B92A7A__INCLUDED_)
