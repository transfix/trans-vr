// GeometryRenderer.h: interface for the GeometryRenderer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GEOMETRYRENDERER_H__4EA27817_C9AD_4331_BCE1_D9E0573CDA12__INCLUDED_)
#define AFX_GEOMETRYRENDERER_H__4EA27817_C9AD_4331_BCE1_D9E0573CDA12__INCLUDED_

#include <VolumeWidget/Renderable.h>
#include <VolumeWidget/Extents.h>
#include <VolumeWidget/RenderableArray.h>

class GeometryRenderer : public Renderable
{
public:
	GeometryRenderer(RenderableArray* geometryArray, Extents* extent);
	virtual ~GeometryRenderer();

	virtual bool render();

protected:
	void setClipPlanes(double aspectX, double aspectY, double aspectZ) const;
	void disableClipPlanes() const;

	RenderableArray* const m_GeometryArray;
	Extents* const m_Extent;
};

#endif // !defined(AFX_GEOMETRYRENDERER_H__4EA27817_C9AD_4331_BCE1_D9E0573CDA12__INCLUDED_)
