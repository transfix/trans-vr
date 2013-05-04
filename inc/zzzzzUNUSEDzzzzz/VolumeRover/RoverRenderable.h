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

// RoverRenderable.h: interface for the RoverRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ROVERRENDERABLE_H__1D35D9A5_1F64_4EBB_A87E_F7E212B92A7A__INCLUDED_)
#define AFX_ROVERRENDERABLE_H__1D35D9A5_1F64_4EBB_A87E_F7E212B92A7A__INCLUDED_

#include <VolumeWidget/Renderable.h>
#include <VolumeFileTypes/VolumeBufferManager.h>
#include <VolumeRover/GeometryRenderer.h>

#ifdef VOLUMEGRIDROVER
#include <VolumeRover/SliceRenderable.h>
#endif

#include <Contouring/MultiContour.h>
#include <VolumeLibrary/VolumeRenderer.h>
#include <VolumeWidget/Extents.h>

#ifdef USING_SKELETONIZATION
#include <VolumeRover/SkeletonRenderable.h>
#endif

class View;

///\class RoverRenderable RoverRenderable.h
///\brief The RoverRenderable class is a Renderable derived container that
///	provides a clean interface to the different rendering modalities present in
///	Volume Rover. These are volume rendering, geometry rendering, and
///	isocontour rendering. Geometry rendering and isocontour rendering are
///	seperate here because their functions are quite different even though they
///	are both drawing triangle meshes. This class also provides access to a
///	VolumeBufferManager which handles access to volume data.
///\author Anthony Thane
///\author John Wiggins

class RoverRenderable : public Renderable  
{
public:
///\fn RoverRenderable::RoverRenderable(Extents* extent, RenderableArray* geometryArray)
///\brief The class contructor. Note that this class contains a pure virtual
///	method and therefore cannot be instantiated by itself.
///\param extent The bounding box of the volume data to be rendered by this object.
///\param geometryArray This is a RenderableArray which contains the loaded Geometry ojects that are rendered by the GeometryRenderer.
	RoverRenderable(Extents* extent, RenderableArray* geometryArray);
	virtual ~RoverRenderable();

///\fn void RoverRenderable::setVolumeRenderer(VolumeRenderer* volumeRenderer)
///\brief This function assigns a VolumeRenderer instance to be used for...
///	(drumroll please) volume rendering.
///\param volumeRenderer The VolumeRenderer instance.
	void setVolumeRenderer(VolumeRenderer* volumeRenderer);
///\fn void RoverRenderable::setOpaqueRenderable(Renderable* renderable)
///\brief This function sets the opaque renderable for the class. This
///	Renderable instance is drawn first and could be anything, but in practice
///	it's the wireframe bounding box of the volume data.
///\param renderable Some object derived from Renderable.
	void setOpaqueRenderable(Renderable* renderable);
///\fn void RoverRenderable::setSuperOpaqueRenderable(Renderable* renderable)
///\brief This function sets the super opaque renderable for the class. This
///	Renderable instance is drawn last and thus can draw over anything that has
///	been drawn before it. This spot is taken by the axes in the Rover widget. As
///	such, it is only used in ZoomedOutVolume.
///\param renderable Some object derived from Renderable.
	void setSuperOpaqueRenderable(Renderable* renderable);

///\fn VolumeBufferManager* RoverRenderable::getVolumeBufferManager()
///\brief This function gives access to the object's VolumeBufferManager
///	instance.
///\return A pointer to a VolumeBufferManager.
	VolumeBufferManager* getVolumeBufferManager();
///\fn GeometryRenderer* RoverRenderable::getGeometryRenderer()
///\brief This function gives access to the object's GeometryRenderer instance.
///\return A pointer to a GeometryRenderer.
	GeometryRenderer* getGeometryRenderer();
///\fn MultiContour* RoverRenderable::getMultiContour()
///\brief This function gives access to the object's MultiContour instance.
///\return A pointer to a MultiContour.
	MultiContour* getMultiContour();
///\fn VolumeRenderer* RoverRenderable::getVolumeRenderer()
///\brief This function gives access to the object's VolumeRenderer instance.
///\return A pointer to a VolumeRenderer.
	VolumeRenderer* getVolumeRenderer();

///\fn virtual bool initForContext()
///\brief This function initializes the VolumeRenderer.
	virtual bool initForContext();
	virtual bool render();

///\fn bool RoverRenderable::getShowIsosurface()
///\brief This function reports whether isosurface rendering is enabled.
///\return A bool. true -> isosurfaces are rendered, false -> isosurfaces are
///	not rendered.
	bool getShowIsosurface();
///\fn void RoverRenderable::setShowIsosurface(bool value)
///\brief This function enables/disables isosurface rendering.
///\param value If true, enables isosurface rendering. If false, disables
///	isosurface rendering.
	void setShowIsosurface(bool value);
///\fn bool RoverRenderable::getShowVolumeRendering()
///\brief This function reports whether volume rendering is enabled.
///\return A bool. true -> the volume is rendered, false -> the volume is not
///	rendered.
	bool getShowVolumeRendering();
///\fn void RoverRenderable::setShowVolumeRendering(bool value)
///\brief This function enables/disables volume rendering.
///\param value If true, enables volume rendering. If false, disables volume
///	rendering.
	void setShowVolumeRendering(bool value);
///\fn bool RoverRenderable::getShadedVolumeRendering()
///\brief This function reports whether shaded volume rendering is enabled.
///\return A bool. true -> shaded rendering is enabled, false -> shaded
///	rendering is disabled.
	bool getShadedVolumeRendering();
///\fn void RoverRenderable::setShadedVolumeRendering(bool value)
///\brief This function enables/disables shaded volume rendering.
///\param value If true, enables shaded rendering. If false, disables shaded
///	rendering.
	void setShadedVolumeRendering(bool value);

///\fn virtual void RoverRenderable::toggleDepthCueing(bool state)
///\brief This function enables/disables depth cueing.
///\param state If true, enables depth cueing. If false, disables depth cueing.
	virtual void toggleDepthCueing(bool state);
///\fn virtual void RoverRenderable::setDepthCueColor(float r, float g, float b)
///\brief This function assigns a color to the "fog" in the depth cue.
///\param r The red component of the depth cue color.
///\param g The green component of the depth cue color.
///\param b The blue component of the depth cue color.
	virtual void setDepthCueColor(float r, float g, float b);

///\fn virtual void RoverRenderable::setAspectRatio(double x, double y, double z) = 0
///\brief This pure virtual function allows derived classes to set the aspect
///	ratio of their child objects.
///\param x The X component of the aspect ratio.
///\param y The Y component of the aspect ratio.
///\param z The Z component of the aspect ratio.
	virtual void setAspectRatio(double x, double y, double z) = 0;

#ifdef VOLUMEGRIDROVER
	void setSliceRendering(bool state);
	bool sliceRendering() { return m_EnableSliceRendering; }
	void setSlice(SliceRenderable::SliceAxis a, int depth);
	SliceRenderable* getSliceRenderable();
#endif


#ifdef USING_SKELETONIZATION
	void setSkeletonRendering(bool state);
	SkeletonRenderable* getSkeletonRenderable();
#endif

protected:
	VolumeBufferManager m_VolumeBufferManager;
	GeometryRenderer m_GeometryRenderer;
	MultiContour m_MultiContour;
	VolumeRenderer* m_VolumeRenderer;

	Renderable* m_OpaqueRenderable;
	Renderable* m_SuperOpaqueRenderable;

	GLfloat m_DepthCueColor[4];

	bool m_EnableVolumeRendering;
	bool m_EnableIsocontourRendering;
	bool m_EnableGeometryRendering;
	bool m_EnableShadedVolumes;
	bool m_EnableDepthCue;

#ifdef VOLUMEGRIDROVER
	bool m_EnableSliceRendering;
	SliceRenderable m_SliceRenderable;
#endif


#ifdef USING_SKELETONIZATION
	bool m_EnableSkeletonRendering;
	SkeletonRenderable m_SkeletonRenderable;
#endif
};

#endif // !defined(AFX_ROVERRENDERABLE_H__1D35D9A5_1F64_4EBB_A87E_F7E212B92A7A__INCLUDED_)
