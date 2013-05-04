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

// GeometryRenderer.h: interface for the GeometryRenderer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GEOMETRYRENDERER_H__4EA27817_C9AD_4331_BCE1_D9E0573CDA12__INCLUDED_)
#define AFX_GEOMETRYRENDERER_H__4EA27817_C9AD_4331_BCE1_D9E0573CDA12__INCLUDED_

#include <VolumeWidget/Renderable.h>
#include <VolumeWidget/Extents.h>
#include <VolumeWidget/RenderableArray.h>
#ifdef WIN32
#include "..\VolumeWidget\Quaternion.h"
#else
#include <VolumeWidget/Quaternion.h>
#endif

// Doxygen info
///\class GeometryRenderer GeometryRenderer.h
///\brief This class renders an array of Geometry objects.
///\author Anythony Thane, John Wiggins
class GeometryRenderer : public Renderable
{
public:
///\fn GeometryRenderer::GeometryRenderer(RenderableArray* geometryArray, Extents* extent)
///\brief The class constructor.
///\param geometryArray A RenderableArray instance. Does not necessarily have
///	to contain geometry.
///\param extent The bounding box that the geometry will be drawn in.
	GeometryRenderer(RenderableArray* geometryArray, Extents* extent);
	virtual ~GeometryRenderer();

///\fn void GeometryRenderer::translateBy(float x, float y, float z)
///\brief Translates the geometry's position by some vector.
///\param x Translation amount along x-axis.
///\param y Translation amount along y-axis.
///\param z Translation amount along z-axis.
	void translateBy(float x, float y, float z);
///\fn void GeometryRenderer::scaleBy(float f)
///\brief Scales the geometry by some factor.
///\param f This is the scale factor. It is added to the class' internal
///		scale factor. There is no 'set' function for the scale factor.
	void scaleBy(float f);
///\fn void GeometryRenderer::rotateBy(float angle, float x, float y, float z)
///\brief This function rotates the geometry relative to its current
///		orientation.
///\param angle The angle of the rotation.
///\param x X component of the rotation axis.
///\param y Y component of the rotation axis.
///\param z Z component of the rotation axis.
	void rotateBy(float angle, float x, float y, float z);
///\fn void GeometryRenderer::clearTransformation()
///\brief Resets the geometry transformation.
	void clearTransformation();

	void setClipGeometry(bool clip);

	virtual bool render();

	virtual void setWireframeMode(bool state);

	virtual void setSurfWithWire(bool state);
	
protected:
	void setClipPlanes(double aspectX, double aspectY, double aspectZ) const;
	void disableClipPlanes() const;

	RenderableArray* const m_GeometryArray;
	Extents* const m_Extent;

	Quaternion m_Orientation;
	float m_TransX, m_TransY, m_TransZ;
	float m_ScaleFactor;

	bool m_ClipGeometry;
};

#endif // !defined(AFX_GEOMETRYRENDERER_H__4EA27817_C9AD_4331_BCE1_D9E0573CDA12__INCLUDED_)
