/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
             Vinay Siddavanahalli <skvinay@cs.utexas.edu>
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

// GeometryRenderable.h: interface for the GeometryRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GEOMETRYRENDERABLE_H__E41AC409_393F_4E0D_9FFC_09EFCE9124A8__INCLUDED_)
#define AFX_GEOMETRYRENDERABLE_H__E41AC409_393F_4E0D_9FFC_09EFCE9124A8__INCLUDED_

#include <VolumeWidget/Renderable.h>
#include <vector>
class Geometry;

///\class GeometryRenderable GeometryRenderable.h
///\brief This Renderable instance is used to render geometry contained in a
///	Geometry instance in various ways (points, lines, triangles, quads).
///\author Anthony Thane
class GeometryRenderable : public Renderable  
{
public:
///\fn GeometryRenderable(Geometry* geometry, bool deleteGeometryOnDestruct = true)
///\brief The class contstructor.
///\param geometry A pointer to a Geometry object
///\param deleteGeometryOnDestruct A boolean that determines whether the class or the caller is responsible for deleting the passed in Geometry object.
	GeometryRenderable(Geometry* geometry, bool deleteGeometryOnDestruct = true);
	virtual ~GeometryRenderable();

	virtual bool initForContext();
	virtual bool deinitForContext();
	virtual bool render();

	virtual void setWireframeMode(bool state) { m_WireframeRender = state; }

	//specific to this class.. if drawing triangles, this will enable drawing
	//the wireframe on top of the surface
	virtual void setSurfWithWire(bool state) { m_SurfWithWire = state; }

///\fn Geometry* getGeometry()
///\brief This function returns a pointer to the class' Geometry instance.
///\return A pointer to a Geometry object
	Geometry* getGeometry();

protected:
	void drawGeometry(Geometry* geometry);
	void drawPoints(Geometry* geometry);
	void drawLines(Geometry* geometry);
	void drawTris(Geometry* geometry);
	void drawFlatTris(Geometry* geometry);
	void drawQuads(Geometry* geometry);
	void drawFlatQuads(Geometry* geometry);

	Geometry* const m_Geometry;
	bool m_DeleteGeometry;
	bool m_WireframeRender;
	bool m_SurfWithWire;

	// *************** LBIE rendering stuff
	void drawGeoFrame();
	int m_CutFlag;
	int m_FlatFlag;
	//functions ripped from MyGLDrawer in LBIE
	void geoframe_display();
	void geoframe_display_tri(int, int, int, int, int);
	void geoframe_display_hexa(int, int, int);
	void geoframe_display_tri0(int, int, int, int, int, int);
	void geoframe_display_tri00(int, int, int, int, int, int, int);
	void geoframe_display_tri_cross(int, int, int, float t0, float t1, float t2, int, int);
	void geoframe_display_tri_cross0(int, int, int, float t0, float t1, float t2, int, int);
	void geoframe_display_prism(int, int, int, int, int);
	void geoframe_display_tetra(int, int, int);
	void geoframe_display_tetra_in(int, int, int);
	void geoframe_display_tri_v(float*, float*, float*, int, int);
	void geoframe_display_tri_vv(float*, float*, float*, int, int, int);
	void geoframe_display_tri_vv_0(float*, float*, float*, int, int, int);
	void geoframe_display_quad_v(float*, float*, float*, float*, int, int);
	void geoframe_display_permute_1(float*, float*, float*, float*, float);
	void geoframe_display_permute_2(float*, float*, float*, float*, float);
	void geoframe_display_permute_3(float*, float*, float*, float*, float);
	void geoframe_display_permute_1_z(float*, float*, float*, float*, float);
	void geoframe_display_permute_2_z(float*, float*, float*, float*, float);
	void geoframe_display_permute_3_z(float*, float*, float*, float*, float);
	void geoframe_display_1(int*, int, float*, float*, float*, float*, float, int, int);
	void geoframe_display_2(int*, int, float*, float*, float*, float*, float, int, int);
	void geoframe_display_3(int*, int, float*, float*, float*, float*, float, int, int);
	void geoframe_display_1_z(int*, int, float*, float*, float*, float*, float, int, int);
	void geoframe_display_2_z(int*, int, float*, float*, float*, float*, float, int, int);
	void geoframe_display_3_z(int*, int, float*, float*, float*, float*, float, int, int);

	// *************** scene_geometry_t rendering stuff
	//If this is true, then we have updated the on-card vertex buffer objects
	//with the contents of _geometries since it was last changed.
	bool _usingVBO;
	bool _vboUpdated;

	//TODO: finish adding this rendering mode!

	//id's of allocated buffers that we should clean up when necessary
	std::vector<unsigned int> _allocatedBuffers;
};

#endif // !defined(AFX_GEOMETRYRENDERABLE_H__E41AC409_393F_4E0D_9FFC_09EFCE9124A8__INCLUDED_)
