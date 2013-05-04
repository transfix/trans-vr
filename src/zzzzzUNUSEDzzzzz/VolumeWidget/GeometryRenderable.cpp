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

/* $Id: GeometryRenderable.cpp 3373 2010-12-17 21:56:16Z zqyork $ */

// GeometryRenderable.cpp: implementation of the GeometryRenderable class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/GeometryRenderable.h>
#include <cvcraw_geometry/Geometry.h>
#include <VolumeWidget/Matrix.h>
#include <iostream>
#include <glew/glew.h>
#include <qwidget.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

GeometryRenderable::GeometryRenderable(Geometry* geometry, bool deleteGeometryOnDestruct) :
  m_Geometry(geometry), m_DeleteGeometry(deleteGeometryOnDestruct), m_WireframeRender(false),
  m_SurfWithWire(false), m_CutFlag(0), m_FlatFlag(1)
{
}

GeometryRenderable::~GeometryRenderable()
{
	if (m_DeleteGeometry) {
		delete m_Geometry;
	}
}

bool GeometryRenderable::initForContext()
{
	return true;
}

bool GeometryRenderable::deinitForContext()
{
	return true;
}

bool GeometryRenderable::render()
{
	drawGeometry(m_Geometry);
	
	return true;
}

#if 0
void GeometryRenderable::applyBaseTransformation()
{
	float xMin,xMax, yMin,yMax, zMin,zMax;
	float aspectX,aspectY,aspectZ;
	float centerX,centerY,centerZ;
	float max;
	
	// this will calculate the extents of the geometry if needed
	m_Geometry->GetReadyToDrawWire();

	// get the extents of the geometry
	xMin = m_Geometry->m_Min[0];
	yMin = m_Geometry->m_Min[1];
	zMin = m_Geometry->m_Min[2];
	xMax = m_Geometry->m_Max[0];
	yMax = m_Geometry->m_Max[1];
	zMax = m_Geometry->m_Max[2];
	
	// determine the max dimension
	aspectX = xMax - xMin;
	aspectY = yMax - yMin;
	aspectZ = zMax - zMin;
	max = (aspectX>aspectY?aspectX:aspectY);
	max = (max>aspectZ?max:aspectZ);
	
	// compute the center of the space
	centerX = (xMax + xMin) / 2.0;
	centerY = (yMax + yMin) / 2.0;
	centerZ = (zMax + zMin) / 2.0;

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

	matrix.preMultiplication(Matrix::translation(0.5,0.5,0.5));
	
	// apply the transformation
	glMultMatrixf(matrix.getMatrix());
}
#endif

Geometry* GeometryRenderable::getGeometry()
{
	return m_Geometry;
}

void GeometryRenderable::drawGeometry(Geometry* geometry)
{
	glColor4fv( geometry->m_DiffuseColor );
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, geometry->m_DiffuseColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, geometry->m_SpecularColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, geometry->m_AmbientColor);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, geometry->m_Shininess);
	glLineWidth(geometry->m_LineWidth);

	//if(geometry->m_GeoFrame)
	//drawGeoFrame();
	  
	if(geometry->m_NumLines == 0 && 
	   geometry->m_NumTris == 0 && 
	   geometry->m_NumQuads == 0 &&
	   geometry->m_NumPoints != 0)
	  {
	    geometry->GetReadyToDrawWire();
	    drawPoints(geometry);
	  }
	else
	  {
	    if(geometry->m_NumLines)
	      {
		geometry->GetReadyToDrawWire();
		drawLines(geometry);
	      }
	    if (geometry->m_NumTris)
	      {
		geometry->GetReadyToDrawSmooth();
		drawTris(geometry);
	      }
	    if (geometry->m_NumQuads)
	      {
		//geometry->GetReadyToDrawSmooth();
		geometry->GetReadyToDrawFlat();
		//drawQuads(geometry);
		drawFlatQuads(geometry);
	      }
	  }
}

void GeometryRenderable::drawPoints(Geometry* geometry)
{
	glPushAttrib(GL_LIGHTING_BIT | GL_POINT_BIT);
	if(geometry->m_PointSize)
	{
		glPointSize(geometry->m_PointSize);
	}
	else glPointSize(2.0);
	
	
	glEnable(GL_POINT_SMOOTH);

	glDisable(GL_LIGHTING);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	if(geometry->m_PointColors) {
	  glEnableClientState(GL_COLOR_ARRAY);
	  glColorPointer(3, GL_FLOAT, 0, geometry->m_PointColors.get());
	}

	glVertexPointer(3, GL_FLOAT, 0, geometry->m_Points.get());
	glDrawArrays(GL_POINTS, 0, geometry->m_NumPoints);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glPopAttrib();
}

void GeometryRenderable::drawLines(Geometry* geometry)
{

	glPushAttrib(GL_LIGHTING_BIT);
	glDisable(GL_LIGHTING);
	glEnableClientState(GL_VERTEX_ARRAY);
	if (geometry->m_LineColors) {
		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(3, GL_FLOAT, 0, geometry->m_LineColors.get());
	}
	if(geometry->m_LineWidth)
		glLineWidth(geometry->m_LineWidth);
	else glLineWidth(1.0);
	
	glVertexPointer(3, GL_FLOAT, 0, geometry->m_LineVerts.get());
	glDrawElements(GL_LINES, geometry->m_NumLines*2, GL_UNSIGNED_INT, geometry->m_Lines.get());
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glPopAttrib();
}

void GeometryRenderable::drawTris(Geometry* geometry)
{
	glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
#if 0
	if (m_WireframeRender) { // wireframe rendering
		int i;
		//glDisable(GL_LIGHTING);
		if (geometry->m_TriVertColors) { //draw with color
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glEnable(GL_COLOR_MATERIAL);

			for (i=0; i<geometry->m_NumTris; i++) {
				glBegin(GL_LINE_LOOP);
				
				glColor3fv(&(geometry->m_TriVertColors[geometry->m_Tris[i*3+0]*3]));
				glNormal3fv(&(geometry->m_TriVertNormals[geometry->m_Tris[i*3+0]*3]));
				glVertex3fv(&(geometry->m_TriVerts[geometry->m_Tris[i*3+0]*3]));
				
				glColor3fv(&(geometry->m_TriVertColors[geometry->m_Tris[i*3+1]*3]));
				glNormal3fv(&(geometry->m_TriVertNormals[geometry->m_Tris[i*3+1]*3]));
				glVertex3fv(&(geometry->m_TriVerts[geometry->m_Tris[i*3+1]*3]));
				
				glColor3fv(&(geometry->m_TriVertColors[geometry->m_Tris[i*3+2]*3]));
				glNormal3fv(&(geometry->m_TriVertNormals[geometry->m_Tris[i*3+2]*3]));
				glVertex3fv(&(geometry->m_TriVerts[geometry->m_Tris[i*3+2]*3]));
				
				glEnd();
			}
		}
		else { // draw without color
			for (i=0; i<geometry->m_NumTris; i++) {
				glBegin(GL_LINE_LOOP);
			
				glNormal3fv(&(geometry->m_TriVertNormals[geometry->m_Tris[i*3+0]*3]));
				glVertex3fv(&(geometry->m_TriVerts[geometry->m_Tris[i*3+0]*3]));
				
				glNormal3fv(&(geometry->m_TriVertNormals[geometry->m_Tris[i*3+1]*3]));
				glVertex3fv(&(geometry->m_TriVerts[geometry->m_Tris[i*3+1]*3]));
				
				glNormal3fv(&(geometry->m_TriVertNormals[geometry->m_Tris[i*3+2]*3]));
				glVertex3fv(&(geometry->m_TriVerts[geometry->m_Tris[i*3+2]*3]));
				
				glEnd();
			}
		}
	}
	else { // triangle rendering
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		if (geometry->m_TriVertTexCoords) {
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			glTexCoordPointer(3, GL_FLOAT, 0, geometry->m_TriVertTexCoords.get());
		}
		if (geometry->m_TriVertColors) {
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glEnable(GL_COLOR_MATERIAL);
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(3, GL_FLOAT, 0, geometry->m_TriVertColors.get());
		}
	
		glVertexPointer(3, GL_FLOAT, 0, geometry->m_TriVerts.get());
		glNormalPointer(GL_FLOAT, 0, geometry->m_TriVertNormals.get());
		glDrawElements(GL_TRIANGLES, geometry->m_NumTris*3, GL_UNSIGNED_INT, geometry->m_Tris.get());
	
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
#endif
	
	GLint params[2];

	//back up current setting
	glGetIntegerv(GL_POLYGON_MODE,params);
	
	{ // triangle rendering
	  glEnableClientState(GL_VERTEX_ARRAY);
	  glEnableClientState(GL_NORMAL_ARRAY);
	  if (geometry->m_TriVertTexCoords)
	    {
	      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	      glTexCoordPointer(3, GL_FLOAT, 0, geometry->m_TriVertTexCoords.get());
	    }
	  if (geometry->m_TriVertColors)
	    {
	      glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	      glEnable(GL_COLOR_MATERIAL);
	      glEnableClientState(GL_COLOR_ARRAY);
	      glColorPointer(3, GL_FLOAT, 0, geometry->m_TriVertColors.get());
	    }
	
	  glVertexPointer(3, GL_FLOAT, 0, geometry->m_TriVerts.get());
	  glNormalPointer(GL_FLOAT, 0, geometry->m_TriVertNormals.get());

	  if(!m_WireframeRender ||
	     (m_WireframeRender && m_SurfWithWire))
	    {
	      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	      if(m_SurfWithWire)
		{
		  glEnable(GL_POLYGON_OFFSET_FILL);
		  glPolygonOffset(1.0,1.0);
		}

	      glDrawElements(GL_TRIANGLES, geometry->m_NumTris*3, GL_UNSIGNED_INT, geometry->m_Tris.get());

	      if(m_SurfWithWire)
		{
		  glPolygonOffset(0.0,0.0);
		  glDisable(GL_POLYGON_OFFSET_FILL);
		}
	    }

	  if(m_WireframeRender)
	    {
	      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	      if(m_SurfWithWire)
		{
		  glDisable(GL_LIGHTING);
		  glDisableClientState(GL_COLOR_ARRAY);
		  //		  glEnable(GL_POLYGON_OFFSET_LINE);
		  //		  glPolygonOffset(-1.0,-1.0);
		  glColor3f(0.0,0.0,0.0); //black wireframe
		}

	      glDrawElements(GL_TRIANGLES, geometry->m_NumTris*3, GL_UNSIGNED_INT, geometry->m_Tris.get());

	      if(m_SurfWithWire)
		{
		  //		  glPolygonOffset(0.0,0.0);
		  //		  glDisable(GL_POLYGON_OFFSET_LINE);
		}
	    }
	
	  glDisableClientState(GL_VERTEX_ARRAY);
	  glDisableClientState(GL_NORMAL_ARRAY);
	  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	  glDisableClientState(GL_COLOR_ARRAY);
	}

	//restore previous setting for polygon mode
	glPolygonMode(GL_FRONT,params[0]);
	glPolygonMode(GL_BACK,params[1]);
	
	glPopAttrib();
}

void GeometryRenderable::drawFlatTris(Geometry* geometry)
{
	glPushAttrib(GL_LIGHTING_BIT);
	glEnable(GL_LIGHTING);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	if (geometry->m_TriFlatTexCoords) {
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(3, GL_FLOAT, 0, geometry->m_TriFlatTexCoords.get());
	}
	
	glVertexPointer(3, GL_FLOAT, 0, geometry->m_TriFlatVerts.get());
	glNormalPointer(GL_FLOAT, 0, geometry->m_TriFlatNormals.get());
	glDrawArrays(GL_TRIANGLES, 0, geometry->m_NumTris*3);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glPopAttrib();
}

void GeometryRenderable::drawQuads(Geometry* geometry)
{
	glPushAttrib(GL_LIGHTING_BIT);
	glEnable(GL_LIGHTING);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	if (geometry->m_QuadVertTexCoords) {
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(3, GL_FLOAT, 0, geometry->m_QuadVertTexCoords.get());
	}
	
	glVertexPointer(3, GL_FLOAT, 0, geometry->m_QuadVerts.get());
	glNormalPointer(GL_FLOAT, 0, geometry->m_QuadVertNormals.get());
	glDrawElements(GL_QUADS, geometry->m_NumQuads*4, GL_UNSIGNED_INT, geometry->m_Quads.get());
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glPopAttrib();
}

void GeometryRenderable::drawFlatQuads(Geometry* geometry)
{
	glPushAttrib(GL_LIGHTING_BIT);
	glEnable(GL_LIGHTING);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	if (geometry->m_QuadFlatTexCoords) {
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(3, GL_FLOAT, 0, geometry->m_QuadFlatTexCoords.get());
	}
	
	glVertexPointer(3, GL_FLOAT, 0, geometry->m_QuadFlatVerts.get());
	glNormalPointer(GL_FLOAT, 0, geometry->m_QuadFlatNormals.get());
	glDrawArrays(GL_QUADS, 0, geometry->m_NumQuads*4);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glPopAttrib();
}

void GeometryRenderable::drawGeoFrame()
{
  geoframe_display();
}

static inline void cross(float* dest, const float* v1, const float* v2)
{
  dest[0] = v1[1]*v2[2] - v1[2]*v2[1];
  dest[1] = v1[2]*v2[0] - v1[0]*v2[2];
  dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

void GeometryRenderable::geoframe_display_tri(int i, int j, int k, int c, int normalflag)
{
  float v1[3], v2[3], norm[3], v0[3], plane_x;
  int vert, ii, index, my_bool;

  plane_x =  32.0f; //(float) ((int)biggestDim/2.0);	
  my_bool =	((m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][0]] == 1) && 
		 (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][1]] == 1) &&
		 (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][2]] == 1) &&
		 (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][3]] == 1)) || normalflag == -1;

  if(my_bool) {

    if(normalflag == -1) {
      vert = m_Geometry->m_GeoFrame->quads[c][i];	v0[0] = m_Geometry->m_GeoFrame->verts[vert][0];
      vert = m_Geometry->m_GeoFrame->quads[c][j];	v1[0] = m_Geometry->m_GeoFrame->verts[vert][0];
      vert = m_Geometry->m_GeoFrame->quads[c][k];	v2[0] = m_Geometry->m_GeoFrame->verts[vert][0];
      if(v0[0] >= plane_x && v1[0] >= plane_x && v2[0] >= plane_x) {
	//if(v0[0] <= plane_x && v1[0] <= plane_x && v2[0] <= plane_x &&
	//	v0[2] <= plane_x && v1[2] <= plane_x && v2[2] <= plane_x) {
	normalflag = -2;
      }
    }
    else {
      vert = m_Geometry->m_GeoFrame->quads[c][i];
      v1[0] = v2[0] = -m_Geometry->m_GeoFrame->verts[vert][0];
      v1[1] = v2[1] = -m_Geometry->m_GeoFrame->verts[vert][1];
      v1[2] = v2[2] = -m_Geometry->m_GeoFrame->verts[vert][2];
							
      vert = m_Geometry->m_GeoFrame->quads[c][j];
      v1[0] += m_Geometry->m_GeoFrame->verts[vert][0];
      v1[1] += m_Geometry->m_GeoFrame->verts[vert][1];
      v1[2] += m_Geometry->m_GeoFrame->verts[vert][2];
							
      vert = m_Geometry->m_GeoFrame->quads[c][k];
      v2[0] += m_Geometry->m_GeoFrame->verts[vert][0];
      v2[1] += m_Geometry->m_GeoFrame->verts[vert][1];
      v2[2] += m_Geometry->m_GeoFrame->verts[vert][2];

      cross(norm, v1, v2);

      // normal flipping
      if (normalflag == 1) {
	norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
      }
    }
    glBegin(GL_TRIANGLES);

    for(ii = 0; ii < 3; ii++) {
      index = i;
      if(ii == 1) index = j;
      if(ii == 2) index = k;
      vert = m_Geometry->m_GeoFrame->quads[c][index];
      if (normalflag == -1 || m_FlatFlag == 2) {	// hexa   flat shading + wireframe
	norm[0] = m_Geometry->m_GeoFrame->normals[vert][0];
	norm[1] = m_Geometry->m_GeoFrame->normals[vert][1];
	norm[2] = m_Geometry->m_GeoFrame->normals[vert][2];
      }
      if(normalflag == -2) {norm[0] = 1.0;	norm[1] = 0.0;	norm[2] = 0.0;}
      glNormal3d(norm[0], norm[1], norm[2]);
      // hexa cross section   //zq changed - to + in the following two lines on March 3, 2009 
      if((m_Geometry->m_GeoFrame->verts[vert][0]+32.5f )*(m_Geometry->m_GeoFrame->verts[vert][0]+32.5f )+(m_Geometry->m_GeoFrame->verts[vert][1]+32.5f )*(m_Geometry->m_GeoFrame->verts[vert][1]+32.5f ) +
	 (m_Geometry->m_GeoFrame->verts[vert][2]+32.5f )*(m_Geometry->m_GeoFrame->verts[vert][2]+32.5f ) < 100000.5f && normalflag != -2 && m_Geometry->m_GeoFrame->numhexas > 0) {//900.5
	//GLfloat diffRefl[] = {0.6f, 0.6f, 1.0f, 1.00f};
	GLfloat diffRefl[] = {1.0f, 1.0f, 1.0f, 1.00f};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl);
	glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl);
      }
      else {
	//GLfloat diffRefl_1[] = {0.9, 0.7, 0.0, 1.00};
	//GLfloat diffRefl_1[] = {1.0f, 0.7f, 0.0f, 1.00f};
	GLfloat diffRefl_1[] = {1.0, 1.0, 1.0, 1.00};
	//GLfloat diffRefl_1[] = {1.0, 0.0, 0.0, 1.00};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
	glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl_1);
      }

      if(normalflag <= -1 && m_Geometry->m_GeoFrame->verts[vert][0] < plane_x)
	//if(normalflag <= -1 && m_Geometry->m_GeoFrame->verts[vert][0] < plane_x && m_Geometry->m_GeoFrame->verts[vert][2] < plane_x)
	glVertex3d(
		   //plane_x,
		   m_Geometry->m_GeoFrame->verts[vert][0],
		   m_Geometry->m_GeoFrame->verts[vert][1],
		   m_Geometry->m_GeoFrame->verts[vert][2]);
      else
	glVertex3d(
		   m_Geometry->m_GeoFrame->verts[vert][0],
		   m_Geometry->m_GeoFrame->verts[vert][1],
		   m_Geometry->m_GeoFrame->verts[vert][2]);
    }
    glEnd();

  }
}

void GeometryRenderable::geoframe_display_tri0(int i, int j, int k, int c, int normalflag, int wire_flag)
{
  float v1[3], v2[3], norm[3];
  int vert, ii, index, my_bool, my_bool_0;

  my_bool =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == 1) && 
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == 1) &&
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == 1);
  my_bool_0 =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == -1) && 
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == -1) &&
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == -1);

  if(my_bool || my_bool_0) {

    vert = m_Geometry->m_GeoFrame->triangles[c][i];
    v1[0] = v2[0] = -m_Geometry->m_GeoFrame->verts[vert][0];
    v1[1] = v2[1] = -m_Geometry->m_GeoFrame->verts[vert][1];
    v1[2] = v2[2] = -m_Geometry->m_GeoFrame->verts[vert][2];
						
    vert = m_Geometry->m_GeoFrame->triangles[c][j];
    v1[0] += m_Geometry->m_GeoFrame->verts[vert][0];
    v1[1] += m_Geometry->m_GeoFrame->verts[vert][1];
    v1[2] += m_Geometry->m_GeoFrame->verts[vert][2];
						
    vert = m_Geometry->m_GeoFrame->triangles[c][k];
    v2[0] += m_Geometry->m_GeoFrame->verts[vert][0];
    v2[1] += m_Geometry->m_GeoFrame->verts[vert][1];
    v2[2] += m_Geometry->m_GeoFrame->verts[vert][2];

    cross(norm, v1, v2);

    // normal flipping
    if (normalflag == 1) {
      norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
    }
    if (m_Geometry->m_GeoFrame->bound_tri[c] == 1) {
      norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
    }

    vert = m_Geometry->m_GeoFrame->triangles[c][i];
    v1[0] = m_Geometry->m_GeoFrame->verts[vert][0];
    v1[1] = m_Geometry->m_GeoFrame->verts[vert][1];
    v1[2] = m_Geometry->m_GeoFrame->verts[vert][2];

    if(my_bool_0) {
      //if(my_bool && v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2] < 90000.0) {
      GLfloat diffRefl[] = { m_Geometry->m_GeoFrame->color[vert][0],
		  m_Geometry->m_GeoFrame->color[vert][1],
   		  m_Geometry->m_GeoFrame->color[vert][2], 1.0f};
      //GLfloat diffRefl[] = {1.0, 0.7, 0.0, 1.00};
      glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl);
      glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl);
    }
    else {
      //GLfloat diffRefl_1[] = {0.0, 0.6, 0.5, 1.00};
      GLfloat diffRefl_1[] = { m_Geometry->m_GeoFrame->color[vert][0],
		  m_Geometry->m_GeoFrame->color[vert][1],
   		  m_Geometry->m_GeoFrame->color[vert][2], 1.0f};
      //GLfloat diffRefl_1[] = {0.6f, 0.6f, 1.0f, 1.00f};
      glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
      glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl_1);
    }
    glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT);

    glEnable(GL_LIGHTING);
    if(wire_flag == 0) {
      glBegin(GL_TRIANGLES);
      for(ii = 0; ii < 3; ii++) {
	index = i;
	if(ii == 1) index = j;
	if(ii == 2) index = k;
	vert = m_Geometry->m_GeoFrame->triangles[c][index];
	if(m_FlatFlag == 2) {
	  norm[0] = m_Geometry->m_GeoFrame->normals[vert][0];
	  norm[1] = m_Geometry->m_GeoFrame->normals[vert][1];
	  norm[2] = m_Geometry->m_GeoFrame->normals[vert][2];
	  /*
	  // show function value
	  if(m_Geometry->m_GeoFrame->funcs[vert][0] < -1.0) {
	  GLfloat diffRefl_1[] = {1, 0.3f, 0.3f, 1};
	  glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
	  }
	  else if(m_Geometry->m_GeoFrame->funcs[vert][0] > 1.0) {
	  GLfloat diffRefl_1[] = {0.6f, 0.6f, 1, 1};
	  glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
	  }
	  else {
	  GLfloat diffRefl_1[] = {1, 1, 1, 1};
	  glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
	  }
	  */
	  //GLfloat diffRefl_1[] = {m_Geometry->m_GeoFrame->funcs[vert][0], -0.5f, 1.0f-m_Geometry->m_GeoFrame->funcs[vert][0], 1.0f};
	  //if (my_bool_0) {
	  //norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
	  //}
	}
	glNormal3d(norm[0], norm[1], norm[2]);
	glColor3d(m_Geometry->m_GeoFrame->color[vert][0],
		  m_Geometry->m_GeoFrame->color[vert][1],
   		  m_Geometry->m_GeoFrame->color[vert][2]);
	/*glColor3d(1.0,0.0,0.0);*/
	/*qDebug("%f %f %f",m_Geometry->m_GeoFrame->color[vert][0],
		  m_Geometry->m_GeoFrame->color[vert][1],
   		  m_Geometry->m_GeoFrame->color[vert][2]);*/

	glVertex3d( m_Geometry->m_GeoFrame->verts[vert][0],
		    m_Geometry->m_GeoFrame->verts[vert][1],
		    m_Geometry->m_GeoFrame->verts[vert][2] );
      }
      glEnd();
    }
    else {
      glBegin(GL_LINE_STRIP);

      for(ii = 0; ii < 3; ii++) {
	index = i;
	if(ii == 1) index = j;
	if(ii == 2) index = k;
	vert = m_Geometry->m_GeoFrame->triangles[c][index];
	if(m_FlatFlag == 2) {
	  norm[0] = m_Geometry->m_GeoFrame->normals[vert][0];
	  norm[1] = m_Geometry->m_GeoFrame->normals[vert][1];
	  norm[2] = m_Geometry->m_GeoFrame->normals[vert][2];
	  if (my_bool_0) {
	    norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
	  }
	}
	glNormal3d(norm[0], norm[1], norm[2]);
	glColor3d(0.0,0.0,0.0);
	glVertex3d(
		   m_Geometry->m_GeoFrame->verts[vert][0],
		   m_Geometry->m_GeoFrame->verts[vert][1],
		   m_Geometry->m_GeoFrame->verts[vert][2]);
      }

      vert = m_Geometry->m_GeoFrame->triangles[c][i];
      glNormal3d(norm[0], norm[1], norm[2]);
      glColor3d(0.0,0.0,0.0);
      glVertex3d(
		 m_Geometry->m_GeoFrame->verts[vert][0],
		 m_Geometry->m_GeoFrame->verts[vert][1],
		 m_Geometry->m_GeoFrame->verts[vert][2]);
      glEnd();
    }
   glPopAttrib();
  }

	// Added by Brado March 3, 2009 to render wireframe
 my_bool =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] != 1) && 
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] != 1) &&
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] != 1);
  my_bool_0 =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] != -1) && 
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] != -1) &&
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] != -1);

     if(my_bool||my_bool_0)
    {
	glBegin(GL_LINE_STRIP);
	GLfloat diffRefl[] = { 0.0,0.0,0.0,0.0};
      //GLfloat diffRefl[] = {1.0, 0.7, 0.0, 1.00};
      glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl);
      glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl);

      for(ii = 0; ii < 3; ii++) {
	index = i;
	if(ii == 1) index = j;
	if(ii == 2) index = k;
	vert = m_Geometry->m_GeoFrame->triangles[c][index];
	if(m_FlatFlag == 2) {
	  norm[0] = m_Geometry->m_GeoFrame->normals[vert][0];
	  norm[1] = m_Geometry->m_GeoFrame->normals[vert][1];
	  norm[2] = m_Geometry->m_GeoFrame->normals[vert][2];
	  if (my_bool_0) {
	    norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
	  }
	}
	glNormal3d(norm[0], norm[1], norm[2]);
	glColor3d(0.0,0.0,0.0);
	glVertex3d(
		   m_Geometry->m_GeoFrame->verts[vert][0],
		   m_Geometry->m_GeoFrame->verts[vert][1],
		   m_Geometry->m_GeoFrame->verts[vert][2]);
      }

      vert = m_Geometry->m_GeoFrame->triangles[c][i];
      glNormal3d(norm[0], norm[1], norm[2]);
      glColor3d(0.0,0.0,0.0);
      glVertex3d(
		 m_Geometry->m_GeoFrame->verts[vert][0],
		 m_Geometry->m_GeoFrame->verts[vert][1],
		 m_Geometry->m_GeoFrame->verts[vert][2]);
      glEnd();
	}

}

void GeometryRenderable::geoframe_display_hexa(int c, int normalflag, int wireframe)
{
  float v0[3], v1[3], v2[3], v3[3], v4[3], v5[3], v6[3], v7[3], plane_x;
  int vert, ii, jj, my_bool, my_bool_1, my_bool_2, my_bool_3;

  vert = m_Geometry->m_GeoFrame->quads[6*c][0];
  for(ii = 0; ii < 3; ii++)	v0[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];
  vert = m_Geometry->m_GeoFrame->quads[6*c][1];
  for(ii = 0; ii < 3; ii++)	v3[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];
  vert = m_Geometry->m_GeoFrame->quads[6*c][2];
  for(ii = 0; ii < 3; ii++)	v7[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];
  vert = m_Geometry->m_GeoFrame->quads[6*c][3];
  for(ii = 0; ii < 3; ii++)	v4[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];

  vert = m_Geometry->m_GeoFrame->quads[6*c+1][0];
  for(ii = 0; ii < 3; ii++)	v2[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];
  vert = m_Geometry->m_GeoFrame->quads[6*c+1][1];
  for(ii = 0; ii < 3; ii++)	v1[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];
  vert = m_Geometry->m_GeoFrame->quads[6*c+1][2];
  for(ii = 0; ii < 3; ii++)	v5[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];
  vert = m_Geometry->m_GeoFrame->quads[6*c+1][3];
  for(ii = 0; ii < 3; ii++)	v6[ii] = m_Geometry->m_GeoFrame->verts[vert][ii];

  plane_x = 32.0f;//(float) ((int)biggestDim/2.0);	//32.0;
  my_bool =	(v0[0] <= plane_x) && (v1[0] <= plane_x) && (v2[0] <= plane_x) && (v3[0] <= plane_x) &&
    (v4[0] <= plane_x) && (v5[0] <= plane_x) && (v6[0] <= plane_x) && (v7[0] <= plane_x);
  my_bool_1 = (v0[0] >= plane_x) && (v1[0] > plane_x) && (v2[0] > plane_x) && (v3[0] >= plane_x) &&
    (v4[0] >= plane_x) && (v5[0] > plane_x) && (v6[0] > plane_x) && (v7[0] >= plane_x);
  my_bool_3 = (v0[0] == plane_x) && (v3[0] == plane_x) && (v4[0] == plane_x) && (v7[0] == plane_x);
  /*
    plane_x = 32.0f;//(float) ((int)biggestDim/2.0);	//32.0;
    my_bool =	((v0[0] >= plane_x) && (v1[0] >= plane_x) && (v2[0] >= plane_x) && (v3[0] >= plane_x) &&
    (v4[0] >= plane_x) && (v5[0] >= plane_x) && (v6[0] >= plane_x) && (v7[0] >= plane_x)) ||
    ((v0[2] >= plane_x) && (v1[2] >= plane_x) && (v2[2] >= plane_x) && (v3[2] >= plane_x) &&
    (v4[2] >= plane_x) && (v5[2] >= plane_x) && (v6[2] >= plane_x) && (v7[2] >= plane_x));
    my_bool_1 = ((v0[0] < plane_x) && (v1[0] <= plane_x) && (v2[0] <= plane_x) && (v3[0] < plane_x) &&
    (v4[0] < plane_x) && (v5[0] <= plane_x) && (v6[0] <= plane_x) && (v7[0] < plane_x)) &&
    ((v0[2] <= plane_x) && (v1[2] <= plane_x) && (v2[2] <= plane_x) && (v3[2] <= plane_x) &&
    (v4[2] < plane_x) && (v5[2] < plane_x) && (v6[2] < plane_x) && (v7[2] < plane_x));
    my_bool_3 = ((v1[0] == plane_x) && (v2[0] == plane_x) && (v5[0] == plane_x) && (v6[0] == plane_x)) ||
    ((v0[2] == plane_x) && (v1[2] == plane_x) && (v2[2] == plane_x) && (v3[2] == plane_x));
  */
  if(wireframe == 0) {
    if(my_bool) {
      for(ii = 0; ii < 6; ii++) {
	geoframe_display_tri(0, 1, 2, 6*c+ii, normalflag);
	geoframe_display_tri(2, 3, 0, 6*c+ii, normalflag);
      }
    }
    else {
      if(my_bool_1 == 0 || my_bool_3 == 1) {
	for(ii = 0; ii < 6; ii++) {
	  normalflag = -1;
	  geoframe_display_tri(0, 1, 2, 6*c+ii, normalflag);
	  geoframe_display_tri(2, 3, 0, 6*c+ii, normalflag);
	}
      }
    }
  }
  else {
    for (ii = 0; ii < 6; ii++) {
      glBegin(GL_LINE_STRIP);

      my_bool_2 =	abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[6*c+ii][0]]) == 1 && 
	abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[6*c+ii][1]]) == 1 &&
	abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[6*c+ii][2]]) == 1 &&
	abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[6*c+ii][3]]) == 1;
      if(my_bool) {
	if(my_bool_2) {
	  for(jj = 0; jj < 4; jj++) {
	    vert = m_Geometry->m_GeoFrame->quads[6*c+ii][jj];
	    glVertex3d(m_Geometry->m_GeoFrame->verts[vert][0], m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	  }
	  vert = m_Geometry->m_GeoFrame->quads[6*c+ii][0];
	  glVertex3d(m_Geometry->m_GeoFrame->verts[vert][0], m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	}
      }
      else {
	if(my_bool_1 == 0 || my_bool_3 == 1) {
	  for(jj = 0; jj < 4; jj++) {
	    vert = m_Geometry->m_GeoFrame->quads[6*c+ii][jj];
	    if(m_Geometry->m_GeoFrame->verts[vert][0] > plane_x)
	      //glVertex3d(plane_x, m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	      glVertex3d(m_Geometry->m_GeoFrame->verts[vert][0], m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	    else
	      glVertex3d(m_Geometry->m_GeoFrame->verts[vert][0], m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	  }
	  vert = m_Geometry->m_GeoFrame->quads[6*c+ii][0];
	  if(m_Geometry->m_GeoFrame->verts[vert][0] > plane_x)
	    //glVertex3d(plane_x, m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	    glVertex3d(m_Geometry->m_GeoFrame->verts[vert][0], m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	  else
	    glVertex3d(m_Geometry->m_GeoFrame->verts[vert][0], m_Geometry->m_GeoFrame->verts[vert][1], m_Geometry->m_GeoFrame->verts[vert][2]);
	}
      }
      glEnd();
    }
  }

}

void GeometryRenderable::geoframe_display_tri00(int i, int j, int k, int c, int normalflag, int wire_flag, int num_eq)
{
  float v1[3], v2[3], norm[3];
  int vert, ii, index, my_bool, my_bool_0 = 0, my_bool_1, my_bool_2;

  int plane_x = 32;//(float) ((int)biggestDim/2.0);	//32.0; 
  int plane_z = 48; 
  my_bool =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == 1) && 
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == 1) &&
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == 1);
  my_bool_0 =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == -1) && 
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == -1) &&
    (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == -1);

  my_bool_1 = (m_Geometry->m_GeoFrame->verts[m_Geometry->m_GeoFrame->triangles[c][i]][0] == plane_x) &&
    (m_Geometry->m_GeoFrame->verts[m_Geometry->m_GeoFrame->triangles[c][j]][0] == plane_x) &&
    (m_Geometry->m_GeoFrame->verts[m_Geometry->m_GeoFrame->triangles[c][k]][0] == plane_x) && (num_eq == 3);

  my_bool_2 = (m_Geometry->m_GeoFrame->verts[m_Geometry->m_GeoFrame->triangles[c][i]][2] == plane_z) &&
    (m_Geometry->m_GeoFrame->verts[m_Geometry->m_GeoFrame->triangles[c][j]][2] == plane_z) &&
    (m_Geometry->m_GeoFrame->verts[m_Geometry->m_GeoFrame->triangles[c][k]][2] == plane_z) && (num_eq == -3);
  //if(my_bool || abs(num_eq) == 3) {
  if(my_bool || my_bool_0 || my_bool_1 || my_bool_2) {

    vert = m_Geometry->m_GeoFrame->triangles[c][i];
    v1[0] = v2[0] = -m_Geometry->m_GeoFrame->verts[vert][0];
    v1[1] = v2[1] = -m_Geometry->m_GeoFrame->verts[vert][1];
    v1[2] = v2[2] = -m_Geometry->m_GeoFrame->verts[vert][2];
						
    vert = m_Geometry->m_GeoFrame->triangles[c][j];
    v1[0] += m_Geometry->m_GeoFrame->verts[vert][0];
    v1[1] += m_Geometry->m_GeoFrame->verts[vert][1];
    v1[2] += m_Geometry->m_GeoFrame->verts[vert][2];
						
    vert = m_Geometry->m_GeoFrame->triangles[c][k];
    v2[0] += m_Geometry->m_GeoFrame->verts[vert][0];
    v2[1] += m_Geometry->m_GeoFrame->verts[vert][1];
    v2[2] += m_Geometry->m_GeoFrame->verts[vert][2];

    cross(norm, v1, v2);

    // normal flipping
    if (normalflag == 1 && my_bool) {
      norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
    }
    if (m_Geometry->m_GeoFrame->bound_tri[c] == 1) {
      norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
    }
    /*
      if((my_bool == 0 || my_bool_0 == 1) && num_eq == 3) {
      norm[0] =  1.0;		norm[1] =  0.0;		norm[2] =  0.0;
      }
      if((my_bool == 0 || my_bool_0 == 1) && num_eq == -3) {
      norm[0] =  0.0;		norm[1] =  0.0;		norm[2] =  1.0;
      }
    */
    vert = m_Geometry->m_GeoFrame->triangles[c][i];
    v1[0] = m_Geometry->m_GeoFrame->verts[vert][0];
    v1[1] = m_Geometry->m_GeoFrame->verts[vert][1];
    v1[2] = m_Geometry->m_GeoFrame->verts[vert][2];

    if(my_bool_0) {
      //if(my_bool && v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2] < 90000.0) {
      GLfloat diffRefl[] = {1.0, 0.0, 0.0, 1.00};
      //GLfloat diffRefl[] = {1.0, 0.7, 0.0, 1.00};
      glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl);
      glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl);
    }
    else {
      //GLfloat diffRefl_1[] = {0.0, 0.6, 0.5, 1.00};
      GLfloat diffRefl_1[] = {1.0f, 0.0f, 0.0f, 1.00f};
      glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
      glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl_1);
    }
    if(wire_flag == 0) {
      glBegin(GL_TRIANGLES);
      for(ii = 0; ii < 3; ii++) {
	index = i;
	if(ii == 1) index = j;
	if(ii == 2) index = k;
	vert = m_Geometry->m_GeoFrame->triangles[c][index];
	if(m_FlatFlag == 2) {
	  std::cout<<"Flat Flag: "<<m_FlatFlag<<std::endl;
	  norm[0] = m_Geometry->m_GeoFrame->normals[vert][0];
	  norm[1] = m_Geometry->m_GeoFrame->normals[vert][1];
	  norm[2] = m_Geometry->m_GeoFrame->normals[vert][2];
	  if (my_bool_0) {
	    norm[0] = -norm[0];		norm[1] = -norm[1];		norm[2] = -norm[2];
	  }
	}
	if((my_bool == 0 || my_bool_0 == 1) && num_eq == 3) {
	  norm[0] =  1.0;		norm[1] =  0.0;		norm[2] =  0.0;
	}
	if((my_bool == 0 || my_bool_0 == 1) && num_eq == -3) {
	  norm[0] =  0.0;		norm[1] =  0.0;		norm[2] =  1.0;
	}
	glNormal3d(norm[0], norm[1], norm[2]);
	glVertex3d(
		   m_Geometry->m_GeoFrame->verts[vert][0],
		   m_Geometry->m_GeoFrame->verts[vert][1],
		   m_Geometry->m_GeoFrame->verts[vert][2]);
      }
      glEnd();
    }
    else {
      glBegin(GL_LINE_STRIP);

      for(ii = 0; ii < 3; ii++) {
	index = i;
	if(ii == 1) index = j;
	if(ii == 2) index = k;
	vert = m_Geometry->m_GeoFrame->triangles[c][index];
	if(m_FlatFlag == 2) {
	  norm[0] = m_Geometry->m_GeoFrame->normals[vert][0];
	  norm[1] = m_Geometry->m_GeoFrame->normals[vert][1];
	  norm[2] = m_Geometry->m_GeoFrame->normals[vert][2];
	}
	if((my_bool == 0 || my_bool_0 == 1) && num_eq == 3) {
	  norm[0] =  1.0;		norm[1] =  0.0;		norm[2] =  0.0;
	}
	if((my_bool == 0 || my_bool_0 == 1) && num_eq == -3) {
	  norm[0] =  0.0;		norm[1] =  0.0;		norm[2] =  1.0;
	}
	glNormal3d(norm[0], norm[1], norm[2]);
	glVertex3d(
		   m_Geometry->m_GeoFrame->verts[vert][0],
		   m_Geometry->m_GeoFrame->verts[vert][1],
		   m_Geometry->m_GeoFrame->verts[vert][2]);
      }

      vert = m_Geometry->m_GeoFrame->triangles[c][i];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(
		 m_Geometry->m_GeoFrame->verts[vert][0],
		 m_Geometry->m_GeoFrame->verts[vert][1],
		 m_Geometry->m_GeoFrame->verts[vert][2]);
      glEnd();
    }

  }
}

void GeometryRenderable::geoframe_display_tri_cross(int i, int j, int k, float t0, float t1, float t2, int c, int normalflag)
{
  float v1[3], v2[3], norm[3], t, tt;
  int vert, ii, index;

  t = 1.0; tt = 1.0;
  if(t0 != 1.0f) { t = 0.8f; tt = 0.9f;}
  vert = m_Geometry->m_GeoFrame->quads[c][i];
  v1[0] = v2[0] = -((m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0);
  v1[1] = v2[1] = -((m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0);
  v1[2] = v2[2] = -((m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0);
					
  t = 1.0; tt = 1.0;
  if(t1 != 1.0) { t = 0.8f; tt = 0.9f;}
  vert = m_Geometry->m_GeoFrame->quads[c][j];
  v1[0] += (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0;
  v1[1] += (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0;
  v1[2] += (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0;
					
  t = 1.0; tt = 1.0;
  if(t2 != 1.0f) { t = 0.8f; tt = 0.9f;}
  vert = m_Geometry->m_GeoFrame->quads[c][k];
  v2[0] += (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0;
  v2[1] += (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0;
  v2[2] += (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0;
					
  cross(norm, v1, v2);
					
  // normal flipping
  if (normalflag==1) {
    norm[0]=-norm[0];
    norm[1]=-norm[1];
    norm[2]=-norm[2];
  }

  glBegin(GL_TRIANGLES);

  for(ii = 0; ii < 3; ii++) {
    index = i;		
    t = 1.0f;  tt = 1.0f; if(t0 != 1.0f) { t = 0.8f; tt = 0.9f;}
    if(ii == 1) {index = j; t = 1.0f;  tt = 1.0f; if(t1 != 1.0f) { t = 0.8f; tt = 0.9f;}}
    if(ii == 2) {index = k;	t = 1.0f;  tt = 1.0f; if(t2 != 1.0f) { t = 0.8f; tt = 0.9f;}}
    vert = m_Geometry->m_GeoFrame->quads[c][index];
    glNormal3d(norm[0], norm[1], norm[2]);
    glVertex3d(
	       (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0,
	       (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0,
	       (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0);
  }
  glEnd();
}

void GeometryRenderable::geoframe_display_tri_cross0(int i, int j, int k, float t0, float t1, float t2, int c, int normalflag)
{
  float v1[3], v2[3], norm[3], t, tt;
  int vert, ii, index;

  t = 1.0f; tt = 1.0f;
  if(t0 != 1.0f) { t = 0.8f; tt = 0.9f;}
  vert = m_Geometry->m_GeoFrame->quads[c][i];
  v1[0] = v2[0] = -((m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0);
  v1[1] = v2[1] = -((m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0);
  v1[2] = v2[2] = -((m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0);
					
  t = 1.0f; tt = 1.0f;
  if(t1 != 1.0f) { t = 0.8f; tt = 0.9f;}
  vert = m_Geometry->m_GeoFrame->quads[c][j];
  v1[0] += (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0;
  v1[1] += (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0;
  v1[2] += (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0;
					
  t = 1.0f; tt = 1.0f;
  if(t2 != 1.0f) { t = 0.8f; tt = 0.9f;}
  vert = m_Geometry->m_GeoFrame->quads[c][k];
  v2[0] += (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0;
  v2[1] += (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0;
  v2[2] += (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0;
					
  cross(norm, v1, v2);
					
  // normal flipping
  if (normalflag==1) {
    norm[0]=-norm[0];
    norm[1]=-norm[1];
    norm[2]=-norm[2];
  }

  glBegin(GL_LINE_STRIP);
  glColor3d(0.2,0.2,0.2);

  for(ii = 0; ii < 3; ii++) {
    index = i;		
    t = 1.0f;  tt = 1.0f; if(t0 != 1.0f) { t = 0.8f; tt = 0.9f;}
    if(ii == 1) {index = j; t = 1.0f;  tt = 1.0f; if(t1 != 1.0f) { t = 0.8f; tt = 0.9f;}}
    if(ii == 2) {index = k;	t = 1.0f;  tt = 1.0f; if(t2 != 1.0f) { t = 0.8f; tt = 0.9f;}}
    vert = m_Geometry->m_GeoFrame->quads[c][index];
    glNormal3d(norm[0], norm[1], norm[2]);
    glVertex3d(
	       (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*t+32.0,
	       (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*t+32.0,
	       (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*tt+32.0);
  }
  glEnd();
}

void GeometryRenderable::geoframe_display_prism(int i, int j, int k, int c, int normalflag)
{
  float v1[3], v2[3], norm[3];
  int vert, ii, index;

  vert = m_Geometry->m_GeoFrame->quads[c][i];
  v1[0] = v2[0] = -m_Geometry->m_GeoFrame->verts[vert][0];
  v1[1] = v2[1] = -m_Geometry->m_GeoFrame->verts[vert][1];
  v1[2] = v2[2] = -m_Geometry->m_GeoFrame->verts[vert][2];
					
  vert = m_Geometry->m_GeoFrame->quads[c][j];
  v1[0] += m_Geometry->m_GeoFrame->verts[vert][0];
  v1[1] += m_Geometry->m_GeoFrame->verts[vert][1];
  v1[2] += m_Geometry->m_GeoFrame->verts[vert][2];
					
  vert = m_Geometry->m_GeoFrame->quads[c][k];
  v2[0] += m_Geometry->m_GeoFrame->verts[vert][0];
  v2[1] += m_Geometry->m_GeoFrame->verts[vert][1];
  v2[2] += m_Geometry->m_GeoFrame->verts[vert][2];
					
  cross(norm, v1, v2);
					
  // normal flipping
  if (normalflag==1) {
    norm[0]=-norm[0];
    norm[1]=-norm[1];
    norm[2]=-norm[2];
  }

  glBegin(GL_TRIANGLES);
  for(ii = 0; ii < 3; ii++) {
    index = i;
    if(ii == 1) index = j;
    if(ii == 2) index = k;
    vert = m_Geometry->m_GeoFrame->quads[c][index];
    glNormal3d(norm[0], norm[1], norm[2]);
    glVertex3d(
	       m_Geometry->m_GeoFrame->verts[vert][0],
	       m_Geometry->m_GeoFrame->verts[vert][1],
	       m_Geometry->m_GeoFrame->verts[vert][2]);
  }

  for(ii = 0; ii < 3; ii++) {
    index = i;
    if(ii == 1) index = j;
    if(ii == 2) index = k;
    vert = m_Geometry->m_GeoFrame->quads[c][index];
    glNormal3d(-norm[0], -norm[1], -norm[2]);
    glVertex3d(
	       (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*0.8+32.0,
	       (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*0.8+32.0,
	       (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*0.9+32.0);
  }
  glEnd();

  if(m_Geometry->m_GeoFrame->verts[vert][2] > 32.0) {
    geoframe_display_tri_cross(i, j, j, 1.0f, 1.0f, 0.8f, c, -normalflag);
    geoframe_display_tri_cross(i, j, i, 1.0f, 0.8f, 0.8f, c, -normalflag);

    geoframe_display_tri_cross(j, k, j, 1.0f, 1.0f, 0.8f, c, -normalflag);
    geoframe_display_tri_cross(k, k, j, 1.0f, 0.8f, 0.8f, c, -normalflag);

    geoframe_display_tri_cross(k, i, k, 1.0f, 1.0f, 0.8f, c, -normalflag);
    geoframe_display_tri_cross(i, i, k, 1.0f, 0.8f, 0.8f, c, -normalflag);
  }

}

void GeometryRenderable::geoframe_display_tri_v(float* v0, float* v1, float* v2, int normalflag, int wire_flag)
{
  float vv01[3], vv02[3], norm[3];
  int ii;

  for(ii = 0; ii < 3; ii++) {
    vv01[ii] = v1[ii] - v0[ii];
    vv02[ii] = v2[ii] - v0[ii];
  }	
  cross(norm, vv01, vv02);
					
  // normal flipping
  if (normalflag==1) {
    for(ii = 0; ii < 3; ii++) norm[ii] = - norm[ii];
  }

  if(wire_flag == 1)	{
    glBegin(GL_LINE_STRIP);
    glNormal3d(norm[0], norm[1], norm[2]);
    glVertex3d(v0[0], v0[1], v0[2]);
    glVertex3d(v1[0], v1[1], v1[2]);
    glVertex3d(v2[0], v2[1], v2[2]);
    glVertex3d(v0[0], v0[1], v0[2]);
    glEnd();
  }
  else {
    glBegin(GL_TRIANGLES);
    glNormal3d(norm[0], norm[1], norm[2]);
    glVertex3d(v0[0], v0[1], v0[2]);
    glVertex3d(v1[0], v1[1], v1[2]);
    glVertex3d(v2[0], v2[1], v2[2]);
    glEnd();
  }
}

void GeometryRenderable::geoframe_display_tri_vv(float* v0, float* v1, float* v2, int c, int normalflag, int wire_flag)
{
  float vv01[3], vv02[3], norm[3];
  int ii, my_bool, my_bool_0, vert;

  if(c == -1) {
    my_bool = 1;	my_bool_0 = 0;
  }
  else {
    my_bool =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == 1) && 
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == 1) &&
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == 1);
    my_bool_0 =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == -1) && 
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == -1) &&
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == -1);
  }

  for(ii = 0; ii < 3; ii++) {
    vv01[ii] = v1[ii] - v0[ii];
    vv02[ii] = v2[ii] - v0[ii];
  }	
  cross(norm, vv01, vv02);
					
  // normal flipping
  if (normalflag == 1) {
    for(ii = 0; ii < 3; ii++) norm[ii] = - norm[ii];
  }

  vert = m_Geometry->m_GeoFrame->triangles[c][0];
  vv01[0] = m_Geometry->m_GeoFrame->verts[vert][0];
  vv01[1] = m_Geometry->m_GeoFrame->verts[vert][1];
  vv01[2] = m_Geometry->m_GeoFrame->verts[vert][2];

  //GLfloat diffRefl_1[] = {0.0, 0.6, 0.5, 1.00};
  GLfloat diffRefl_1[] = {1.0f, 0.7f, 0.0f, 1.00f};
  glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
  glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl_1);

  if(my_bool_0) {
    //if(my_bool && vv01[0]*vv01[0] + vv01[1]*vv01[1] + vv01[2]*vv01[2] < 90000.0) {
    GLfloat diffRefl[] = {1.0, 0.0, 0.0, 1.00};
    //GLfloat diffRefl[] = {1.0, 0.7, 0.0, 1.00};
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl);
    glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl);
  }

  if(wire_flag == 1)	{
    glBegin(GL_LINE_STRIP);
    if(m_FlatFlag == 2 && c != -1) {
      vert = m_Geometry->m_GeoFrame->triangles[c][0];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);

      vert = m_Geometry->m_GeoFrame->triangles[c][1];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v1[0], v1[1], v1[2]);

      vert = m_Geometry->m_GeoFrame->triangles[c][2];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v2[0], v2[1], v2[2]);

      vert = m_Geometry->m_GeoFrame->triangles[c][0];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
    }
    else {
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
      glVertex3d(v1[0], v1[1], v1[2]);
      glVertex3d(v2[0], v2[1], v2[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
    }
    glEnd();
  }
  else {
    glBegin(GL_TRIANGLES);
    if(m_FlatFlag == 2 && c != -1) {
      vert = m_Geometry->m_GeoFrame->triangles[c][0];
      if(m_Geometry->m_GeoFrame->bound_sign[vert] == -1) {
	glNormal3d(-norm[0], -norm[1], -norm[2]);
      }
      else {
	glNormal3d(m_Geometry->m_GeoFrame->normals[vert][0], 
		   m_Geometry->m_GeoFrame->normals[vert][1], m_Geometry->m_GeoFrame->normals[vert][2]);
      }
      glVertex3d(v0[0], v0[1], v0[2]);

      vert = m_Geometry->m_GeoFrame->triangles[c][1];
      if(m_Geometry->m_GeoFrame->bound_sign[vert] == -1) {
	glNormal3d(-norm[0], -norm[1], -norm[2]);
      }
      else {
	glNormal3d(m_Geometry->m_GeoFrame->normals[vert][0], 
		   m_Geometry->m_GeoFrame->normals[vert][1], m_Geometry->m_GeoFrame->normals[vert][2]);
      }
      glVertex3d(v1[0], v1[1], v1[2]);

      vert = m_Geometry->m_GeoFrame->triangles[c][2];
      if(m_Geometry->m_GeoFrame->bound_sign[vert] == -1) {
	glNormal3d(-norm[0], -norm[1], -norm[2]);
      }
      else {
	glNormal3d(m_Geometry->m_GeoFrame->normals[vert][0], 
		   m_Geometry->m_GeoFrame->normals[vert][1], m_Geometry->m_GeoFrame->normals[vert][2]);
      }
      glVertex3d(v2[0], v2[1], v2[2]);
    }
    else {
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
      glVertex3d(v1[0], v1[1], v1[2]);
      glVertex3d(v2[0], v2[1], v2[2]);
    }
    glEnd();
  }
}

void GeometryRenderable::geoframe_display_tri_vv_0(float* v0, float* v1, float* v2, int c, int normalflag, int wire_flag)
{
  float vv01[3], vv02[3], norm[3];
  int ii, my_bool, my_bool_0, vert;

  if(c == -1) {
    my_bool = 1;	my_bool_0 = 0;
  }
  else {
    my_bool =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == 1) && 
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == 1) &&
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == 1);
    my_bool_0 =	(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]] == -1) && 
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]] == -1) &&
      (m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]] == -1);
  }

  for(ii = 0; ii < 3; ii++) {
    vv01[ii] = v1[ii] - v0[ii];
    vv02[ii] = v2[ii] - v0[ii];
  }	
  cross(norm, vv01, vv02);
					
  // normal flipping
  if (normalflag==1) {
    for(ii = 0; ii < 3; ii++) norm[ii] = - norm[ii];
  }

  vert = m_Geometry->m_GeoFrame->triangles[c][0];
  vv01[0] = m_Geometry->m_GeoFrame->verts[vert][0];
  vv01[1] = m_Geometry->m_GeoFrame->verts[vert][1];
  vv01[2] = m_Geometry->m_GeoFrame->verts[vert][2];

  //GLfloat diffRefl_1[] = {0.0, 0.6, 0.5, 1.00};
  GLfloat diffRefl_1[] = {1.0f, 0.7f, 0.0f, 1.00f};
  glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl_1);
  glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl_1);

  //if(my_bool_0) {
  if(my_bool && vv01[0]*vv01[0] + vv01[1]*vv01[1] + vv01[2]*vv01[2] < 90000.0f) {
    GLfloat diffRefl[] = {1.0, 0.0, 0.0, 1.00};
    //GLfloat diffRefl[] = {1.0, 0.7, 0.0, 1.00};
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffRefl);
    glMaterialfv(GL_BACK, GL_DIFFUSE, diffRefl);
  }

  if(wire_flag == 1)	{
    glBegin(GL_LINE_STRIP);
    /*		if(m_FlatFlag == 2 && c != -1) {
      vert = m_Geometry->m_GeoFrame->triangles[c][0];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
      vert = m_Geometry->m_GeoFrame->triangles[c][1];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v1[0], v1[1], v1[2]);
      vert = m_Geometry->m_GeoFrame->triangles[c][2];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v2[0], v2[1], v2[2]);
      vert = m_Geometry->m_GeoFrame->triangles[c][0];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
      }
      else {
    */			glNormal3d(norm[0], norm[1], norm[2]);
    glVertex3d(v0[0], v0[1], v0[2]);
    glVertex3d(v1[0], v1[1], v1[2]);
    glVertex3d(v2[0], v2[1], v2[2]);
    glVertex3d(v0[0], v0[1], v0[2]);
    //		}
    glEnd();
  }
  else {
    glBegin(GL_TRIANGLES);
    /*		if(m_FlatFlag == 2 && c != -1) {
      vert = m_Geometry->m_GeoFrame->triangles[c][0];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
      vert = m_Geometry->m_GeoFrame->triangles[c][1];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v1[0], v1[1], v1[2]);
      vert = m_Geometry->m_GeoFrame->triangles[c][2];
      for(ii = 0; ii < 3; ii++)	norm[ii] = m_Geometry->m_GeoFrame->normals[vert][ii];
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v2[0], v2[1], v2[2]);
      }
      else {
    */			glNormal3d(norm[0], norm[1], norm[2]);
    glVertex3d(v0[0], v0[1], v0[2]);
    glVertex3d(v1[0], v1[1], v1[2]);
    glVertex3d(v2[0], v2[1], v2[2]);
    //		}
    glEnd();
  }
}


void GeometryRenderable::geoframe_display_quad_v(float* v0, float* v1, float* v2, float* v3, int normalflag, int wire_flag)
{
  float vv01[3], vv02[3], norm[3];
  int ii;

  int my_bool_0 = (fabs(v1[0] - v2[0]) > 0.01) || (fabs(v1[1] - v2[1]) > 0.01)
    || (fabs(v1[2] - v2[2]) > 0.01);
  int my_bool_1 = (fabs(v0[0] - v3[0]) > 0.01) || (fabs(v0[1] - v3[1]) > 0.01)
    || (fabs(v0[2] - v3[2]) > 0.01);
  int my_bool_2 = (fabs(v1[0] - v2[0]) <= 0.01) && (fabs(v1[1] - v2[1]) <= 0.01)
    && (fabs(v1[2] - v2[2]) <= 0.01);
  int my_bool_3 = (fabs(v0[0] - v3[0]) <= 0.01) && (fabs(v0[1] - v3[1]) <= 0.01)
    && (fabs(v0[2] - v3[2]) <= 0.01);
  if( my_bool_0 || my_bool_1) {
    for(ii = 0; ii < 3; ii++) {
      if(v1[0] != v2[0] && v1[0] != v2[0] && v1[0] != v2[0]) {
	vv01[ii] = v1[ii] - v0[ii];
	vv02[ii] = v2[ii] - v0[ii];
      }
      else {
	vv01[ii] = v1[ii] - v0[ii];
	vv02[ii] = v3[ii] - v0[ii];
      }
    }	
    cross(norm, vv01, vv02);
						
    // normal flipping
    if (normalflag==1) {
      for(ii = 0; ii < 3; ii++) norm[ii] = - norm[ii];
    }

    if(wire_flag == 1)	glBegin(GL_LINE_STRIP);
    else glBegin(GL_QUADS);

    glNormal3d(norm[0], norm[1], norm[2]);
    if(my_bool_2) {
      glVertex3d(v0[0], v0[1], v0[2]);
      glVertex3d(v1[0], v1[1], v1[2]);
      glVertex3d(v3[0], v3[1], v3[2]);
    }
    else if(my_bool_3) {
      glVertex3d(v0[0], v0[1], v0[2]);
      glVertex3d(v1[0], v1[1], v1[2]);
      glVertex3d(v2[0], v2[1], v2[2]);
    }
    else {
      glVertex3d(v0[0], v0[1], v0[2]);
      glVertex3d(v1[0], v1[1], v1[2]);
      glVertex3d(v2[0], v2[1], v2[2]);
      glVertex3d(v3[0], v3[1], v3[2]);
    }

    if(wire_flag) {
      glNormal3d(norm[0], norm[1], norm[2]);
      glVertex3d(v0[0], v0[1], v0[2]);
    }
    glEnd();
  }
}

void GeometryRenderable::geoframe_display_permute_1(float* v0, float* v1, float* v2, float* v3, float plane_x) {
  float vv0[3], vv1[3], vv2[3], vv3[3];
  int ii;

  for(ii = 0; ii < 3; ii++) {
    vv0[ii] = v0[ii];	vv1[ii] = v1[ii];
    vv2[ii] = v2[ii];	vv3[ii] = v3[ii];
  }
  if(vv0[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv2[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv1[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv2[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv1[ii];
    }
  }
  if(vv2[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv0[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv2[ii];
    }
  }
}

void GeometryRenderable::geoframe_display_permute_1_z(float* v0, float* v1, float* v2, float* v3, float plane_z) {
  float vv0[3], vv1[3], vv2[3], vv3[3];
  int ii;

  for(ii = 0; ii < 3; ii++) {
    vv0[ii] = v0[ii];	vv1[ii] = v1[ii];
    vv2[ii] = v2[ii];	vv3[ii] = v3[ii];
  }
  if(vv0[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv2[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv1[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv2[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv1[ii];
    }
  }
  if(vv2[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv0[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv2[ii];
    }
  }
}

void GeometryRenderable::geoframe_display_permute_2(float* v0, float* v1, float* v2, float* v3, float plane_x) {
  float vv0[3], vv1[3], vv2[3], vv3[3];
  int ii;

  for(ii = 0; ii < 3; ii++) {
    vv0[ii] = v0[ii];	vv1[ii] = v1[ii];
    vv2[ii] = v2[ii];	vv3[ii] = v3[ii];
  }
  if(vv0[0] <= plane_x && vv2[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv2[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv1[ii];
    }
  }
  if(vv0[0] <= plane_x && vv3[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv1[ii];	v3[ii] = vv2[ii];
    }
  }
  if(vv2[0] <= plane_x && vv1[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv2[ii];	v1[ii] = vv1[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv1[0] <= plane_x && vv3[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv2[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv2[0] <= plane_x && vv3[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv2[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv0[ii];	v3[ii] = vv1[ii];
    }
  }
}

void GeometryRenderable::geoframe_display_permute_2_z(float* v0, float* v1, float* v2, float* v3, float plane_z) {
  float vv0[3], vv1[3], vv2[3], vv3[3];
  int ii;

  for(ii = 0; ii < 3; ii++) {
    vv0[ii] = v0[ii];	vv1[ii] = v1[ii];
    vv2[ii] = v2[ii];	vv3[ii] = v3[ii];
  }
  if(vv0[2] <= plane_z && vv2[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv2[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv1[ii];
    }
  }
  if(vv0[2] <= plane_z && vv3[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv1[ii];	v3[ii] = vv2[ii];
    }
  }
  if(vv2[2] <= plane_z && vv1[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv2[ii];	v1[ii] = vv1[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv1[2] <= plane_z && vv3[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv2[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv2[2] <= plane_z && vv3[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv2[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv0[ii];	v3[ii] = vv1[ii];
    }
  }
}

void GeometryRenderable::geoframe_display_permute_3(float* v0, float* v1, float* v2, float* v3, float plane_x) {
  float vv0[3], vv1[3], vv2[3], vv3[3];
  int ii;

  for(ii = 0; ii < 3; ii++) {
    vv0[ii] = v0[ii];	vv1[ii] = v1[ii];
    vv2[ii] = v2[ii];	vv3[ii] = v3[ii];
  }
  if(vv1[0] <= plane_x && vv2[0] <= plane_x && vv3[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv2[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv0[0] <= plane_x && vv2[0] <= plane_x && vv3[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv2[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv1[ii];
    }
  }
  if(vv0[0] <= plane_x && vv1[0] <= plane_x && vv3[0] <= plane_x) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv0[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv2[ii];
    }
  }
}

void GeometryRenderable::geoframe_display_permute_3_z(float* v0, float* v1, float* v2, float* v3, float plane_z) {
  float vv0[3], vv1[3], vv2[3], vv3[3];
  int ii;

  for(ii = 0; ii < 3; ii++) {
    vv0[ii] = v0[ii];	vv1[ii] = v1[ii];
    vv2[ii] = v2[ii];	vv3[ii] = v3[ii];
  }
  if(vv1[2] <= plane_z && vv2[2] <= plane_z && vv3[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv3[ii];
      v2[ii] = vv2[ii];	v3[ii] = vv0[ii];
    }
  }
  if(vv0[2] <= plane_z && vv2[2] <= plane_z && vv3[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv0[ii];	v1[ii] = vv2[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv1[ii];
    }
  }
  if(vv0[2] <= plane_z && vv1[2] <= plane_z && vv3[2] <= plane_z) {
    for(ii = 0; ii < 3; ii++) {
      v0[ii] = vv1[ii];	v1[ii] = vv0[ii];
      v2[ii] = vv3[ii];	v3[ii] = vv2[ii];
    }
  }
}

void GeometryRenderable::geoframe_display_1(int* vert_bound, int c, float* v0, float* v1, float* v2, float* v3, float plane_x,
			   int normalflag, int wire_flag) {

  float vv0[3], vv1[3], vv2[3], ratio_0, ratio_1, ratio_2;
  int dummy = 0;
  dummy = normalflag;
  vv0[0] = plane_x;
  ratio_0 = (plane_x - v0[0]) / (v3[0] - v0[0]);
  vv0[1] = v0[1] + ratio_0 * (v3[1] - v0[1]);
  vv0[2] = v0[2] + ratio_0 * (v3[2] - v0[2]);

  vv1[0] = plane_x;
  ratio_1 = (plane_x - v1[0]) / (v3[0] - v1[0]);
  vv1[1] = v1[1] + ratio_1 * (v3[1] - v1[1]);
  vv1[2] = v1[2] + ratio_1 * (v3[2] - v1[2]);

  vv2[0] = plane_x;
  ratio_2 = (plane_x - v2[0]) / (v3[0] - v2[0]);
  vv2[1] = v2[1] + ratio_2 * (v3[1] - v2[1]);
  vv2[2] = v2[2] + ratio_2 * (v3[2] - v2[2]);

  geoframe_display_tri_vv(vv0, vv2, vv1, -1, 1, wire_flag);
  if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
    geoframe_display_tri_vv(vv1, vv2, v3,  4*c+1, 1, wire_flag);
  if(abs(vert_bound[0]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
    geoframe_display_tri_vv(vv2, vv0, v3,  4*c+2, 1, wire_flag);
  if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3)
    geoframe_display_tri_vv(vv0, vv1, v3,  4*c+3, 1, wire_flag);
}

void GeometryRenderable::geoframe_display_1_z(int* vert_bound, int c, float* v0, float* v1, float* v2, float* v3, float plane_z,
			     int normalflag, int wire_flag) {

  float vv0[3], vv1[3], vv2[3], ratio_0, ratio_1, ratio_2;
  int dummy = 0;
  dummy = normalflag;

  vv0[2] = plane_z;
  ratio_0 = (plane_z - v0[2]) / (v3[2] - v0[2]);
  vv0[1] = v0[1] + ratio_0 * (v3[1] - v0[1]);
  vv0[0] = v0[0] + ratio_0 * (v3[0] - v0[0]);

  vv1[2] = plane_z;
  ratio_1 = (plane_z - v1[2]) / (v3[2] - v1[2]);
  vv1[1] = v1[1] + ratio_1 * (v3[1] - v1[1]);
  vv1[0] = v1[0] + ratio_1 * (v3[0] - v1[0]);

  vv2[2] = plane_z;
  ratio_2 = (plane_z - v2[2]) / (v3[2] - v2[2]);
  vv2[1] = v2[1] + ratio_2 * (v3[1] - v2[1]);
  vv2[0] = v2[0] + ratio_2 * (v3[0] - v2[0]);

  geoframe_display_tri_vv(vv0, vv2, vv1, -1, 1, wire_flag);
  if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
    geoframe_display_tri_vv(vv1, vv2, v3,  4*c+1, 1, wire_flag);
  if(abs(vert_bound[0]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
    geoframe_display_tri_vv(vv2, vv0, v3,  4*c+2, 1, wire_flag);
  if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3)
    geoframe_display_tri_vv(vv0, vv1, v3,  4*c+3, 1, wire_flag);
}

void GeometryRenderable::geoframe_display_2(int* vert_bound, int c, float* v0, float* v1, float* v2, float* v3, float plane_x,
			   int normalflag, int wire_flag) {

  float vv0[3], vv1[3], vv2[3], vv3[3], ratio_0, ratio_1, ratio_2, ratio_3;
  int dummy = 0;
  dummy = normalflag;

  vv0[0] = plane_x;
  ratio_0 = (plane_x - v0[0]) / (v3[0] - v0[0]);
  vv0[1] = v0[1] + ratio_0 * (v3[1] - v0[1]);
  vv0[2] = v0[2] + ratio_0 * (v3[2] - v0[2]);

  vv1[0] = plane_x;
  ratio_1 = (plane_x - v1[0]) / (v3[0] - v1[0]);
  vv1[1] = v1[1] + ratio_1 * (v3[1] - v1[1]);
  vv1[2] = v1[2] + ratio_1 * (v3[2] - v1[2]);

  vv2[0] = plane_x;
  ratio_2 = (plane_x - v0[0]) / (v2[0] - v0[0]);
  vv2[1] = v0[1] + ratio_2 * (v2[1] - v0[1]);
  vv2[2] = v0[2] + ratio_2 * (v2[2] - v0[2]);

  vv3[0] = plane_x;
  ratio_3 = (plane_x - v1[0]) / (v2[0] - v1[0]);
  vv3[1] = v1[1] + ratio_3 * (v2[1] - v1[1]);
  vv3[2] = v1[2] + ratio_3 * (v2[2] - v1[2]);

  if(ratio_0 != 0.0 && ratio_1 == 0.0) {
    geoframe_display_tri_vv(vv0, v1,  vv2, -1, 1, wire_flag);
    if(abs(vert_bound[0]) == 1)
      geoframe_display_tri_vv(vv0, vv2, v0,  4*c+2, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv0, v0,  v1,  4*c+3, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3)
      geoframe_display_tri_vv(vv2, v1,  v0,  4*c, 1, wire_flag);
  }
  if(ratio_0 == 0.0 && ratio_1 != 0.0) {
    geoframe_display_tri_vv(vv1, vv3, v0,  -1, 1, wire_flag);
    if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv1, v1,  vv3, 4*c+1, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv1, v0,  v1,  4*c+3, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3)
      geoframe_display_tri_vv(vv3, v1,  v0,  4*c, 1, wire_flag);
  }
  if(ratio_0 != 0.0 && ratio_1 != 0.0) {
    geoframe_display_tri_vv(vv0, vv1, vv2, -1, 1, wire_flag);
    geoframe_display_tri_vv(vv1, vv3, vv2, -1, 1, wire_flag);
    if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv3, vv1, v1,  4*c+1, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv0, vv2, v0,  4*c+2, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3) {
      geoframe_display_tri_vv(vv1, vv0, v0,  4*c+3, 1, wire_flag);
      geoframe_display_tri_vv(vv1, v0,  v1,  4*c+3, 1, wire_flag);
    }
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3) {
      geoframe_display_tri_vv(vv2, vv3, v1,  4*c, 1, wire_flag);
      geoframe_display_tri_vv(vv2, v1,  v0,  4*c, 1, wire_flag);
    }
  }
}

void GeometryRenderable::geoframe_display_2_z(int* vert_bound, int c, float* v0, float* v1, float* v2, float* v3, float plane_z,
			     int normalflag, int wire_flag) {

  float vv0[3], vv1[3], vv2[3], vv3[3], ratio_0, ratio_1, ratio_2, ratio_3;
  int dummy = 0;
  dummy = normalflag;

  vv0[2] = plane_z;
  ratio_0 = (plane_z - v0[2]) / (v3[2] - v0[2]);
  vv0[1] = v0[1] + ratio_0 * (v3[1] - v0[1]);
  vv0[0] = v0[0] + ratio_0 * (v3[0] - v0[0]);

  vv1[2] = plane_z;
  ratio_1 = (plane_z - v1[2]) / (v3[2] - v1[2]);
  vv1[1] = v1[1] + ratio_1 * (v3[1] - v1[1]);
  vv1[0] = v1[0] + ratio_1 * (v3[0] - v1[0]);

  vv2[2] = plane_z;
  ratio_2 = (plane_z - v0[2]) / (v2[2] - v0[2]);
  vv2[1] = v0[1] + ratio_2 * (v2[1] - v0[1]);
  vv2[0] = v0[0] + ratio_2 * (v2[0] - v0[0]);

  vv3[2] = plane_z;
  ratio_3 = (plane_z - v1[2]) / (v2[2] - v1[2]);
  vv3[1] = v1[1] + ratio_3 * (v2[1] - v1[1]);
  vv3[0] = v1[0] + ratio_3 * (v2[0] - v1[0]);

  if(ratio_0 != 0.0 && ratio_1 == 0.0) {
    geoframe_display_tri_vv(vv0, v1,  vv2, -1, 1, wire_flag);
    //geoframe_display_tri_v(vv0, vv2, v0,  1, wire_flag);
    //geoframe_display_tri_v(vv0, v0,  v1,  1, wire_flag);
    //geoframe_display_tri_v(vv2, v1,  v0,  1, wire_flag);
    if(abs(vert_bound[0]) == 1)
      geoframe_display_tri_vv(vv0, vv2, v0,  4*c+2, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv0, v0,  v1,  4*c+3, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3)
      geoframe_display_tri_vv(vv2, v1,  v0,  4*c, 1, wire_flag);
  }
  if(ratio_0 == 0.0 && ratio_1 != 0.0) {
    geoframe_display_tri_vv(vv1, vv3, v0,  -1, 1, wire_flag);
    //geoframe_display_tri_v(vv1, v1,  vv3, 1, wire_flag);
    //geoframe_display_tri_v(vv1, v0,  v1,  1, wire_flag);
    //geoframe_display_tri_v(vv3, v1,  v0,  1, wire_flag);
    if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv1, v1,  vv3, 4*c+1, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv1, v0,  v1,  4*c+3, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3)
      geoframe_display_tri_vv(vv3, v1,  v0,  4*c, 1, wire_flag);
  }
  if(ratio_0 != 0.0 && ratio_1 != 0.0) {
    geoframe_display_tri_vv(vv0, vv1, vv2, -1, 1, wire_flag);
    geoframe_display_tri_vv(vv1, vv3, vv2, -1, 1, wire_flag);
    //geoframe_display_tri_v(vv3, vv1, v1,  1, wire_flag);
    //geoframe_display_tri_v(vv0, vv2, v0,  1, wire_flag);
    //geoframe_display_tri_v(vv1, vv0, v0,  1, wire_flag);
    //geoframe_display_tri_v(vv1, v0,  v1,  1, wire_flag);
    //geoframe_display_tri_v(vv2, vv3, v1,  1, wire_flag);
    //geoframe_display_tri_v(vv2, v1,  v0,  1, wire_flag);
    if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv3, vv1, v1,  4*c+1, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3)
      geoframe_display_tri_vv(vv0, vv2, v0,  4*c+2, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[3]) == 3) {
      geoframe_display_tri_vv(vv1, vv0, v0,  4*c+3, 1, wire_flag);
      geoframe_display_tri_vv(vv1, v0,  v1,  4*c+3, 1, wire_flag);
    }
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3) {
      geoframe_display_tri_vv(vv2, vv3, v1,  4*c, 1, wire_flag);
      geoframe_display_tri_vv(vv2, v1,  v0,  4*c, 1, wire_flag);
    }
  }
}

void GeometryRenderable::geoframe_display_3(int* vert_bound, int c, float* v0, float* v1, float* v2, float* v3, float plane_x,
			   int normalflag, int wire_flag) {

  float vv0[3], vv1[3], vv2[3], ratio_0, ratio_1, ratio_2;
  int dummy = 0;
  dummy = normalflag;

  vv0[0] = plane_x;
  ratio_0 = (plane_x - v0[0]) / (v3[0] - v0[0]);
  vv0[1] = v0[1] + ratio_0 * (v3[1] - v0[1]);
  vv0[2] = v0[2] + ratio_0 * (v3[2] - v0[2]);

  vv1[0] = plane_x;
  ratio_1 = (plane_x - v1[0]) / (v3[0] - v1[0]);
  vv1[1] = v1[1] + ratio_1 * (v3[1] - v1[1]);
  vv1[2] = v1[2] + ratio_1 * (v3[2] - v1[2]);

  vv2[0] = plane_x;
  ratio_2 = (plane_x - v2[0]) / (v3[0] - v2[0]);
  vv2[1] = v2[1] + ratio_2 * (v3[1] - v2[1]);
  vv2[2] = v2[2] + ratio_2 * (v3[2] - v2[2]);

  if(ratio_0 <= 0.001 && ratio_1 <= 0.01 && ratio_2 <= 0.001) {
    //if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 0) {
    geoframe_display_tri_vv(vv0, vv1, vv2, -1, 1, wire_flag);
  }
  else {
    geoframe_display_tri_vv(vv0, vv1, vv2, -1, 1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3)
      geoframe_display_tri_vv(v0,  v2,  v1,  4*c, 1, wire_flag);
    if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3) {
      geoframe_display_tri_vv(v1,  v2,  vv2, 4*c+1, 1, wire_flag);
      geoframe_display_tri_vv(v1,  vv2, vv1, 4*c+1, 1, wire_flag);
    }
    if(abs(vert_bound[0]) + abs(vert_bound[2]) == 2) {
      geoframe_display_tri_vv(v2,  v0,  vv2, 4*c+2, 1, wire_flag);
      geoframe_display_tri_vv(vv2, v0,  vv0, 4*c+2, 1, wire_flag);
    }
    if(abs(vert_bound[0]) + abs(vert_bound[1]) == 2) {
      geoframe_display_tri_vv(v1,  vv0, v0,  4*c+3, 1, wire_flag);
      geoframe_display_tri_vv(v1,  vv1, vv0, 4*c+3, 1, wire_flag);
    }
  }
}

void GeometryRenderable::geoframe_display_3_z(int* vert_bound, int c, float* v0, float* v1, float* v2, float* v3, float plane_z,
			     int normalflag, int wire_flag) {

  float vv0[3], vv1[3], vv2[3], ratio_0, ratio_1, ratio_2;
  int dummy = 0;
  dummy = normalflag;

  vv0[2] = plane_z;
  ratio_0 = (plane_z - v0[2]) / (v3[2] - v0[2]);
  vv0[1] = v0[1] + ratio_0 * (v3[1] - v0[1]);
  vv0[0] = v0[0] + ratio_0 * (v3[0] - v0[0]);

  vv1[2] = plane_z;
  ratio_1 = (plane_z - v1[2]) / (v3[2] - v1[2]);
  vv1[1] = v1[1] + ratio_1 * (v3[1] - v1[1]);
  vv1[0] = v1[0] + ratio_1 * (v3[0] - v1[0]);

  vv2[2] = plane_z;
  ratio_2 = (plane_z - v2[2]) / (v3[2] - v2[2]);
  vv2[1] = v2[1] + ratio_2 * (v3[1] - v2[1]);
  vv2[0] = v2[0] + ratio_2 * (v3[0] - v2[0]);

  if(ratio_0 == 0.0 && ratio_1 == 0.0 && ratio_2 == 0.0) {
    geoframe_display_tri_vv(vv0, vv1, vv2, -1, 1, wire_flag);
  }
  else {
    geoframe_display_tri_vv(vv0, vv1, vv2, -1, 1, wire_flag);
    //geoframe_display_tri_v(v0,  v2,  v1,  1, wire_flag);
    //geoframe_display_tri_v(v1,  v2,  vv2, 1, wire_flag);
    //geoframe_display_tri_v(v1,  vv2, vv1, 1, wire_flag);
    //geoframe_display_tri_v(v2,  v0,  vv2, 1, wire_flag);
    //geoframe_display_tri_v(vv2, v0,  vv0, 1, wire_flag);
    //geoframe_display_tri_v(v1,  vv1, vv0, 1, wire_flag);
    //geoframe_display_tri_v(v1,  vv0, v0,  1, wire_flag);
    if(abs(vert_bound[0]) + abs(vert_bound[1]) + abs(vert_bound[2]) == 3)
      geoframe_display_tri_vv(v0,  v2,  v1,  4*c, 1, wire_flag);
    if(abs(vert_bound[1]) + abs(vert_bound[2]) + abs(vert_bound[3]) == 3) {
      geoframe_display_tri_vv(v1,  v2,  vv2, 4*c+1, 1, wire_flag);
      geoframe_display_tri_vv(v1,  vv2, vv1, 4*c+1, 1, wire_flag);
    }
    if(abs(vert_bound[0]) + abs(vert_bound[2]) == 2) {
      geoframe_display_tri_vv(v2,  v0,  vv2, 4*c+2, 1, wire_flag);
      geoframe_display_tri_vv(vv2, v0,  vv0, 4*c+2, 1, wire_flag);
    }
    if(abs(vert_bound[0]) + abs(vert_bound[1]) == 2) {
      geoframe_display_tri_vv(v1,  vv1, vv0, 4*c+3, 1, wire_flag);
      geoframe_display_tri_vv(v1,  vv0, v0,  4*c+3, 1, wire_flag);
    }
  }
}

void GeometryRenderable::geoframe_display_tetra(int c, int normalflag, int wire_flag)
{
  float v0[3], v1[3], v2[3], v3[3], v[4][3], plane_x;
  int vert, ii, jj, num_le, num_eq, vert_bound[4];

  for(ii = 0; ii < 3; ii++) {
    vert = m_Geometry->m_GeoFrame->triangles[4*c][ii];
    vert_bound[ii] = m_Geometry->m_GeoFrame->bound_sign[vert];
    for(jj = 0; jj < 3; jj++)
      v[ii][jj] = m_Geometry->m_GeoFrame->verts[vert][jj];
  }
  vert = m_Geometry->m_GeoFrame->triangles[4*c+1][2];
  vert_bound[3] = m_Geometry->m_GeoFrame->bound_sign[vert];
  for(jj = 0; jj < 3; jj++)
    v[3][jj] = m_Geometry->m_GeoFrame->verts[vert][jj];

  plane_x = 32.0f;//(float) ((int)biggestDim/2.0);	
  num_le = 0;	num_eq = 0;
  for(ii = 0; ii < 4; ii++) {
    if(v[ii][0] <= plane_x) num_le++;
    if(v[ii][0] == plane_x) num_eq++;
  }

  assert(num_le >= 0 && num_le <= 4);

  for(ii = 0; ii < 3; ii++) {
    v0[ii] = v[0][ii];		v1[ii] = v[2][ii];
    v2[ii] = v[1][ii];		v3[ii] = v[3][ii];
  }

  if(num_le == 1) {
    geoframe_display_permute_1(v0, v1, v2, v3, plane_x);
    geoframe_display_1(vert_bound, c, v0, v1, v2, v3, plane_x, normalflag, wire_flag);
  }
  if(num_le == 2) {
    geoframe_display_permute_2(v0, v1, v2, v3, plane_x);
    geoframe_display_2(vert_bound, c, v0, v1, v2, v3, plane_x, normalflag, wire_flag);
  }
  if(num_le == 3) {
    geoframe_display_permute_3(v0, v1, v2, v3, plane_x);
    geoframe_display_3(vert_bound, c, v0, v1, v2, v3, plane_x, normalflag, wire_flag);
  }
  if(num_le == 4) {
    geoframe_display_tri00(0, 1, 2, 4*c,   normalflag, wire_flag, num_eq);
    geoframe_display_tri00(0, 1, 2, 4*c+1, normalflag, wire_flag, num_eq);
    geoframe_display_tri00(0, 1, 2, 4*c+2, normalflag, wire_flag, num_eq);
    geoframe_display_tri00(0, 1, 2, 4*c+3, normalflag, wire_flag, num_eq);
  }
}

void GeometryRenderable::geoframe_display_tetra_in(int c, int normalflag, int wire_flag)
{
  float v0[3], v1[3], v2[3], v3[3], v[4][3], plane_x, plane_z;
  int vert, ii, jj, num_le, num_eq, vert_bound[4];

  for(ii = 0; ii < 3; ii++) {
    vert = m_Geometry->m_GeoFrame->triangles[4*c][ii];
    vert_bound[ii] = m_Geometry->m_GeoFrame->bound_sign[vert];
    for(jj = 0; jj < 3; jj++)
      v[ii][jj] = m_Geometry->m_GeoFrame->verts[vert][jj];
  }
  vert = m_Geometry->m_GeoFrame->triangles[4*c+1][2];
  vert_bound[3] = m_Geometry->m_GeoFrame->bound_sign[vert];
  for(jj = 0; jj < 3; jj++)
    v[3][jj] = m_Geometry->m_GeoFrame->verts[vert][jj];

  plane_x =  32.0f;//(float) ((int)biggestDim/2.0);	//32.0;
  plane_z =  48.0f;//48.00001;  
  num_le = 0;	num_eq = 0;
  for(ii = 0; ii < 4; ii++) {
    if(v[ii][2] <= plane_z) num_le++;
    if(v[ii][2] == plane_z) num_eq++;
  }

  assert(num_le >= 0 && num_le <= 4);

  for(ii = 0; ii < 3; ii++) {
    v0[ii] = v[0][ii];		v1[ii] = v[2][ii];
    v2[ii] = v[1][ii];		v3[ii] = v[3][ii];
  }

  if( (v[0][2] >= plane_z && v[0][0] >= plane_x) || (v[1][2] >= plane_z && v[1][0] >= plane_x) ||
      (v[2][2] >= plane_z && v[2][0] >= plane_x) || (v[3][2] >= plane_z && v[3][0] >= plane_x)) {

    geoframe_display_tetra(c, normalflag, wire_flag);

    if(num_le == 1) {
      geoframe_display_permute_1_z(v0, v1, v2, v3, plane_z);
      geoframe_display_1_z(vert_bound, c, v0, v1, v2, v3, plane_z, normalflag, wire_flag);
    }
    if(num_le == 2) {
      geoframe_display_permute_2_z(v0, v1, v2, v3, plane_z);
      geoframe_display_2_z(vert_bound, c, v0, v1, v2, v3, plane_z, normalflag, wire_flag);
    }
    if(num_le == 3) {
      geoframe_display_permute_3_z(v0, v1, v2, v3, plane_z);
      geoframe_display_3_z(vert_bound, c, v0, v1, v2, v3, plane_z, normalflag, wire_flag);
    }
    if(num_le == 4) {
      geoframe_display_tri00(0, 1, 2, 4*c,   normalflag, wire_flag, -num_eq);
      geoframe_display_tri00(0, 1, 2, 4*c+1, normalflag, wire_flag, -num_eq);
      geoframe_display_tri00(0, 1, 2, 4*c+2, normalflag, wire_flag, -num_eq);
      geoframe_display_tri00(0, 1, 2, 4*c+3, normalflag, wire_flag, -num_eq);
    }

  }
  else {
    geoframe_display_tri0(0, 1, 2, 4*c,   normalflag, wire_flag);
    geoframe_display_tri0(0, 1, 2, 4*c+1, normalflag, wire_flag);
    geoframe_display_tri0(0, 1, 2, 4*c+2, normalflag, wire_flag);
    geoframe_display_tri0(0, 1, 2, 4*c+3, normalflag, wire_flag);
  }
}

static inline void get_trinorm(float* tn, LBIE::geoframe* g, int c, int normal_flag)
{
  float v1[3],v2[3];
  int vert;

  vert = g->triangles[c][0];
  v1[0] = v2[0] = -g->verts[vert][0];
  v1[1] = v2[1] = -g->verts[vert][1];
  v1[2] = v2[2] = -g->verts[vert][2];
					
  vert = g->triangles[c][1];
  v1[0] += g->verts[vert][0];
  v1[1] += g->verts[vert][1];
  v1[2] += g->verts[vert][2];
					
  vert = g->triangles[c][2];
  v2[0] += g->verts[vert][0];
  v2[1] += g->verts[vert][1];
  v2[2] += g->verts[vert][2];
					
  cross(tn, v1, v2);

	
  // normal flipping
  if (normal_flag==1) {
    tn[0]=-tn[0];
    tn[1]=-tn[1];
    tn[2]=-tn[2];
  }

}

void GeometryRenderable::geoframe_display()
{
  int vert, c, index, ii, my_bool;
  //int i, t, idx, v, vidx2;
  //int* tri_num;
  //int** v_tri;
  //float tri_norm[3];
  char* p = NULL;
  //float norm[3], v1[3], v2[3], v[4][3];
	
  int normal_flag = 0; // 0, 1  //zq change 0 to 1 on Feb 20, 2009

  //flat_flag=1;
  //wireframe_flag=1;
  //int cut_flag = 0;

  //if ((p=strstr("Head",fn))!= NULL) normal_flag = 0;
		
  m_FlatFlag=2;
  if (m_Geometry->m_GeoFrame) {

    glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_POLYGON_BIT);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, 1);
		
    if((m_Geometry->m_GeoFrame->numhexas * 6) != m_Geometry->m_GeoFrame->numquads) m_CutFlag = 0;

    //if (flat_flag==1) {
    if (m_FlatFlag >= 1) {
	
      //for (i=0;i<m_Geometry->m_GeoFrame->numquads;i++) {
      //	m_Geometry->m_GeoFrame->AddTri(m_Geometry->m_GeoFrame->quads[i][0],m_Geometry->m_GeoFrame->quads[i][1],m_Geometry->m_GeoFrame->quads[i][2]);
      //	m_Geometry->m_GeoFrame->AddTri(m_Geometry->m_GeoFrame->quads[i][2],m_Geometry->m_GeoFrame->quads[i][3],m_Geometry->m_GeoFrame->quads[i][0]);
      //}
      /*		
      // calculate normal vector
      tri_num = new int[m_Geometry->m_GeoFrame->numverts];
      for(i = 0;i < m_Geometry->m_GeoFrame->numverts; i++) tri_num[i] = 0;

      for(t = 0; t < m_Geometry->m_GeoFrame->numtris; t++) {
      for(idx = 0; idx < 3; idx++) {
      v = m_Geometry->m_GeoFrame->triangles[t][idx];
      tri_num[v]++;
      }
      }
			
      v_tri = (int**)malloc(sizeof(int*)*m_Geometry->m_GeoFrame->numverts);
      for(i = 0; i < m_Geometry->m_GeoFrame->numverts; i++) {
      v_tri[i] = (int*)malloc(sizeof(int)*tri_num[i]);
      tri_num[i] = 0;
      }
			
      for(t = 0; t < m_Geometry->m_GeoFrame->numtris; t++) {
      //if (fabs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[t][0]]) + 
      //	fabs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[t][1]]) +
      //	fabs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[t][2]]) == 3) {
      for (idx = 0; idx < 3; idx++) {
      v = m_Geometry->m_GeoFrame->triangles[t][idx];
      v_tri[v][tri_num[v]] = t;
      tri_num[v]++;
      }
      //}
      }
			
      for(vidx2 = 0; vidx2 < m_Geometry->m_GeoFrame->numverts; vidx2++) {
				
      m_Geometry->m_GeoFrame->normals[vidx2][0] = 0; 
      m_Geometry->m_GeoFrame->normals[vidx2][1] = 0;
      m_Geometry->m_GeoFrame->normals[vidx2][2] = 0;
				
      for(idx = 0; idx < tri_num[vidx2]; idx++) {
      //t = v_tri[vidx2][idx];
      get_trinorm(tri_norm, m_Geometry->m_GeoFrame, v_tri[vidx2][idx], normal_flag);
      //if(fabs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[t][0]] + 
      //		m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[t][1]] +
      //		m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[t][2]]) > 3) {
      m_Geometry->m_GeoFrame->normals[vidx2][0] += tri_norm[0];
      m_Geometry->m_GeoFrame->normals[vidx2][1] += tri_norm[1];
      m_Geometry->m_GeoFrame->normals[vidx2][2] += tri_norm[2];
      //}
      }
				
      if(m_Geometry->m_GeoFrame->bound_sign[vidx2] == 1) {
      m_Geometry->m_GeoFrame->normals[vidx2][0] /= -(float)tri_num[vidx2];
      m_Geometry->m_GeoFrame->normals[vidx2][1] /= -(float)tri_num[vidx2];
      m_Geometry->m_GeoFrame->normals[vidx2][2] /= -(float)tri_num[vidx2];
      }
      else {
      m_Geometry->m_GeoFrame->normals[vidx2][0] /= (float)tri_num[vidx2];
      m_Geometry->m_GeoFrame->normals[vidx2][1] /= (float)tri_num[vidx2];
      m_Geometry->m_GeoFrame->normals[vidx2][2] /= (float)tri_num[vidx2];
      }
				
      }
			
      for (i=0;i<m_Geometry->m_GeoFrame->numverts;i++) {
      if (tri_num[i]>0) free(v_tri[i]);
      }
      free(v_tri);	// finish normal calculation
      */		

      if(m_CutFlag == 1) {
	for(c = 0; c < ((m_Geometry->m_GeoFrame->numtris)/4); c++) {
	  geoframe_display_tetra_in(c, normal_flag, 0);
	  //geoframe_display_tetra(c, normal_flag, 0);
	}
	for(c = 0; c < m_Geometry->m_GeoFrame->numhexas; c++) {
	  geoframe_display_hexa(c, normal_flag, 0);
	}
      }
      else if(m_CutFlag == 2) {
	for (c=0; c<((m_Geometry->m_GeoFrame->numtris)/4); c++) {
	  geoframe_display_tetra_in(c, normal_flag, 0);
	}
      }
      else {
	
	for (c = 0; c < m_Geometry->m_GeoFrame->numtris; c++) 
	{
	geoframe_display_tri0(0, 1, 2, c, normal_flag, 0);
//	printf("Strange!\n");
	}
	//geoframe_display_tri0(0, 1, 2, c, 1, 0);
	for (c = 0; c < m_Geometry->m_GeoFrame->numquads; c++) {
	  geoframe_display_tri(0, 1, 2, c, normal_flag);
	  geoframe_display_tri(2, 3, 0, c, normal_flag);
	}
      }
	
    }
    /*
      } else if (flat_flag == 2) {
				
      glBegin(GL_TRIANGLES);
      for (c = 0; c < m_Geometry->m_GeoFrame->numtris; c++) {
					
      if (normal_flag == 0) {
      for(ii = 0; ii < 3; ii++) {
      vert = m_Geometry->m_GeoFrame->triangles[c][ii];
      glNormal3d(	-m_Geometry->m_GeoFrame->normals[vert][0],
      -m_Geometry->m_GeoFrame->normals[vert][1],
      -m_Geometry->m_GeoFrame->normals[vert][2] );
      glVertex3d(	 m_Geometry->m_GeoFrame->verts[vert][0],
      m_Geometry->m_GeoFrame->verts[vert][1],
      m_Geometry->m_GeoFrame->verts[vert][2] );
      }
      } 
      else {
      for(ii = 0; ii < 3; ii++) {
      vert = m_Geometry->m_GeoFrame->triangles[c][ii];
      glNormal3d( m_Geometry->m_GeoFrame->normals[vert][0],
      m_Geometry->m_GeoFrame->normals[vert][1],
      m_Geometry->m_GeoFrame->normals[vert][2] );
      glVertex3d(	m_Geometry->m_GeoFrame->verts[vert][0],
      m_Geometry->m_GeoFrame->verts[vert][1],
      m_Geometry->m_GeoFrame->verts[vert][2] );
      }
      }
      }
				
      glEnd();

      glBegin(GL_QUADS);
      for (c=0; c<m_Geometry->m_GeoFrame->numquads; c++) {
					
      if (normal_flag==0) {
      for(ii = 0; ii < 4; ii++) {
      vert = m_Geometry->m_GeoFrame->quads[c][ii];
      glNormal3d(	-m_Geometry->m_GeoFrame->normals[vert][0],
      -m_Geometry->m_GeoFrame->normals[vert][1],
      -m_Geometry->m_GeoFrame->normals[vert][2] );
      glVertex3d(  m_Geometry->m_GeoFrame->verts[vert][0],
      m_Geometry->m_GeoFrame->verts[vert][1],
      m_Geometry->m_GeoFrame->verts[vert][2] );
      }
      } 
      else {
      for(ii = 0; ii < 4; ii++) {
      vert = m_Geometry->m_GeoFrame->quads[c][ii];
      glNormal3d( m_Geometry->m_GeoFrame->normals[vert][0],
      m_Geometry->m_GeoFrame->normals[vert][1],
      m_Geometry->m_GeoFrame->normals[vert][2] );
      glVertex3d(	m_Geometry->m_GeoFrame->verts[vert][0],
      m_Geometry->m_GeoFrame->verts[vert][1],
      m_Geometry->m_GeoFrame->verts[vert][2] );
      }
      }
      }
      glEnd();
				
      }
    */		glPopAttrib();

    glDisable(GL_LIGHTING);

    //glLineWidth(2.0);

    //if (wireframe_flag==1) {
    if (m_WireframeRender) {
      glColor3d(0.2,0.2,0.2);
			
      if(m_CutFlag == 1) {
	for (c=0; c<m_Geometry->m_GeoFrame->numtris/4; c++) {
	  geoframe_display_tetra_in(c, normal_flag, 1);
	  //geoframe_display_tetra(c, normal_flag, 1);
	}
	//for (c = 0; c < ((m_Geometry->m_GeoFrame->numquads)/6); c++) {
	for (c = 0; c < m_Geometry->m_GeoFrame->numhexas; c++) {
	  geoframe_display_hexa(c, normal_flag, 1);
	}
      }
      else if(m_CutFlag == 2) {
	for (c=0; c<m_Geometry->m_GeoFrame->numtris/4; c++) 
	  geoframe_display_tetra(c, normal_flag, 1);
      }
      else {
	for (c=0; c<m_Geometry->m_GeoFrame->numtris; c++) {
	  my_bool =	abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][0]]) == 1 && 
	    abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][1]]) == 1 &&
	    abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->triangles[c][2]]) == 1;
	  if(my_bool) {
	    glBegin(GL_LINE_STRIP);
	    for(ii = 0; ii < 4; ii++) {
	      index = ii;
	      if(ii == 3) index = 0;
	      vert = m_Geometry->m_GeoFrame->triangles[c][index];
	      //glNormal3d(norm[0], norm[1], norm[2]);
	      glVertex3d(
			 m_Geometry->m_GeoFrame->verts[vert][0],
			 m_Geometry->m_GeoFrame->verts[vert][1],
			 m_Geometry->m_GeoFrame->verts[vert][2]);
	    }
	    glEnd();
	  }
	}
			
	for (c=0; c<m_Geometry->m_GeoFrame->numquads; c++) {
	  glBegin(GL_LINE_STRIP);

	  my_bool =	abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][0]]) == 1 && 
	    abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][1]]) == 1 &&
	    abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][2]]) == 1 &&
	    abs(m_Geometry->m_GeoFrame->bound_sign[m_Geometry->m_GeoFrame->quads[c][3]]) == 1;
	  if(my_bool) {
	    for(ii = 0; ii < 4; ii++) {
	      vert = m_Geometry->m_GeoFrame->quads[c][ii];
	      //glNormal3d(norm[0], norm[1], norm[2]);
	      glVertex3d(
			 m_Geometry->m_GeoFrame->verts[vert][0],
			 m_Geometry->m_GeoFrame->verts[vert][1],
			 m_Geometry->m_GeoFrame->verts[vert][2]);
	    }
	    vert = m_Geometry->m_GeoFrame->quads[c][0];
	    //glNormal3d(norm[0], norm[1], norm[2]);
	    glVertex3d(
		       m_Geometry->m_GeoFrame->verts[vert][0],
		       m_Geometry->m_GeoFrame->verts[vert][1],
		       m_Geometry->m_GeoFrame->verts[vert][2]);
	  }
	  /*
	  //glColor3d(0.2,0.2,0.6);
	  for(ii = 0; ii < 4; ii++) {
	  vert = m_Geometry->m_GeoFrame->quads[c][ii];
	  glNormal3d(norm[0], norm[1], norm[2]);
	  glVertex3d(
	  (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*0.8 + 32.0,
	  (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*0.8 + 32.0,
	  (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*0.9 + 32.0);
	  }
	  vert = m_Geometry->m_GeoFrame->quads[c][0];
	  glNormal3d(norm[0], norm[1], norm[2]);
	  glVertex3d(
	  (m_Geometry->m_GeoFrame->verts[vert][0] - 32.0)*0.8 + 32.0,
	  (m_Geometry->m_GeoFrame->verts[vert][1] - 32.0)*0.8 + 32.0,
	  (m_Geometry->m_GeoFrame->verts[vert][2] - 32.0)*0.9 + 32.0);

	  if(m_Geometry->m_GeoFrame->verts[vert][2] > 32.0) {
	  geoframe_display_tri_cross0(0, 1, 1, 1.0, 1.0, 0.8, c, -normal_flag);
	  geoframe_display_tri_cross0(0, 1, 0, 1.0, 0.8, 0.8, c, -normal_flag);

	  geoframe_display_tri_cross0(1, 2, 1, 1.0, 1.0, 0.8, c, -normal_flag);
	  geoframe_display_tri_cross0(2, 2, 1, 1.0, 0.8, 0.8, c, -normal_flag);

	  geoframe_display_tri_cross0(2, 0, 2, 1.0, 1.0, 0.8, c, -normal_flag);
	  geoframe_display_tri_cross0(0, 0, 2, 1.0, 0.8, 0.8, c, -normal_flag);
	  }
	  */
	  //}
	  glEnd();
						
	  //}
	}

      }
      glEnable(GL_LIGHTING);
      glColor3d(1, 1, 1);
    }
			
  }
			
}
