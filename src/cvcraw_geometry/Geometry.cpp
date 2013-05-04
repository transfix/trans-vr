/*
  Copyright 2002-2005 The University of Texas at Austin

        Authors: Anthony Thane <thanea@ices.utexas.edu>
                 Vinay Siddavanahalli <skvinay@cs.utexas.edu>
                 Jose Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: Geometry.cpp 4741 2011-10-21 21:22:06Z transfix $ */

// Geometry.cpp: implementation of the Geometry class.
//
//////////////////////////////////////////////////////////////////////

#include <cvcraw_geometry/Geometry.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <boost/scoped_ptr.hpp>

#include <LBIE/quality_improve.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Geometry::Geometry()
  : m_LineVerts(m_Points), m_NumLineVerts(m_NumPoints), m_LineColors(m_PointColors),
    m_QuadVerts(m_Points), m_NumQuadVerts(m_NumPoints), m_QuadVertNormals(m_PointNormals), m_QuadVertTexCoords(m_PointTexCoords),
    m_QuadFlatVerts(m_Points), m_QuadFlatNormals(m_PointFlatNormals), m_QuadFlatTexCoords(m_PointTexCoords),
    m_TriVerts(m_Points), m_NumTriVerts(m_NumPoints), m_TriVertNormals(m_PointNormals), m_TriVertColors(m_PointColors), m_TriVertTexCoords(m_PointTexCoords),
    m_TriFlatVerts(m_Points), m_TriFlatNormals(m_PointFlatNormals), m_TriFlatTexCoords(m_PointTexCoords)
{
  ZeroMembers();
  InitializeColors();
}

Geometry::~Geometry()
{
  ClearGeometry();
}

void Geometry::ZeroMembers()
{
#if 0
  m_Points = 0;
  m_NumPoints = 0;
  m_PointScalars = 0;

  m_LineVerts = 0;
  m_LineColors = 0;
  m_NumLineVerts = 0;
  m_Lines = 0;
  m_NumLines = 0;

  m_QuadVerts = 0;
  m_QuadVertNormals = 0;
  m_QuadVertTexCoords = 0;
  m_NumQuadVerts = 0;
  m_Quads = 0;
  m_NumQuads = 0;
  m_QuadFlatVerts = 0;
  m_QuadFlatNormals = 0;
  m_QuadFlatTexCoords = 0;
  m_bQuadFlatNormalsReady = 0;
  m_bQuadSmoothNormalsReady = 0;

  m_TriVerts = 0;
  m_TriVertNormals = 0;
  m_TriVertTexCoords = 0;
  m_TriVertColors = 0;
  m_NumTriVerts = 0;
  m_Tris = 0;
  m_NumTris = 0;
  m_TriFlatVerts = 0;
  m_TriFlatNormals = 0;
  m_TriFlatTexCoords = 0;
  m_bTriFlatNormalsReady = 0;
  m_bTriSmoothNormalsReady = 0;

  m_bExtentsReady = false;
#endif

  m_Points.reset();
  m_PointNormals.reset();
  m_PointFlatNormals.reset();
  m_PointColors.reset();
  m_PointTexCoords.reset();

  m_Lines.reset();
  m_NumLineVerts = 0;
  m_NumLines = 0;

  m_Quads.reset();
  m_NumQuadVerts = 0;
  m_NumQuads = 0;
  m_bQuadFlatNormalsReady = 0;
  m_bQuadSmoothNormalsReady = 0;

  m_Tris.reset();
  m_NumTriVerts = 0;
  m_NumTris = 0;
  m_bTriFlatNormalsReady = 0;
  m_bTriSmoothNormalsReady = 0;
  
  m_bExtentsReady = false;
}

void Geometry::InitializeColors()
{
  m_LineWidth = 1.0;

  m_DiffuseColor[0] = 1.0;
  m_DiffuseColor[1] = 1.0;
  m_DiffuseColor[2] = 1.0;

  m_SpecularColor[0] = 1.0;
  m_SpecularColor[1] = 1.0;
  m_SpecularColor[2] = 1.0;

  m_AmbientColor[0] = 0.0;
  m_AmbientColor[1] = 0.0;
  m_AmbientColor[2] = 0.0;

  m_Shininess = 30.0;
}

Geometry::Geometry(const Geometry& copy)
  : m_LineVerts(m_Points), m_NumLineVerts(m_NumPoints), m_LineColors(m_PointColors),
    m_QuadVerts(m_Points), m_NumQuadVerts(m_NumPoints), m_QuadVertNormals(m_PointNormals), m_QuadVertTexCoords(m_PointTexCoords),
    m_QuadFlatVerts(m_Points), m_QuadFlatNormals(m_PointFlatNormals), m_QuadFlatTexCoords(m_PointTexCoords),
    m_TriVerts(m_Points), m_NumTriVerts(m_NumPoints), m_TriVertNormals(m_PointNormals), m_TriVertColors(m_PointColors), m_TriVertTexCoords(m_PointTexCoords),
    m_TriFlatVerts(m_Points), m_TriFlatNormals(m_PointFlatNormals), m_TriFlatTexCoords(m_PointTexCoords)
{
  ZeroMembers();
  CopyPoints(copy);
  CopyPointNormals(copy);
  CopyPointFlatNormals(copy);
  CopyPointColors(copy);
  CopyPointTexCoords(copy);
  CopyPointScalars(copy);

  CopyLines(copy);
  //CopyLineColors(copy);
  CopyQuads(copy);
  //CopyQuadTexCoords(copy);
  CopyTris(copy);
  //CopyTriTexCoords(copy);
  //CopyTriVertColors(copy);
  CopyColors(copy);

  if(copy.m_GeoFrame)
    m_GeoFrame.reset(new LBIE::geoframe(*copy.m_GeoFrame.get()));
}

Geometry& Geometry::operator=(const Geometry& copy)
{
  if (this!=&copy) {
    ClearGeometry();
    /*
    CopyPoints(copy);
    CopyPointScalars(copy);
    CopyLines(copy);
    CopyQuads(copy);
    CopyQuadTexCoords(copy);
    CopyTris(copy);
    CopyTriTexCoords(copy);
    CopyTriVertColors(copy);
    CopyColors(copy);
    */

    CopyPoints(copy);
    CopyPointNormals(copy);
    CopyPointFlatNormals(copy);
    CopyPointColors(copy);
    CopyPointTexCoords(copy);
    CopyPointScalars(copy);
    
    CopyLines(copy);
    //CopyLineColors(copy);
    CopyQuads(copy);
    //CopyQuadTexCoords(copy);
    CopyTris(copy);
    //CopyTriTexCoords(copy);
    //CopyTriVertColors(copy);
    CopyColors(copy);

    if(copy.m_GeoFrame)
      m_GeoFrame.reset(new LBIE::geoframe(*copy.m_GeoFrame.get()));

    m_SceneGeometry = copy.m_SceneGeometry;
  }
  return *this;
}

void Geometry::CopyPoints(const Geometry& copy)
{
  AllocatePoints(copy.m_NumPoints);
  unsigned int c;
  for (c=0; c<m_NumPoints*3; c++) { // for every coordinate of every point
    m_Points[c] = copy.m_Points[c];
  }
}

void Geometry::CopyPointNormals(const Geometry& copy)
{
  if(copy.m_PointNormals)
    {
      AllocatePointNormals();
      memcpy(m_PointNormals.get(),copy.m_PointNormals.get(),sizeof(float)*m_NumPoints*3);
    }
  else
    {
      m_PointNormals.reset();
    }
}

void Geometry::CopyPointFlatNormals(const Geometry& copy)
{
  if(copy.m_PointFlatNormals)
    {
      AllocatePointFlatNormals();
      memcpy(m_PointFlatNormals.get(),copy.m_PointFlatNormals.get(),sizeof(float)*m_NumPoints*3);
    }
  else
    {
      m_PointFlatNormals.reset();
    }
}

void Geometry::CopyPointColors(const Geometry& copy)
{
  if(copy.m_PointColors)
    {
      AllocatePointColors();
      memcpy(m_PointColors.get(),copy.m_PointColors.get(),sizeof(float)*m_NumPoints*3);
    }
  else
    {
      m_PointColors.reset();
    }
}

void Geometry::CopyPointTexCoords(const Geometry& copy)
{
  if(copy.m_PointTexCoords)
    {
      AllocatePointTexCoords();
      memcpy(m_PointTexCoords.get(),copy.m_PointTexCoords.get(),sizeof(float)*m_NumPoints*2);
    }
  else
    {
      m_PointTexCoords.reset();
    }
}

void Geometry::CopyPointScalars(const Geometry& copy)
{
  if(copy.m_PointScalars)
    {
      AllocatePointScalars();
      memcpy(m_PointScalars.get(),copy.m_PointScalars.get(),sizeof(float)*m_NumPoints);
    }
  else
    {
      m_PointScalars.reset();
    }
}

void Geometry::CopyLines(const Geometry& copy)
{
  if(m_NumLineVerts != copy.m_NumLineVerts)
    AllocateLines(copy.m_NumLineVerts, copy.m_NumLines);
  else
    {
      m_Lines.reset(new unsigned int[copy.m_NumLines*2]);
      m_NumLines = copy.m_NumLines;
    }
  unsigned int c;
  for (c=0; c<m_NumLineVerts*3; c++) { // for every coordinate of every LineVert
    m_LineVerts[c] = copy.m_LineVerts[c];
  }
  for (c=0; c<m_NumLines*2; c++) { // for every Vert of every Line
    m_Lines[c] = copy.m_Lines[c];
  }
}

void Geometry::CopyLineColors(const Geometry& copy)
{
  unsigned int c;
  if (copy.m_LineColors) {
    AllocateLineColors();
    for (c=0; c<m_NumLineVerts*3; c++) {
      m_LineColors[c] = copy.m_LineColors[c];
    }
  }
  else {
    //m_LineColors = 0;
    m_LineColors.reset();
  }
}

void Geometry::CopyQuads(const Geometry& copy)
{
  if(m_NumQuadVerts != copy.m_NumQuadVerts)
    AllocateQuads(copy.m_NumQuadVerts, copy.m_NumQuads);
  else
    {
      m_Quads.reset(new unsigned int[copy.m_NumQuads*4]);
      m_NumQuads = copy.m_NumQuads;
    }
  unsigned int c;
  for (c=0; c<m_NumQuadVerts*3; c++) { // for every coordinate of every QuadVert
    m_QuadVerts[c] = copy.m_QuadVerts[c];
  }
  if(copy.m_QuadVertNormals)
    {
      AllocatePointNormals();
      for (c=0; c<m_NumQuadVerts*3; c++) { // for every coordinate of every QuadVertNormal
	m_QuadVertNormals[c] = copy.m_QuadVertNormals[c];
      }
    }
  for (c=0; c<m_NumQuads*4; c++) { // for every Vert of every Quad
    m_Quads[c] = copy.m_Quads[c];
  }
  if(copy.m_QuadFlatNormals)
    {
      AllocatePointFlatNormals();
      for (c=0; c<m_NumQuadVerts*3; c++) { // for every coordinate of every vertex of every flat quad
	m_QuadFlatVerts[c] = copy.m_QuadFlatVerts[c];
	m_QuadFlatNormals[c] = copy.m_QuadFlatNormals[c];
      }
    }
  m_bQuadFlatNormalsReady = copy.m_bQuadFlatNormalsReady;
  m_bQuadSmoothNormalsReady = copy.m_bQuadSmoothNormalsReady;
}

void Geometry::CopyQuadTexCoords(const Geometry& copy)
{
  unsigned int c;
  if (copy.m_QuadVertTexCoords) {
    AllocateQuadTexCoords();
    for (c=0; c<m_NumQuadVerts*2; c++) {
      m_QuadVertTexCoords[c] = copy.m_QuadVertTexCoords[c];
    }
    for (c=0; c<m_NumQuadVerts*2; c++) {
      m_QuadFlatTexCoords[c] = copy.m_QuadFlatTexCoords[c];
    }
  }
  else {
    //m_QuadVertTexCoords = 0;
    //m_QuadFlatTexCoords = 0;
    m_QuadVertTexCoords.reset();
    m_QuadFlatTexCoords.reset();
  }
}

void Geometry::CopyTris(const Geometry& copy)
{
  if(m_NumTriVerts != copy.m_NumTriVerts)
    AllocateTris(copy.m_NumTriVerts, copy.m_NumTris);
  else
    {
      m_Tris.reset(new unsigned int[copy.m_NumTris*3]);
      m_NumTris = copy.m_NumTris;
    }
  unsigned int c;
  for (c=0; c<m_NumTriVerts*3; c++) { // for every coordinate of every TriVert
    m_TriVerts[c] = copy.m_TriVerts[c];
  }
  if(copy.m_TriVertNormals)
    {
      AllocatePointNormals();
      for (c=0; c<m_NumTriVerts*3; c++) { // for every coordinate of every TriVertNormal
	m_TriVertNormals[c] = copy.m_TriVertNormals[c];
      }
    }
  for (c=0; c<m_NumTris*3; c++) { // for every Vert of every Tri
    m_Tris[c] = copy.m_Tris[c];
  }
  if(copy.m_TriFlatNormals)
    {
      AllocatePointFlatNormals();
      for (c=0; c<m_NumTriVerts*3; c++)
	{ 
	  // for every coordinate of every vertex of every flat Tri
	  m_TriFlatVerts[c] = copy.m_TriFlatVerts[c];
	  m_TriFlatNormals[c] = copy.m_TriFlatNormals[c];
	}
    }
  m_bTriFlatNormalsReady = copy.m_bTriFlatNormalsReady;
  m_bTriSmoothNormalsReady = copy.m_bTriSmoothNormalsReady;
}

void Geometry::CopyTriTexCoords(const Geometry& copy)
{
  unsigned int c;
  if (copy.m_TriVertTexCoords) {
    AllocateTriTexCoords();
    for (c=0; c<m_NumTriVerts*2; c++) {
      m_TriVertTexCoords[c] = copy.m_TriVertTexCoords[c];
    }
    for (c=0; c<m_NumTriVerts*2; c++) {
      m_TriFlatTexCoords[c] = copy.m_TriFlatTexCoords[c];
    }
  }
  else {
    //m_TriVertTexCoords = 0;
    //m_TriFlatTexCoords = 0;
    m_TriVertTexCoords.reset();
    m_TriFlatTexCoords.reset();
  }
}

void Geometry::CopyTriVertColors(const Geometry& copy)
{
  unsigned int c;
  if (copy.m_TriVertColors) {
    AllocateTriVertColors();
    for (c=0; c<m_NumTriVerts*3; c++) {
      m_TriVertColors[c] = copy.m_TriVertColors[c];
    }
  }
  else {
    //m_TriVertColors = 0;
    m_TriVertColors.reset();
  }
}

void Geometry::CopyColors(const Geometry& copy)
{
  m_LineWidth = copy.m_LineWidth;

  m_DiffuseColor[0] = copy.m_DiffuseColor[0];
  m_DiffuseColor[1] = copy.m_DiffuseColor[1];
  m_DiffuseColor[2] = copy.m_DiffuseColor[2];

  m_SpecularColor[0] = copy.m_SpecularColor[0];
  m_SpecularColor[1] = copy.m_SpecularColor[1];
  m_SpecularColor[2] = copy.m_SpecularColor[2];

  m_AmbientColor[0] = copy.m_AmbientColor[0];
  m_AmbientColor[1] = copy.m_AmbientColor[1];
  m_AmbientColor[2] = copy.m_AmbientColor[2];

  m_Shininess = copy.m_Shininess;
}

void Geometry::ClearGeometry()
{
  ClearPoints();
  ClearPointNormals();
  ClearPointFlatNormals();
  ClearPointColors();
  ClearPointTexCoords();
  ClearPointScalars();
  ClearLines();
  //ClearLineColors();
  ClearQuads();
  //ClearQuadTexCoords();
  ClearTris();
  //ClearTriTexCoords();
  //ClearTriVertColors();
  m_bExtentsReady = false;
}

void Geometry::ClearPoints()
{
  //delete [] m_Points;
  m_Points.reset();
  //m_Points = 0;
  m_NumPoints = /*m_NumLineVerts = m_NumQuadVerts = m_NumTriVerts =*/ 0;
  m_bExtentsReady = false;
}

void Geometry::ClearPointNormals()
{
  m_PointNormals.reset();
}

void Geometry::ClearPointFlatNormals()
{
  m_PointFlatNormals.reset();
}

void Geometry::ClearPointColors()
{
  m_PointColors.reset();
}

void Geometry::ClearPointTexCoords()
{
  m_PointTexCoords.reset();
}

void Geometry::ClearPointScalars()
{
  //if(m_PointScalars)
  //delete [] m_PointScalars;
  //m_PointScalars = 0;
  m_PointScalars.reset();
}

void Geometry::ClearLines()
{
  //delete [] m_LineVerts;
  //m_LineVerts = 0;
  //m_NumLineVerts = 0;
  //delete [] m_Lines;
  //m_Lines = 0;
  m_Lines.reset();
  m_NumLines = 0;
}

void Geometry::ClearLineColors()
{
  //delete [] m_LineColors;
  //m_LineColors = 0;
  ClearPointColors();
}

void Geometry::ClearQuads()
{
  //delete [] m_QuadVerts;
  //m_QuadVerts = 0;
  //delete [] m_QuadVertNormals;
  //m_QuadVertNormals = 0;
  //m_NumQuadVerts = 0;
  //delete [] m_Quads;
  //m_Quads = 0;
  m_Quads.reset();
  m_NumQuads = 0;
  //delete [] m_QuadFlatVerts;
  //m_QuadFlatVerts = 0;
  //delete [] m_QuadFlatNormals;
  //m_QuadFlatNormals = 0;
  m_bQuadFlatNormalsReady = false;
  m_bQuadSmoothNormalsReady = false;
}

void Geometry::ClearQuadTexCoords()
{
  //delete [] m_QuadVertTexCoords;
  //m_QuadVertTexCoords = 0;
  //delete [] m_QuadFlatTexCoords;
  //m_QuadFlatTexCoords = 0;
  ClearPointTexCoords();
}

void Geometry::ClearTris()
{
  //delete [] m_TriVerts;
  //m_TriVerts = 0;
  //delete [] m_TriVertNormals;
  //m_TriVertNormals = 0;
  //m_NumTriVerts = 0;
  //delete [] m_Tris;
  //m_Tris = 0;
  m_Tris.reset();
  m_NumTris = 0;
  //delete [] m_TriFlatVerts;
  //m_TriFlatVerts = 0;
  //delete [] m_TriFlatNormals;
  //m_TriFlatNormals = 0;
  m_bTriFlatNormalsReady = false;
  m_bTriSmoothNormalsReady = false;
}

void Geometry::ClearTriTexCoords()
{
  //delete [] m_TriVertTexCoords;
  //m_TriVertTexCoords = 0;
  //delete [] m_TriFlatTexCoords;
  //m_TriFlatTexCoords = 0;
  ClearPointTexCoords();
}

void Geometry::ClearTriVertColors()
{
  //delete [] m_TriVertColors;
  //m_TriVertColors = 0;
  ClearPointColors();
}

void Geometry::AllocatePoints(unsigned int NumPoints)
{
  ClearPoints();
  m_Points.reset(new float[NumPoints*3]);
  m_NumPoints = /*m_NumLineVerts = m_NumQuadVerts = m_NumTriVerts =*/ NumPoints;
  m_bExtentsReady = false;
}

void Geometry::AllocatePointNormals()
{
  ClearPointNormals();
  m_PointNormals.reset(new float[m_NumPoints*3]);
}

void Geometry::AllocatePointFlatNormals()
{
  ClearPointFlatNormals();
  m_PointFlatNormals.reset(new float[m_NumPoints*3]);
}

void Geometry::AllocatePointColors()
{
  ClearPointColors();
  m_PointColors.reset(new float[m_NumPoints*3]);
  for(int i = 0; i < m_NumPoints*3; i++)
    m_PointColors[i] = 1.0f;
}

void Geometry::AllocatePointTexCoords()
{
  ClearPointTexCoords();
  m_PointTexCoords.reset(new float[m_NumPoints*2]);
}

void Geometry::AllocatePointScalars()
{
  ClearPointScalars();
  m_PointScalars.reset(new float[m_NumPoints]);
}

void Geometry::AllocateLines(unsigned int NumLineVerts, unsigned int NumLines)
{
  ClearLines();
  //m_LineVerts = new float[NumLineVerts*3];
  //m_NumLineVerts = NumLineVerts;
  AllocatePoints(NumLineVerts);
  m_Lines.reset(new unsigned int[NumLines*2]);
  m_NumLines = NumLines;
  m_bExtentsReady = false;
}

void Geometry::AllocateLineColors()
{
  ClearLineColors();
  //m_LineColors = boost::shared_array<float>(new float[m_NumLineVerts*3]);
  AllocatePointColors();
  unsigned int c;
  for (c=0; c<m_NumLineVerts; c++) {
    m_LineColors[c*3+0] = m_DiffuseColor[0];
    m_LineColors[c*3+1] = m_DiffuseColor[1];
    m_LineColors[c*3+2] = m_DiffuseColor[2];
  }
}

void Geometry::AllocateQuads(unsigned int NumQuadVerts, unsigned int NumQuads)
{
  ClearQuads();
  //m_QuadVerts = new float[NumQuadVerts*3];
  //m_QuadVertNormals = new float[NumQuadVerts*3];
  //m_NumQuadVerts = NumQuadVerts;
  AllocatePoints(NumQuadVerts);
  AllocatePointNormals();
  AllocatePointFlatNormals();
  AllocatePointColors();
  m_Quads.reset(new unsigned int[NumQuads*4]);
  m_NumQuads = NumQuads;
  //m_QuadFlatVerts = new float[NumQuads*4*3];
  //m_QuadFlatNormals = new float[NumQuads*4*3];
  /*
  unsigned int c;
  for (c=0; c<NumQuads*4*3; c++) {
    m_QuadFlatVerts[c] = 0.0;
    m_QuadFlatNormals[c] = 0.0;
    }*/
  m_bExtentsReady = false;
}

void Geometry::AllocateQuadTexCoords()
{
  ClearQuadTexCoords();
  //m_QuadVertTexCoords = new float[m_NumQuadVerts*3];
  AllocatePointTexCoords();
  unsigned int c;
  for (c=0; c<m_NumQuadVerts*2; c++) {
    m_QuadVertTexCoords[c] = 0.0f;
  }
  /*m_QuadFlatTexCoords = new float[m_NumQuads*4*3];
  for (c=0; c<m_NumQuads*4*3; c++) {
    m_QuadFlatTexCoords[c] = 0.0f;
    }*/
}

void Geometry::AllocateTris(unsigned int NumTriVerts, unsigned int NumTris)
{
  ClearTris();
  //m_TriVerts = new float[NumTriVerts*3];
  //m_TriVertNormals = new float[NumTriVerts*3];
  //m_NumTriVerts = NumTriVerts;
  AllocatePoints(NumTriVerts);
  AllocatePointNormals();
  AllocatePointFlatNormals();
  AllocatePointColors();
  m_Tris.reset(new unsigned int[NumTris*3]);
  m_NumTris = NumTris;
  //m_TriFlatVerts = new float[NumTris*3*3];
  //m_TriFlatNormals = new float[NumTris*3*3];
  /*unsigned int c;
  for (c=0; c<NumTris*3*3; c++) {
    m_TriFlatVerts[c] = 0.0;
    m_TriFlatNormals[c] = 0.0;
    }*/
  m_bExtentsReady = false;
}

void Geometry::AllocateTriTexCoords()
{
  ClearTriTexCoords();
  //m_TriVertTexCoords = new float[m_NumTriVerts*3];
  AllocatePointTexCoords();
  unsigned int c;
  for (c=0; c<m_NumTriVerts*3; c++) {
    m_TriVertTexCoords[c] = 0.0f;
  }
  /*
  m_TriFlatTexCoords = new float[m_NumTris*3*3];
  for (c=0; c<m_NumTris*3*3; c++) {
    m_TriFlatTexCoords[c] = 0.0f;
  }
  */
}

void Geometry::AllocateTriVertColors()
{
  ClearTriVertColors();
  //m_TriVertColors = new float[m_NumTriVerts*3];
  AllocatePointColors();
  unsigned int c;
  for (c=0; c<m_NumTriVerts*3; c++) {
    m_TriVertColors[c] = m_DiffuseColor[c%3];//1.0f;
  }
}

void Geometry::cross(float* dest, const float* v1, const float* v2)
{

  dest[0] = v1[1]*v2[2] - v1[2]*v2[1];
  dest[1] = v1[2]*v2[0] - v1[0]*v2[2];
  dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

void Geometry::normalize(float* v)
{
  float len = (float)sqrt(
			  v[0] * v[0] +
			  v[1] * v[1] +
			  v[2] * v[2]);
  if (len!=0.0) {
    v[0]/=len; //*biggestDim;
    v[1]/=len; //*biggestDim;
    v[2]/=len; //*biggestDim;
  }
  else {
    v[0] = 1.0;
  }
}

void Geometry::add(float* dest, const float* v)
{
  dest[0] += v[0];
  dest[1] += v[1];
  dest[2] += v[2];
}

void Geometry::set(float* dest, const float* v)
{
  dest[0] = v[0];
  dest[1] = v[1];
  dest[2] = v[2];
}

void Geometry::CalculateQuadSmoothNormals()
{
  unsigned int c, v0, v1, v2, v3;
  float normal[3];

  float zero[3] = {0.0f, 0.0f, 0.0f};

  if(!m_QuadVertNormals)
    AllocatePointNormals();

  for (c=0; c<m_NumQuadVerts; c++) {
    set(m_QuadVertNormals.get()+c*3, zero);
  }

		
  // for each Quadangle
  for (c=0; c<m_NumQuads; c++) {
    v0 = m_Quads[c*4+0];
    v1 = m_Quads[c*4+1];
    v2 = m_Quads[c*4+2];
    v3 = m_Quads[c*4+3];
    CalculateQuadNormal(normal, v0, v1, v3);
    add(m_QuadVertNormals.get()+v0*3, normal);
    add(m_QuadVertNormals.get()+v1*3, normal);
    add(m_QuadVertNormals.get()+v3*3, normal);
    CalculateQuadNormal(normal, v2, v3, v1);
    add(m_QuadVertNormals.get()+v2*3, normal);
    add(m_QuadVertNormals.get()+v3*3, normal);
    add(m_QuadVertNormals.get()+v1*3, normal);
  }
	
  // normalize the vectors
  for (c=0; c<m_NumQuadVerts; c++) {
    normalize(m_QuadVertNormals.get()+c*3);
  }
  m_bQuadSmoothNormalsReady = true;
}

void Geometry::CalculateTriSmoothNormals()
{
  unsigned int c, v0, v1, v2;
  float normal[3];
		
  float zero[3] = {0.0f, 0.0f, 0.0f};

  if(!m_TriVertNormals)
    AllocatePointNormals();

  for (c=0; c<m_NumTriVerts; c++) {
    set(m_TriVertNormals.get()+c*3, zero);
  }

  // for each triangle
  for (c=0; c<m_NumTris; c++) {
    v0 = m_Tris[c*3+0];
    v1 = m_Tris[c*3+1];
    v2 = m_Tris[c*3+2];
    CalculateTriangleNormal(normal, v0, v1, v2);
    add(m_TriVertNormals.get()+v0*3, normal);
    add(m_TriVertNormals.get()+v1*3, normal);
    add(m_TriVertNormals.get()+v2*3, normal);
  }
	
  // normalize the vectors
  for (c=0; c<m_NumTriVerts; c++) {
    normalize(m_TriVertNormals.get()+c*3);
  }
  m_bTriSmoothNormalsReady = true;
}

//TODO: this is broken, we need just 1 normal for each face
void Geometry::CalculateQuadFlatNormals()
{
  unsigned int c, v0, v1, v2, v3;
  float normal1[3], normal2[3];

  float zero[3] = {0.0f, 0.0f, 0.0f};

  if(!m_QuadFlatNormals)
    AllocatePointFlatNormals();

  for (c=0; c<m_NumQuadVerts; c++) {
    set(m_QuadFlatNormals.get()+c*3, zero);
  }

  for (c=0; c<m_NumQuads; c++) { // for every quad
    v0 = m_Quads[c*4+0];
    v1 = m_Quads[c*4+1];
    v2 = m_Quads[c*4+2];
    v3 = m_Quads[c*4+3];
    CalculateQuadNormal(normal1, v0, v1, v3);
    CalculateQuadNormal(normal2, v2, v3, v1);
    add(normal1, normal2);
    normalize(normal1);
    //set(m_QuadFlatVerts+c*12+0*3, m_QuadVerts+v0*3); // set the verts
    //set(m_QuadFlatVerts+c*12+1*3, m_QuadVerts+v1*3);
    //set(m_QuadFlatVerts+c*12+2*3, m_QuadVerts+v2*3);
    //set(m_QuadFlatVerts+c*12+3*3, m_QuadVerts+v3*3);
    //if (m_QuadVertTexCoords) {
    //  set(m_QuadFlatTexCoords+c*12+0*3, m_QuadVertTexCoords+v0*3); // set the tex coords
    //  set(m_QuadFlatTexCoords+c*12+1*3, m_QuadVertTexCoords+v1*3);
    //  set(m_QuadFlatTexCoords+c*12+2*3, m_QuadVertTexCoords+v2*3);
    //  set(m_QuadFlatTexCoords+c*12+3*3, m_QuadVertTexCoords+v3*3);
    //}
    //set(m_QuadFlatNormals+c*12+0*3, normal1); // set the normals
    //set(m_QuadFlatNormals+c*12+1*3, normal1);
    //set(m_QuadFlatNormals+c*12+2*3, normal1);
    //set(m_QuadFlatNormals+c*12+3*3, normal1);
    set(m_QuadFlatNormals.get()+v0*3, normal1);
    set(m_QuadFlatNormals.get()+v1*3, normal1);
    set(m_QuadFlatNormals.get()+v2*3, normal1);
    set(m_QuadFlatNormals.get()+v3*3, normal1);
  }
  m_bQuadFlatNormalsReady = true;
}

//TODO: this is broken, we need just 1 normal for each face
void Geometry::CalculateTriFlatNormals()
{
  unsigned int c, v0, v1, v2;
  float normal[3];

  float zero[3] = {0.0f, 0.0f, 0.0f};

  if(!m_TriFlatNormals)
    AllocatePointFlatNormals();

  for (c=0; c<m_NumQuadVerts; c++) {
    set(m_TriFlatNormals.get()+c*3, zero);
  }

  for (c=0; c<m_NumTris; c++) { // for every triangle
    v0 = m_Tris[c*3+0];
    v1 = m_Tris[c*3+1];
    v2 = m_Tris[c*3+2];
    CalculateTriangleNormal(normal, v0, v1, v2);
    normalize(normal);
    //set(m_TriFlatVerts+c*9+0*3, m_TriVerts+v0*3); // set the verts
    //set(m_TriFlatVerts+c*9+1*3, m_TriVerts+v1*3);
    //set(m_TriFlatVerts+c*9+2*3, m_TriVerts+v2*3);
    //if (m_TriVertTexCoords) {
    //set(m_TriFlatTexCoords+c*9+0*3, m_TriVertTexCoords+v0*3); // set the tex coords
    //set(m_TriFlatTexCoords+c*9+1*3, m_TriVertTexCoords+v1*3);
    //set(m_TriFlatTexCoords+c*9+2*3, m_TriVertTexCoords+v2*3);
    //}
    //set(m_TriFlatNormals+c*9+0*3, normal); // set the normals
    //set(m_TriFlatNormals+c*9+1*3, normal);
    //set(m_TriFlatNormals+c*9+2*3, normal);
    set(m_TriFlatNormals.get()+v0*3, normal);
    set(m_TriFlatNormals.get()+v1*3, normal);
    set(m_TriFlatNormals.get()+v2*3, normal);
  }
  m_bTriFlatNormalsReady = true;
}

void Geometry::SetTriNormalsReady()
{
  m_bTriSmoothNormalsReady = true;
}

void Geometry::SetQuadNormalsReady()
{
  m_bQuadSmoothNormalsReady = true;
}

void Geometry::SetDiffusedColor( float r, float g, float b )
{
  m_DiffuseColor[0] = r;
  m_DiffuseColor[1] = g;
  m_DiffuseColor[2] = b;
}

void Geometry::SetSpecularColor( float r, float g, float b )
{
  m_SpecularColor[0] = r;
  m_SpecularColor[1] = g;
  m_SpecularColor[2] = b;
}

void Geometry::SetAmbientColor( float r, float g, float b )
{
  m_AmbientColor[0] = r;
  m_AmbientColor[1] = g;
  m_AmbientColor[2] = b;
}

void Geometry::SetShininess( float s )
{
  m_Shininess = s;
}

void Geometry::SetLineWidth( float lineWidth )
{
  m_LineWidth = lineWidth;
}

void Geometry::SetPointSize( float pointSize )
{
  m_PointSize = pointSize;
}

void Geometry::GetReadyToDrawWire()
{
  if (!m_bExtentsReady) {
    CalculateExtents();
  }
}

void Geometry::GetReadyToDrawSmooth()
{
  if (!m_bExtentsReady) {
    CalculateExtents();
  }
  if ((m_NumQuads > 0) && (!m_bQuadSmoothNormalsReady)) {
    CalculateQuadSmoothNormals();
  }
  if ((m_NumTris > 0) && (!m_bTriSmoothNormalsReady)) {
    CalculateTriSmoothNormals();
  }
}

void Geometry::GetReadyToDrawFlat()
{
  if (!m_bExtentsReady) {
    CalculateExtents();
  }
  if ((m_NumQuads > 0) && (!m_bQuadFlatNormalsReady)) {
    CalculateQuadFlatNormals();
  }
  if ((m_NumTris > 0) && (!m_bTriFlatNormalsReady)) {
    CalculateTriFlatNormals();
  }
}

Geometry& Geometry::merge(const Geometry& geometry)
{
  Geometry merged;

  //anything to copy?
  if(m_NumPoints + geometry.m_NumPoints == 0) return *this;

  //vertex array
  merged.AllocatePoints(m_NumPoints + geometry.m_NumPoints);
  memcpy(merged.m_Points.get(),
	 m_Points.get(),
	 m_NumPoints*3*sizeof(float));
  memcpy(merged.m_Points.get() + m_NumPoints*3,
	 geometry.m_Points.get(),
	 geometry.m_NumPoints*3*sizeof(float));
  
  //color array
  merged.AllocatePointColors();
  if(m_PointColors)
    memcpy(merged.m_PointColors.get(),
	   m_PointColors.get(),
	   m_NumPoints*3*sizeof(float));
  if(geometry.m_PointColors)
    memcpy(merged.m_PointColors.get() + m_NumPoints*3,
	   geometry.m_PointColors.get(),
	   geometry.m_NumPoints*3*sizeof(float));
  
  //texture coordinates
  merged.AllocatePointTexCoords();
  if(m_PointTexCoords)
    memcpy(merged.m_PointTexCoords.get(),
	   m_PointTexCoords.get(),
	   m_NumPoints*2*sizeof(float));
  if(geometry.m_PointTexCoords)
    memcpy(merged.m_PointTexCoords.get() + m_NumPoints*2,
	   geometry.m_PointTexCoords.get(),
	   geometry.m_NumPoints*2*sizeof(float));

  //scalars
  merged.AllocatePointScalars();
  if(m_PointScalars)
    memcpy(merged.m_PointScalars.get(),
	   m_PointScalars.get(),
	   m_NumPoints*1*sizeof(float));
  if(geometry.m_PointScalars)
    memcpy(merged.m_PointScalars.get() + m_NumPoints*1,
	   geometry.m_PointScalars.get(),
	   geometry.m_NumPoints*1*sizeof(float));

  //lines
  if(m_NumLines + geometry.m_NumLines > 0)
    {
      merged.m_Lines.reset(new unsigned int[m_NumLines*2 + geometry.m_NumLines*2]);
      merged.m_NumLines = m_NumLines + geometry.m_NumLines;
      memcpy(merged.m_Lines.get(),
	     m_Lines.get(),
	     m_NumLines*2*sizeof(unsigned int));
      memcpy(merged.m_Lines.get() + m_NumLines*2,
	     geometry.m_Lines.get(),
	     geometry.m_NumLines*2*sizeof(unsigned int));
      for(unsigned int i = 0; i < geometry.m_NumLines*2; i++)
	merged.m_Lines[m_NumLines*2 + i] += m_NumPoints;
    }

  //quads
  if(m_NumQuads + geometry.m_NumQuads > 0)
    {
      merged.m_Quads.reset(new unsigned int[m_NumQuads*4 + geometry.m_NumQuads*4]);
      merged.m_NumQuads = m_NumQuads + geometry.m_NumQuads;
      memcpy(merged.m_Quads.get(),
	     m_Quads.get(),
	     m_NumQuads*4*sizeof(unsigned int));
      memcpy(merged.m_Quads.get() + m_NumQuads*4,
	     geometry.m_Quads.get(),
	     geometry.m_NumQuads*4*sizeof(unsigned int));
      for(unsigned int i = 0; i < geometry.m_NumQuads*4; i++)
	merged.m_Quads[m_NumQuads*4 + i] += m_NumPoints;
    }

  //tris
  if(m_NumTris + geometry.m_NumTris > 0)
    {
      merged.m_Tris.reset(new unsigned int[m_NumTris*3 + geometry.m_NumTris*3]);
      merged.m_NumTris = m_NumTris + geometry.m_NumTris;
      memcpy(merged.m_Tris.get(),
	     m_Tris.get(),
	     m_NumTris*3*sizeof(unsigned int));
      memcpy(merged.m_Tris.get() + m_NumTris*3,
	     geometry.m_Tris.get(),
	     geometry.m_NumTris*3*sizeof(unsigned int));
      for(unsigned int i = 0; i < geometry.m_NumTris*3; i++)
	merged.m_Tris[m_NumTris*3 + i] += m_NumPoints;
    }

  *this = merged;
  return *this;
}

void Geometry::quads2tris()
{
  m_Tris.reset(new unsigned int[m_NumQuads*2*3]);
  m_NumTris = m_NumQuads*2;
  for(unsigned int i = 0; i < m_NumQuads; i++)
    {
      m_Tris[(i*2*3) + 0] = m_Quads[i*4 + 0];
      m_Tris[(i*2*3) + 1] = m_Quads[i*4 + 1];
      m_Tris[(i*2*3) + 2] = m_Quads[i*4 + 3];

      m_Tris[(i*2*3) + 3] = m_Quads[i*4 + 1];
      m_Tris[(i*2*3) + 4] = m_Quads[i*4 + 2];
      m_Tris[(i*2*3) + 5] = m_Quads[i*4 + 3];
    }
  m_Quads.reset();
  m_NumQuads = 0;
}

void Geometry::CheckMin(bool& initialized, float* CurrentMin, float* Check)
{
  if (initialized) {
    CurrentMin[0] = (CurrentMin[0]<Check[0]?CurrentMin[0]:Check[0]);
    CurrentMin[1] = (CurrentMin[1]<Check[1]?CurrentMin[1]:Check[1]);
    CurrentMin[2] = (CurrentMin[2]<Check[2]?CurrentMin[2]:Check[2]);
  }
  else {
    CurrentMin[0] = Check[0];
    CurrentMin[1] = Check[1];
    CurrentMin[2] = Check[2];
    initialized = true;
  }
}

void Geometry::CheckMax(bool& initialized, float* CurrentMax, float* Check)
{
  if (initialized) {
    CurrentMax[0] = (CurrentMax[0]>Check[0]?CurrentMax[0]:Check[0]);
    CurrentMax[1] = (CurrentMax[1]>Check[1]?CurrentMax[1]:Check[1]);
    CurrentMax[2] = (CurrentMax[2]>Check[2]?CurrentMax[2]:Check[2]);
  }
  else {
    CurrentMax[0] = Check[0];
    CurrentMax[1] = Check[1];
    CurrentMax[2] = Check[2];
    initialized = true;
  }
}

void Geometry::CalculateExtents()
{
  unsigned int c;
  bool min_initialized = false, max_initialized = false;

  for(c=0;c<m_NumPoints;c++)
    {
      CheckMin(min_initialized, m_Min, m_Points.get()+c*3);
      CheckMax(max_initialized, m_Max, m_Points.get()+c*3);
    }
  for(c=0;c<m_NumLineVerts;c++)
    {
      CheckMin(min_initialized, m_Min, m_LineVerts.get()+c*3);
      CheckMax(max_initialized, m_Max, m_LineVerts.get()+c*3);
    }
  for(c=0;c<m_NumTriVerts;c++)
    {
      CheckMin(min_initialized, m_Min, m_TriVerts.get()+c*3);
      CheckMax(max_initialized, m_Max, m_TriVerts.get()+c*3);
    }
  for(c=0;c<m_NumQuadVerts;c++)
    {
      CheckMin(min_initialized, m_Min, m_QuadVerts.get()+c*3);
      CheckMax(max_initialized, m_Max, m_QuadVerts.get()+c*3);
    }

  m_Center[0]=( m_Max[0] + m_Min[0] ) / 2.0f;
  m_Center[1]=( m_Max[1] + m_Min[1] ) / 2.0f;
  m_Center[2]=( m_Max[2] + m_Min[2] ) / 2.0f;
  m_bExtentsReady = true;
}

void Geometry::conv(cvcraw_geometry::geometry_t& dest,
		    const LBIE::geoframe& src)
{
  dest.clear();

  dest.points.resize(src.verts.size());
  for(size_t i = 0; i < dest.points.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.points[i][j] = src.verts[i][j];

  dest.boundary.resize(src.bound_sign.size());
  for(size_t i = 0; i < dest.boundary.size(); i++)
    dest.boundary[i] = src.bound_sign[i];

  dest.normals.resize(src.normals.size());
  for(size_t i = 0; i < dest.normals.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.normals[i][j] = src.normals[i][j];

  dest.colors.resize(src.color.size());
  for(size_t i = 0; i < dest.colors.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.colors[i][j] = src.color[i][j];

  dest.tris.resize(src.triangles.size());
  for(size_t i = 0; i < dest.tris.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.tris[i][j] = src.triangles[i][j];

  dest.quads.resize(src.quads.size());
  for(size_t i = 0; i < dest.quads.size(); i++)
    for(int j = 0; j < 4; j++)
      dest.quads[i][j] = src.quads[i][j];

  dest.min_ext[0] = src.min_x;
  dest.min_ext[1] = src.min_y;
  dest.min_ext[2] = src.min_z;
  dest.max_ext[0] = src.max_x;
  dest.max_ext[1] = src.max_y;
  dest.max_ext[2] = src.max_z;
}
void Geometry::convto(CVCGEOM_NAMESPACE::cvcgeom_t & dest)
{  
  //dest.clear();

  for(size_t i = 0; i < m_NumTriVerts; i++) {
    CVCGEOM_NAMESPACE::cvcgeom_t::point_t newVertex;
    newVertex[0] = m_TriVerts[3*i+0];
    newVertex[1] = m_TriVerts[3*i+1];
    newVertex[2] = m_TriVerts[3*i+2];
    dest.points().push_back(newVertex);
  }

//  dest.boundary.resize(src.bound_sign.size());
 // for(size_t i = 0; i < dest.boundary.size(); i++)
  //  dest.boundary[i] = src.bound_sign[i];


  /*
  dest.normals.resize(m_NumTriVerts);
  for(size_t i = 0; i < dest.normals.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.normals[i][j] = m_TriVertNormals[3*i + j];
  */


  //dest.colors.resize(m_NumTriVerts);
  for(size_t i = 0; i < m_NumTriVerts; i++) {
    CVCGEOM_NAMESPACE::cvcgeom_t::color_t meshColor;
    if (m_TriVertColors) {	
      meshColor[0] = m_TriVertColors[3*i + 0];
      meshColor[1] = m_TriVertColors[3*i + 1];
      meshColor[2] = m_TriVertColors[3*i + 2];
    } else {
      meshColor[0] = 0.0; meshColor[1] = 1.0; meshColor[2] = 0.001;
    }
    dest.colors().push_back(meshColor);
  }

  //dest.tris.resize(m_NumTris);
  for(size_t i = 0; i < m_NumTris; i++) {
    CVCGEOM_NAMESPACE::cvcgeom_t::triangle_t newTri;
    newTri[0] = m_Tris[3*i+0];
    newTri[1] = m_Tris[3*i+1];
    newTri[2] = m_Tris[3*i+2];
    dest.triangles().push_back(newTri);
  }


  /*
  dest.quads.resize(m_NumQuads);
  for(size_t i = 0; i < dest.quads.size(); i++)
    for(int j = 0; j < 4; j++)
      dest.quads[i][j] = m_Quads[4*i+j];
  */
  
  /*
  dest.min_ext[0] = m_Min[0];
  dest.min_ext[1] = m_Min[1];
  dest.min_ext[2] = m_Min[2];
  dest.max_ext[0] = m_Max[0];
  dest.max_ext[1] = m_Max[1];
  dest.max_ext[2] = m_Max[2]; 
  */
} 

void Geometry::conv(LBIE::geoframe& dest,
		    const cvcraw_geometry::geometry_t& src)
{
  dest.reset();
  
  dest.verts.resize(src.points.size());
  for(size_t i = 0; i < dest.verts.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.verts[i][j] = src.points[i][j];

  dest.bound_sign.resize(src.boundary.size());
  for(size_t i = 0; i < dest.bound_sign.size(); i++)
    dest.bound_sign[i] = src.boundary[i];

  dest.normals.resize(src.normals.size());
  for(size_t i = 0; i < dest.normals.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.normals[i][j] = src.normals[i][j];

  dest.color.resize(src.colors.size());
  for(size_t i = 0; i < dest.color.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.color[i][j] = src.colors[i][j];

  dest.triangles.resize(src.tris.size());
  for(size_t i = 0; i < dest.triangles.size(); i++)
    for(int j = 0; j < 3; j++)
      dest.triangles[i][j] = src.tris[i][j];

  dest.quads.resize(src.quads.size());
  for(size_t i = 0; i < dest.quads.size(); i++)
    for(int j = 0; j < 4; j++)
      dest.quads[i][j] = src.quads[i][j];

  dest.numverts = dest.verts.size();
  dest.numtris = dest.num_tris = dest.triangles.size();
  dest.numquads = dest.quads.size();
  dest.numhexas = dest.quads.size()/6;

  dest.min_x = src.min_ext[0];
  dest.min_y = src.min_ext[1];
  dest.min_z = src.min_ext[2];
  dest.max_x = src.max_ext[0];
  dest.max_y = src.max_ext[1];
  dest.max_z = src.max_ext[2];

  if(!dest.bound_sign.empty())
    {
      dest.bound_tri.resize(src.tris.size());
      for(size_t i = 0; i < dest.bound_tri.size(); i++)
	if(dest.bound_sign[dest.triangles[i][0]] &&
	   dest.bound_sign[dest.triangles[i][1]] &&
	   dest.bound_sign[dest.triangles[i][2]])
	  dest.bound_tri[i] = 1;
    }

  const cvcraw_geometry::geometry_t &geom = src;
  if(!geom.tris.empty())
    dest.mesh_type = LBIE::geoframe::SINGLE;
  if(!geom.quads.empty())
    dest.mesh_type = LBIE::geoframe::QUAD;
  if(!geom.boundary.empty() && !geom.tris.empty())
    dest.mesh_type = LBIE::geoframe::TETRA;;
  if(!geom.boundary.empty() && !geom.quads.empty())
    dest.mesh_type = LBIE::geoframe::HEXA;
}

Geometry Geometry::conv(const Geometry& src)
{
  Geometry ret;

  //Geometry part
  ret.CopyPoints(src);
  ret.CopyPointNormals(src);
  ret.CopyPointFlatNormals(src);
  ret.CopyPointColors(src);
  ret.CopyPointTexCoords(src);
  ret.CopyPointScalars(src);
  ret.CopyLines(src);
  ret.CopyQuads(src);
  ret.CopyTris(src);
  ret.CopyColors(src);

  //GeoFrame part
  ret.m_GeoFrame.reset(new LBIE::geoframe);
  LBIE::copyGeometryToGeoframe(ret, *ret.m_GeoFrame);

  //geometry_t part
  conv(ret.m_SceneGeometry.geometry, *ret.m_GeoFrame);
  cvcraw_geometry::geometry_t &geom = ret.m_SceneGeometry.geometry;
  scene_geometry_t::render_mode_t &mode = ret.m_SceneGeometry.render_mode;
  if(!geom.lines.empty())
    mode = scene_geometry_t::LINES;
  if(!geom.tris.empty())
    mode = scene_geometry_t::TRIANGLES;
  if(!geom.quads.empty())
    mode = scene_geometry_t::QUADS;

  if(!geom.boundary.empty() && !geom.tris.empty())
    {
      geom = geom.generate_wire_interior();
      mode = scene_geometry_t::TETRA;
    }
  else if(!geom.boundary.empty() && !geom.quads.empty())
    {
      geom = geom.generate_wire_interior();
      mode = scene_geometry_t::HEXA;
    }

  return ret;
}

Geometry Geometry::conv(const LBIE::geoframe& src)
{
  Geometry ret;

  //GeoFrame part
  ret.m_GeoFrame.reset(new LBIE::geoframe(src));

  //geometry_t part
  conv(ret.m_SceneGeometry.geometry, src);
  cvcraw_geometry::geometry_t &geom = ret.m_SceneGeometry.geometry;
  scene_geometry_t::render_mode_t &mode = ret.m_SceneGeometry.render_mode;
  if(!geom.lines.empty())
    mode = scene_geometry_t::LINES;
  if(!geom.tris.empty())
    mode = scene_geometry_t::TRIANGLES;
  if(!geom.quads.empty())
    mode = scene_geometry_t::QUADS;

  if(!geom.boundary.empty() && !geom.tris.empty())
    {
      geom = geom.generate_wire_interior();
      mode = scene_geometry_t::TETRA;
    }
  else if(!geom.boundary.empty() && !geom.quads.empty())
    {
      geom = geom.generate_wire_interior();
      mode = scene_geometry_t::HEXA;
    }

  //Geometry part, only copy boundary surface
  cvcraw_geometry::geometry_t surf = 
    ret.m_SceneGeometry.geometry.tri_surface();
  LBIE::geoframe geo_surf;
  conv(geo_surf, surf);
  LBIE::copyGeoframeToGeometry(geo_surf, ret);

  return ret;
}

Geometry Geometry::conv(const cvcraw_geometry::geometry_t& src)
{
  Geometry ret;

  //GeoFrame part
  ret.m_GeoFrame.reset(new LBIE::geoframe);
  conv(*ret.m_GeoFrame, src);

  //geometry_t part
  ret.m_SceneGeometry.geometry = src;
  cvcraw_geometry::geometry_t &geom = ret.m_SceneGeometry.geometry;
  scene_geometry_t::render_mode_t &mode = ret.m_SceneGeometry.render_mode;
  
  if(!geom.points.empty())
  	mode = scene_geometry_t::POINTS;
  if(!geom.lines.empty())
    mode = scene_geometry_t::LINES;
  if(!geom.tris.empty())
    mode = scene_geometry_t::TRIANGLES;
  if(!geom.quads.empty())
    mode = scene_geometry_t::QUADS;

  if(!geom.boundary.empty() && !geom.tris.empty())
    {
      geom = geom.generate_wire_interior();
      mode = scene_geometry_t::TETRA;
    }
  else if(!geom.boundary.empty() && !geom.quads.empty())
    {
      geom = geom.generate_wire_interior();
      mode = scene_geometry_t::HEXA;
    }  

  //Geometry part, only copy boundary surface
  cvcraw_geometry::geometry_t surf = 
    ret.m_SceneGeometry.geometry.tri_surface();
  LBIE::geoframe geo_surf;
  conv(geo_surf, surf);
  LBIE::copyGeoframeToGeometry(geo_surf, ret);
  
/*  if(mode== scene_geometry_t::POINTS)
  {
    ret.AllocateTris(geom.points.size(), 0);
  	for(int i=0; i< geom.points.size(); i++)
	 for(int j=0; j<3; j++)
		ret.m_TriVerts[3*i+j]= src.points[i][j];
	if(geom.colors.size()== geom.points.size())
	 for(int i=0; i< geom.colors.size(); i++)
	 	for(int j=0; j<3; j++)
		ret.m_TriVertColors[3*i + j] = geom.colors[i][j]; 
  } */ 
	
  return ret;
}

void Geometry::CalculateTriangleNormal(float* norm, unsigned int v0, unsigned int v1, unsigned int v2)
{
  float vec1[3], vec2[3];
  vec1[0] = vec2[0] = -m_TriVerts[v0*3+0];
  vec1[1] = vec2[1] = -m_TriVerts[v0*3+1];
  vec1[2] = vec2[2] = -m_TriVerts[v0*3+2];
	
  vec1[0] += m_TriVerts[v1*3+0];
  vec1[1] += m_TriVerts[v1*3+1];
  vec1[2] += m_TriVerts[v1*3+2];
	
  vec2[0] += m_TriVerts[v2*3+0];
  vec2[1] += m_TriVerts[v2*3+1];
  vec2[2] += m_TriVerts[v2*3+2];
	
  cross(norm, vec1, vec2);
}

void Geometry::CalculateQuadNormal(float* norm, unsigned int v0, unsigned int v1, unsigned int v2)
{
  float vec1[3], vec2[3];
  vec1[0] = vec2[0] = -m_QuadVerts[v0*3+0];
  vec1[1] = vec2[1] = -m_QuadVerts[v0*3+1];
  vec1[2] = vec2[2] = -m_QuadVerts[v0*3+2];
	
  vec1[0] += m_QuadVerts[v1*3+0];
  vec1[1] += m_QuadVerts[v1*3+1];
  vec1[2] += m_QuadVerts[v1*3+2];
	
  vec2[0] += m_QuadVerts[v2*3+0];
  vec2[1] += m_QuadVerts[v2*3+1];
  vec2[2] += m_QuadVerts[v2*3+2];


	
  cross(norm, vec1, vec2);
}
