/*
  Copyright 2002-2005 The University of Texas at Austin

        Authors: Anthony Thane <thanea@ices.utexas.edu>
                 Vinay Siddavanahalli <skvinay@cs.utexas.edu>
                 Joe Rivera <transfix@ices.utexas.edu>
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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: Geometry.h 4741 2011-10-21 21:22:06Z transfix $ */

// Geometry.h: interface for the Geometry class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GEOMETRY_H__A99E4F7E_DEED_48DB_9F49_0DBB21EC4211__INCLUDED_)
#define AFX_GEOMETRY_H__A99E4F7E_DEED_48DB_9F49_0DBB21EC4211__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <LBIE/LBIE_geoframe.h>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_array.hpp>
#include <cvcraw_geometry/cvcgeom.h>
#include <cvcraw_geometry/scene_geometry.h>

///\class Geometry Geometry.h
///\author Anthony Thane
///\author Joe Rivera
///\brief The Geometry class is a simple container for 3D surfaces
class Geometry {
public:
  Geometry();
  Geometry(const Geometry &copy);
  Geometry &operator=(const Geometry &copy);
  virtual ~Geometry();

  ///\fn void ZeroMembers()
  ///\brief Sets all member variables to zero
  void ZeroMembers();
  ///\fn void InitializeColors()
  ///\brief Initializes member variables pertaining to color
  void InitializeColors();

  ///\fn void CopyPoints(const Geometry& copy)
  ///\brief Copies points from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyPoints(const Geometry &copy);
  ///\fn void CopyPointNormals(const Geometry& copy)
  ///\brief Copies normals from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyPointNormals(const Geometry &copy);
  ///\fn void CopyPointNormals(const Geometry& copy)
  ///\brief Copies normals from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyPointFlatNormals(const Geometry &copy);
  ///\fn void CopyPointColors(const Geometry& copy)
  ///\brief Copies normals from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyPointColors(const Geometry &copy);
  ///\fn void CopyPointTexCoords(const Geometry& copy)
  ///\brief Copies normals from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyPointTexCoords(const Geometry &copy);
  ///\fn void CopyPointScalars(const Geometry& copy)
  ///\brief Copies point scalars from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyPointScalars(const Geometry &copy);
  ///\fn void CopyLines(const Geometry& copy)
  ///\brief Copies lines from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyLines(const Geometry &copy);
  ///\fn void CopyLineColors(const Geometry& copy)
  ///\brief Copies line colors from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyLineColors(const Geometry &copy);
  ///\fn void CopyQuads(const Geometry& copy)
  ///\brief Copies quads from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyQuads(const Geometry &copy);
  ///\fn void CopyQuadTexCoords(const Geometry& copy)
  ///\brief Copies quad texture coordinates from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyQuadTexCoords(const Geometry &copy);
  ///\fn void CopyTris(const Geometry& copy)
  ///\brief Copies triangles from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyTris(const Geometry &copy);
  ///\fn void CopyTriTexCoords(const Geometry& copy)
  ///\brief Copies triangle texture coordinates from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyTriTexCoords(const Geometry &copy);
  ///\fn void CopyTriVertColors(const Geometry& copy)
  ///\brief Copies triangle vertex colors from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyTriVertColors(const Geometry &copy);
  ///\fn void CopyColors(const Geometry& copy)
  ///\brief Copies colors from another Geometry object
  ///\param copy A Geometry object to copy data from
  void CopyColors(const Geometry &copy);

  ///\fn void ClearGeometry()
  ///\brief Clears all geometry data held by the object (like ZeroMembers, but
  /// used for objects that have buffers allocated already)
  void ClearGeometry();
  ///\fn void ClearPoints()
  ///\brief Clears any point data held by the object
  void ClearPoints();
  ///\fn void ClearPoints()
  ///\brief Clears any normal data held by the object
  void ClearPointNormals();
  ///\fn void ClearPointFlatNormals()
  ///\brief Clears any flat normal data held by the object
  void ClearPointFlatNormals();
  ///\fn void ClearPoints()
  ///\brief Clears any vertex color data held by the object
  void ClearPointColors();
  ///\fn void ClearPoints()
  ///\brief Clears any vertex texture coordinate data held by the object
  void ClearPointTexCoords();
  ///\fn void ClearPoints()
  ///\brief Clears any point scalar data held by the object
  void ClearPointScalars();
  ///\fn void ClearLines()
  ///\brief Clears any line data held by the object
  void ClearLines();
  ///\fn void ClearLineColors()
  ///\brief Clears any line color data held by the object
  void ClearLineColors();
  ///\fn void ClearQuads()
  ///\brief Clears any quad data held by the object
  void ClearQuads();
  ///\fn void ClearQuadTexCoords()
  ///\brief Clears any quad texture coordinate data held by the object
  void ClearQuadTexCoords();
  ///\fn void ClearTris()
  ///\brief Clears any triangle data held by the object
  void ClearTris();
  ///\fn void ClearTriTexCoords()
  ///\brief Clears any triangle texture coordinate data held by the object
  void ClearTriTexCoords();
  ///\fn void ClearTriVertColors()
  ///\brief Clears any triangle vertex color data held by the object
  void ClearTriVertColors();

  ///\fn void AllocatePoints(unsigned int NumPoints)
  ///\brief Allocates memory for point data
  ///\param NumPoints The number of points
  void AllocatePoints(unsigned int NumPoints);
  ///\fn void AllocatePointNormals()
  ///\brief Allocates memory for normal data
  ///\param NumPoints The number of points
  void AllocatePointNormals();
  ///\fn void AllocatePointFlatNormals()
  ///\brief Allocates memory for normal data
  ///\param NumPoints The number of points
  void AllocatePointFlatNormals();
  ///\fn void AllocatePointColors()
  ///\brief Allocates memory for color data
  ///\param NumPoints The number of points
  void AllocatePointColors();
  ///\fn void AllocatePointTexCoords()
  ///\brief Allocates memory for tex coord data
  ///\param NumPoints The number of points
  void AllocatePointTexCoords();
  ///\fn void AllocatePointScalars()
  ///\brief Allocates memory for point scalar data
  void AllocatePointScalars();
  ///\fn void AllocateLines(unsigned int NumLineVerts, unsigned int NumLines)
  ///\brief Allocates memory for line data
  ///\param NumLineVerts The number of vertices
  ///\param NumLines The number of lines
  void AllocateLines(unsigned int NumLineVerts, unsigned int NumLines);
  ///\fn void AllocateLineColors()
  ///\brief Allocates memory for line color data
  void AllocateLineColors();
  ///\fn void AllocateQuads(unsigned int NumQuadVerts, unsigned int NumQuads)
  ///\brief Allocates memory for quad data
  ///\param NumQuadVerts The number of vertices
  ///\param NumQuads The number of quads
  void AllocateQuads(unsigned int NumQuadVerts, unsigned int NumQuads);
  ///\fn void AllocateQuadTexCoords()
  ///\brief Allocates memory for quad texture coordinate data
  void AllocateQuadTexCoords();
  ///\fn void AllocateTris(unsigned int NumTriVerts, unsigned int NumTris)
  ///\brief Allocates memory for triangle data
  ///\param The number of vertices
  ///\param The number of triangles
  void AllocateTris(unsigned int NumTriVerts, unsigned int NumTris);
  ///\fn void AllocateTriTexCoords()
  ///\brief Allocates memory for triangle texture coordinate data
  void AllocateTriTexCoords();
  ///\fn void AllocateTriVertColors()
  ///\brief Allocates memory for triangle vertex color data
  void AllocateTriVertColors();

  ///\fn void CalculateQuadSmoothNormals()
  ///\brief Calculates smooth normals for a quad mesh
  void CalculateQuadSmoothNormals();
  ///\fn void CalculateTriSmoothNormals()
  ///\brief Calculates smooth normals for a triangle mesh
  void CalculateTriSmoothNormals();
  ///\fn void CalculateQuadFlatNormals()
  ///\brief Calculates flat normals for a quad mesh
  void CalculateQuadFlatNormals();
  ///\fn void CalculateTriFlatNormals()
  ///\brief Calculates flat normals for a triangle mesh
  void CalculateTriFlatNormals();
  ///\fn void SetTriNormalsReady()
  ///\brief Sets the flag for smooth triangle normals to true. (useful if the
  /// normals were not computed by CalculateTriSmoothNormals)
  void SetTriNormalsReady();
  ///\fn void SetQuadNormalsReady()
  ///\brief Sets the flag for smooth quad normals to true. (useful if the
  /// normals were not computed by CalculateQuadSmoothNormals)
  void SetQuadNormalsReady();

  ///\fn void SetDiffusedColor( float r, float g, float b )
  ///\brief
  void SetDiffusedColor(float r, float g, float b);
  ///\fn void SetSpecularColor( float r, float g, float b )
  ///\brief
  void SetSpecularColor(float r, float g, float b);
  ///\fn void SetAmbientColor( float r, float g, float b )
  ///\brief
  void SetAmbientColor(float r, float g, float b);
  ///\fn void SetShininess( float s )
  ///\brief
  void SetShininess(float s);
  ///\fn void SetLineWidth( float lineWidth )
  ///\brief
  void SetLineWidth(float lineWidth);
  ///\fn void SetLineWidth( float lineWidth )
  ///\brief
  void SetPointSize(float pointSize);

  ///\fn void GetReadyToDrawWire()
  ///\brief
  void GetReadyToDrawWire();
  ///\fn void GetReadyToDrawSmooth()
  ///\brief
  void GetReadyToDrawSmooth();
  ///\fn void GetReadyToDrawFlat()
  ///\brief
  void GetReadyToDrawFlat();

  Geometry &merge(const Geometry &);
  void quads2tris(); // convert quad array to tri array

  float m_LineWidth;
  float m_PointSize;

  float m_DiffuseColor[3];
  float m_SpecularColor[3];
  float m_AmbientColor[3];
  float m_Shininess;

  // Unfortunately, clients are responsible for synchronizing the
  // different geometry reprensentations.

  boost::shared_array<float> m_Points;
  boost::shared_array<float> m_PointNormals;
  boost::shared_array<float> m_PointFlatNormals;
  boost::shared_array<float> m_PointColors;
  boost::shared_array<float> m_PointTexCoords;
  unsigned int m_NumPoints;
  boost::shared_array<float>
      m_PointScalars; // single scalar value assigned to each point

  boost::shared_array<float> &m_LineVerts;
  boost::shared_array<float> &m_LineColors;
  unsigned int &m_NumLineVerts;
  boost::shared_array<unsigned int> m_Lines;
  unsigned int m_NumLines;

  boost::shared_array<float> &m_QuadVerts;
  boost::shared_array<float> &m_QuadVertNormals;
  boost::shared_array<float> &m_QuadVertTexCoords;
  unsigned int &m_NumQuadVerts;
  boost::shared_array<unsigned int> m_Quads;
  unsigned int m_NumQuads;
  boost::shared_array<float> &m_QuadFlatVerts;
  boost::shared_array<float> &m_QuadFlatNormals;
  boost::shared_array<float> &m_QuadFlatTexCoords;
  bool m_bQuadFlatNormalsReady;
  bool m_bQuadSmoothNormalsReady;

  boost::shared_array<float> &m_TriVerts;
  boost::shared_array<float> &m_TriVertNormals;
  boost::shared_array<float> &m_TriVertColors;
  boost::shared_array<float> &m_TriVertTexCoords;
  unsigned int &m_NumTriVerts;
  boost::shared_array<unsigned int> m_Tris;
  unsigned int m_NumTris;
  boost::shared_array<float> &m_TriFlatVerts;
  boost::shared_array<float> &m_TriFlatNormals;
  boost::shared_array<float> &m_TriFlatTexCoords;
  bool m_bTriFlatNormalsReady;
  bool m_bTriSmoothNormalsReady;

  float m_Center[3];
  float m_Min[3];
  float m_Max[3];
  bool m_bExtentsReady;

  // This is a terrible hack but it will have to do for now!
  boost::scoped_ptr<LBIE::geoframe> m_GeoFrame;

  // this too...
  scene_geometry_t m_SceneGeometry;

  static void conv(cvcraw_geometry::geometry_t &dest,
                   const LBIE::geoframe &src);
  static void conv(LBIE::geoframe &dest,
                   const cvcraw_geometry::geometry_t &src);
  void convto(cvcraw_geometry::cvcgeom_t &dest);

  // Conversion functions.  They each return a 'complete' Geometry
  // object (i.e. an object that has it's 3 geometry represenstations
  // filled out).  They also make an attempt to choose the most appropriate
  // rendering mode for scene_geometry_t based on what geometry is provided
  static Geometry conv(const Geometry &src);
  static Geometry conv(const LBIE::geoframe &src);
  static Geometry conv(const cvcraw_geometry::geometry_t &src);

  static void cross(float *dest, const float *v1, const float *v2);
  static void normalize(float *v);
  static void add(float *dest, const float *v);
  static void set(float *dest, const float *v);
  static void CheckMin(bool &initialized, float *CurrentMin, float *Check);
  static void CheckMax(bool &initialized, float *CurrentMax, float *Check);

private:
  void CalculateTriangleNormal(float *norm, unsigned int v0, unsigned int v1,
                               unsigned int v2);
  void CalculateQuadNormal(float *norm, unsigned int v0, unsigned int v1,
                           unsigned int v2);
  void CalculateExtents();
};

#endif // !defined(AFX_GEOMETRY_H__A99E4F7E_DEED_48DB_9F49_0DBB21EC4211__INCLUDED_)
