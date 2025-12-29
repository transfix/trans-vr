/*
  Copyright 2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
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

#include <cvcalgo/cvcalgo.h>

#ifndef NO_MULTI_SDF
#include <multi_sdf/multi_sdf.h>
#endif

#ifndef NO_SDFLIB
#include <SignDistanceFunction/sdfLib.h>
#endif

#include <algorithm>
#include <cstring>
#include <iostream>

namespace cvcalgo {
/*
  // arand: seems to be out of date
#ifndef NO_VOLUMEROVER_EXTENSION
void CVCAlgoExtension::init()
{
  //placeholder for library initialization
  //TODO: add placeholder for library de-initialization
  //The point of this is that this is intended to be set up in the future
  //as a separate dynamic plugin.
  std::cerr << "loaded CVCAlgoExtension" << std::endl;
}

boost::shared_ptr<CVCAlgoExtension> _instance;

boost::shared_ptr<CVCAlgoExtension> instance()
{
  if(!_instance)
    _instance.reset(new CVCAlgoExtension);
  return _instance;
}

void init()
{
  instance()->init();
}
#endif
*/

#ifndef NO_LBIE
cvcraw_geometry::cvcgeom_t convert(const LBIE::geoframe &geo) {
  using namespace std;
  cvcraw_geometry::cvcgeom_t ret_geom;
  ret_geom.points().resize(geo.verts.size());
  copy(geo.verts.begin(), geo.verts.end(), ret_geom.points().begin());
  ret_geom.normals().resize(geo.normals.size());
  copy(geo.normals.begin(), geo.normals.end(), ret_geom.normals().begin());
  ret_geom.colors().resize(geo.color.size());
  copy(geo.color.begin(), geo.color.end(), ret_geom.colors().begin());
  ret_geom.boundary().resize(geo.bound_sign.size());
  for (vector<unsigned int>::const_iterator j = geo.bound_sign.begin();
       j != geo.bound_sign.end(); j++)
    ret_geom.boundary()[distance(j, geo.bound_sign.begin())] = *j;
  ret_geom.triangles().resize(geo.triangles.size());
  copy(geo.triangles.begin(), geo.triangles.end(),
       ret_geom.triangles().begin());
  ret_geom.quads().resize(geo.quads.size());
  copy(geo.quads.begin(), geo.quads.end(), ret_geom.quads().begin());
  return ret_geom;
}
#endif

#ifndef NO_FASTCONTOURING
cvcraw_geometry::cvcgeom_t convert(const FastContouring::TriSurf &geo) {
  using namespace std;
  cvcraw_geometry::cvcgeom_t ret_geom;
  ret_geom.points().resize(geo.verts.size() / 3);
  memcpy(&(ret_geom.points()[0]), &(geo.verts[0]),
         geo.verts.size() * sizeof(double));
  ret_geom.normals().resize(geo.normals.size() / 3);
  memcpy(&(ret_geom.normals()[0]), &(geo.normals[0]),
         geo.normals.size() * sizeof(double));
  ret_geom.colors().resize(geo.colors.size() / 3);
  memcpy(&(ret_geom.colors()[0]), &(geo.colors[0]),
         geo.colors.size() * sizeof(double));
  ret_geom.triangles().resize(geo.tris.size() / 3);
  memcpy(&(ret_geom.triangles()[0]), &(geo.tris[0]),
         geo.tris.size() * sizeof(unsigned int));
  return ret_geom;
}
#endif

Geometry convert(const cvcraw_geometry::cvcgeom_t &src) {
  // TODO: do direct conversion because this is slow
  return Geometry::conv(src);
}

cvcraw_geometry::cvcgeom_t convert(const Geometry &src) {
  // TODO: do direct conversion because this is slow
  Geometry full = Geometry::conv(src);
  return cvcraw_geometry::cvcgeom_t(full.m_SceneGeometry.geometry);
}

VolMagick::Volume sdf(const cvcraw_geometry::cvcgeom_t &geom,
                      const VolMagick::Dimension &dim,
                      const VolMagick::BoundingBox &bbox, SDFMethod method) {
  VolMagick::Volume vol;

  switch (method) {
  case MULTI_SDF:
#ifndef NO_MULTI_SDF
    vol = multi_sdf::signedDistanceFunction(
        boost::shared_ptr<Geometry>(new Geometry(convert(geom))), dim, bbox);
    vol.desc("Signed Distance Function - multi_sdf");
#else
    // throw UnsupportedException("multi_sdf unsupported");
    std::cerr << "multi_sdf unsupported" << std::endl;
    return VolMagick::Volume();
#endif
    break;
  case SDFLIB:
#ifndef NO_SDFLIB
    vol = SDFLibrary::signedDistanceFunction(
        boost::shared_ptr<Geometry>(new Geometry(convert(geom))), dim, bbox);
    vol.desc("Signed Distance Function - SDFLibrary");
#else
    throw UnsupportedException("SDFLibrary unsupported");
#endif
    break;
  }

  return vol;
}
} // namespace cvcalgo
