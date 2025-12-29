/*
  Copyright 2008-2011 The University of Texas at Austin

        Authors: John Edwards <john.edwards@utexas.edu>
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

#ifdef USING_TILING

#ifndef __CVCGEOM_CONTOURS_H__
#define __CVCGEOM_CONTOURS_H__

#include <ContourTiler/Slice.h>
#include <ContourTiler/config.h>
#include <cvcraw_geometry/cvcgeom.h>
#include <vector>

#ifndef CVCGEOM_NAMESPACE
#define CVCGEOM_NAMESPACE cvcraw_geometry
#endif

#ifndef DISABLE_CONVERSION
namespace CVCGEOM_NAMESPACE {
class geometry_t;
}
#endif

// namespace CONTOURTILER_NAMESPACE {
//   class Slice;
// }

namespace CVCGEOM_NAMESPACE {
typedef boost::uint64_t uint64_t;

// --------
// cvcgeom_t
// --------
// Purpose:
//   Standard geometry container for cvc algorithms.
// ---- Change History ----
// 07/03/2010 -- Joe R. -- Initial implementation.
class contours_t {
public:
  contours_t();
  contours_t(const std::list<std::string> &components,
             const std::vector<CONTOURTILER_NAMESPACE::Slice> &slices,
             int z_first, int z_last, const std::string &name);
  ~contours_t();

  cvcgeom_t geom() const { return _geom; }
  const std::vector<CONTOURTILER_NAMESPACE::Slice> &slices() const {
    return _slices;
  }
  std::vector<CONTOURTILER_NAMESPACE::Slice> &slices() { return _slices; }
  const std::vector<std::string> &components() const { return _components; }
  std::vector<std::string> &components() { return _components; }
  int z_first() const { return _z_first; }
  int z_last() const { return _z_last; }
  void set_z_scale(double z_scale) {
    _z_scale = z_scale;
    refresh_geom();
  }
  double z_scale() const { return _z_scale; }
  std::string name() const { return _name; }

  void refresh_geom();

private:
  cvcgeom_t _geom;
  std::vector<std::string> _components;
  std::vector<CONTOURTILER_NAMESPACE::Slice> _slices;
  int _z_first, _z_last; // _z_last is the last slice (not one past the last)
  double _z_scale;
  std::string _name;
};
} // namespace CVCGEOM_NAMESPACE

#endif
#endif
