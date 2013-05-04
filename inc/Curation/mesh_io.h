/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Curation.

  Curation is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Curation is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef MESH_IO_H
#define MESH_IO_H

#include <Curation/mds.h>

#include <cvcraw_geometry/cvcgeom.h>
#include <boost/shared_ptr.hpp>

namespace Curation
{

void
read_mesh(Mesh &mesh, const char* ip_filename, bool read_color_opacity);

void read_mesh_from_geom(Mesh &mesh, const CVCGEOM_NAMESPACE::cvcgeom_t &);

}

#endif
