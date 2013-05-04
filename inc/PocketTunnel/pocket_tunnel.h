/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PocketTunnel.

  PocketTunnel is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PocketTunnel is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef POCKETTUNNEL_H
#define POCKETTUNNEL_H

#include <cvcraw_geometry/cvcgeom.h>

#include <PocketTunnel/datastruct.h>
#include <PocketTunnel/tcocone.h>
#include <PocketTunnel/rcocone.h>
#include <PocketTunnel/util.h>
#include <PocketTunnel/op.h>
#include <PocketTunnel/init.h>
#include <PocketTunnel/smax.h>
#include <PocketTunnel/handle.h>

namespace PocketTunnel
{
  CVCGEOM_NAMESPACE::cvcgeom_t* pocket_tunnel_fromsurf(const CVCGEOM_NAMESPACE::cvcgeom_t* molsurf,int, int ); // input surface data.
};

#endif // POCKETTUNNEL_H

