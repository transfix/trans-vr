/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Skeletonization.

  Skeletonization is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Skeletonization is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: PolyTess.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef __SKELETONIZATION__POLYTESS_H__
#define __SKELETONIZATION__POLYTESS_H__

#include <Skeletonization/Skeletonization.h>

namespace PolyTess {
boost::tuple<std::vector<Skeletonization::Simple_vertex>, /* verts */
             std::vector<unsigned int>>                   /* tri indices */
getTris(const Skeletonization::Polygon_set &polygons);
}

#endif
