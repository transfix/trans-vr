/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of SecondaryStructures.

  SecondaryStructures is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  SecondaryStructures is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef __INTERSECT_H__
#define __INTERSECT_H__

#include <SecondaryStructures/datastruct_ss.h>
#include <SecondaryStructures/util.h>

bool does_intersect_ray3_seg3_in_plane(const SecondaryStructures::Ray_3 &r,
                                       const SecondaryStructures::Segment &s);
SecondaryStructures::Point
intersect_ray3_seg3(const SecondaryStructures::Ray_3 &r,
                    const SecondaryStructures::Segment &s,
                    bool &is_correct_intersection);
bool does_intersect_convex_polygon_segment_3_in_3d(
    const vector<SecondaryStructures::Point> &conv_poly,
    const SecondaryStructures::Segment &s);

#endif
