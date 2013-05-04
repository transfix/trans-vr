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

#ifndef INTERSECT_H
#define INTERSECT_H

#include <Curation/datastruct.h>
#include <Curation/util.h>

namespace Curation
{

bool
does_intersect_ray3_seg3_in_plane(const Ray& r, 
                                  const Segment& s);

Point
intersect_ray3_seg3(const Ray& r, 
                    const Segment& s, 
                    bool& is_correct_intersection);

bool
does_intersect_convex_polygon_segment_3_in_3d(const vector<Point>& conv_poly, 
                                              const Segment& s);

}

#endif // INTERSECT_H

