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

/* $Id: Curation.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef __CURATION_H__
#define __CURATION_H__

/*
  Main header!! Include only this!
*/

#include <vector>
#include <boost/shared_ptr.hpp>
#include <cvcraw_geometry/cvcgeom.h>

namespace Curation
{
  // --- segmentation ---

  //const double DEFAULT_MERGE_RATIO = .3*.3;   // ratio to merge two segments.
  const double DEFAULT_MERGE_RATIO = 9999;

  // --- segmentation output ---
  const int DEFAULT_KEEP_POCKETS_COUNT = 3;
  const int DEFAULT_KEEP_TUNNELS_COUNT = 3;

//  std::vector<CVCGEOM_NAMESPACE::cvcgeom_t > curate(const CVCGEOM_NAMESPACE::cvcgeom_t &,
  CVCGEOM_NAMESPACE::cvcgeom_t  curate(const CVCGEOM_NAMESPACE::cvcgeom_t &,
						   float mr = DEFAULT_MERGE_RATIO,
						   int keep_pockets_count = DEFAULT_KEEP_POCKETS_COUNT,
						   int keep_tunnels_count = DEFAULT_KEEP_TUNNELS_COUNT);
}

#endif
