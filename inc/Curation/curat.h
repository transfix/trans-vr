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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef CURATION_H
#define CURATION_H

#include <Curation/datastruct.h>
#include <Curation/hfn_util.h>
#include <Curation/op.h>
#include <Curation/robust_cc.h>
#include <Curation/util.h>
// #include "sdf.h"

namespace Curation {

void curate_tr(Triangulation &triang, map<int, cell_cluster> &cluster_set,
               const vector<int> &sorted_cluster_index_vector,
               const int output_tunnel_count, const int output_pocket_count);
}

/*void
curate_vol( KdTree& kd_tree,
         const Triangulation& triang,
         const vector<Vertex_handle>& vertices,
         const vector<Facet>& cur_facets,
         TrMesh& trmesh );
 */
#endif
