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

#ifndef SMAX_H
#define SMAX_H

#include <PocketTunnel/datastruct.h>
#include <PocketTunnel/hfn_util.h>
#include <PocketTunnel/robust_cc.h>
#include <PocketTunnel/op.h>

namespace PocketTunnel
{

  std::vector<int>
    compute_smax(Triangulation& triang, 
		 std::map<int, cell_cluster> &cluster_set,
		 const double& mr);

};

#endif
