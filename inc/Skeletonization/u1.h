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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __SKELETONIZATION__U1_H__
#define __SKELETONIZATION__U1_H__

#include <Skeletonization/datastruct.h>
#include <Skeletonization/util.h>
#include <Skeletonization/robust_cc.h>
#include <Skeletonization/intersect.h>
#include <Skeletonization/hfn_util.h>
#include <Skeletonization/op.h>
#include <Skeletonization/graph.h>
#include <Skeletonization/degen.h>

namespace Skeletonization
{
  void compute_u1(Triangulation& triang);
}

#endif // U1_H
