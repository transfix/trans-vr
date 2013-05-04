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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __U2_H__
#define __U2_H__

#include <SecondaryStructures/datastruct_ss.h>
#include <SecondaryStructures/util.h>
#include <SecondaryStructures/robust_cc.h>
#include <SecondaryStructures/intersect.h>
#include <SecondaryStructures/hfn_util.h>
#include <SecondaryStructures/op.h>
#include <SecondaryStructures/degen.h>

pair< vector< vector<SecondaryStructures::Cell_handle> >, vector<SecondaryStructures::Facet> >
compute_u2(SecondaryStructures::Triangulation& triang, char* prefix);

#endif
