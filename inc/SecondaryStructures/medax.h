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

#ifndef __MEDAX_H__
#define __MEDAX_H__

#include <SecondaryStructures/datastruct_ss.h>
#include <SecondaryStructures/op.h>
#include <SecondaryStructures/robust_cc.h>
#include <SecondaryStructures/util.h>

void compute_medial_axis(SecondaryStructures::Triangulation &triang,
                         const double theta, const double medial_ratio,
                         int &biggest_medax_comp_id);

#endif
