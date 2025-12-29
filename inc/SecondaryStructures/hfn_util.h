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

#ifndef __HFN_UTIL_H__
#define __HFN_UTIL_H__

#include <SecondaryStructures/datastruct_ss.h>
#include <SecondaryStructures/intersect.h>
#include <SecondaryStructures/util.h>

bool is_maxima(const SecondaryStructures::Cell_handle &c);
bool is_outflow(const SecondaryStructures::Facet &f);
bool is_transversal_flow(const SecondaryStructures::Facet &f);
bool find_acceptor(const SecondaryStructures::Cell_handle &c, const int &id,
                   int &uid, int &vid, int &wid);
bool is_acceptor_for_any_VE(const SecondaryStructures::Triangulation &triang,
                            const SecondaryStructures::Edge &e);
bool is_i2_saddle(const SecondaryStructures::Facet &f);
bool is_i1_saddle(const SecondaryStructures::Edge &e,
                  const SecondaryStructures::Triangulation &triang);
void grow_maxima(SecondaryStructures::Triangulation &triang,
                 SecondaryStructures::Cell_handle c_max);
void find_flow_direction(SecondaryStructures::Triangulation &triang);

#endif
