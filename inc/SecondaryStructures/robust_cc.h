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

#ifndef ROBUST_CC_H
#define ROBUST_CC_H

#include <SecondaryStructures/datastruct_ss.h>

using namespace SecondaryStructures;

SecondaryStructures::Point nondg_voronoi_point(
    const SecondaryStructures::Point &a, const SecondaryStructures::Point &b,
    const SecondaryStructures::Point &c, const SecondaryStructures::Point &d,
    bool &is_correct_computation);

SecondaryStructures::Point dg_voronoi_point(
    const SecondaryStructures::Point &a, const SecondaryStructures::Point &b,
    const SecondaryStructures::Point &c, const SecondaryStructures::Point &d,
    bool &is_correct_computation);

SecondaryStructures::Point nondg_cc_tr_3(const SecondaryStructures::Point &a,
                                         const SecondaryStructures::Point &b,
                                         const SecondaryStructures::Point &c,
                                         bool &is_correct_computation);

SecondaryStructures::Point cc_tr_3(const SecondaryStructures::Point &a,
                                   const SecondaryStructures::Point &b,
                                   const SecondaryStructures::Point &c);

double sq_cr_tr_3(const SecondaryStructures::Point &a,
                  const SecondaryStructures::Point &b,
                  const SecondaryStructures::Point &c);

SecondaryStructures::Point circumcenter(const SecondaryStructures::Facet &f);

double circumradius(const SecondaryStructures::Facet &f);

#endif
