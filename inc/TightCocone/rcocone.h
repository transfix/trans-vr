/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef RCOCONE_H
#define RCOCONE_H

#include <TightCocone/datastruct.h>
#include <TightCocone/robust_cc.h>
#include <TightCocone/util.h>

namespace TightCocone {

void robust_cocone(const double bb_ratio, const double theta_ff,
                   const double theta_if, Triangulation &triang,
                   const char *outfile_prefix);

std::vector<Point> robust_cocone(const double bb_ratio, const double theta_ff,
                                 const double theta_if,
                                 Triangulation &triang);
} // namespace TightCocone

#endif // RCOCONE_H
