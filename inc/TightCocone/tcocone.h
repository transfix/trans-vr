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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef TCOCONE_H
#define TCOCONE_H

#include <TightCocone/datastruct.h>
#include <TightCocone/util.h>
#include <TightCocone/robust_cc.h>

namespace TightCocone
{

void 
compute_poles( Triangulation &triang);

void 
mark_flat_vertices( Triangulation &triang,
		    double ratio, double cocone_phi, double flat_phi);

void
tcocone(const double DEFAULT_ANGLE,
        const double DEFAULT_SHARP,
	const double DEFAULT_FLAT,
	const double DEFAULT_RATIO,
	Triangulation &triang);

}

#endif // TCOCONE_H

