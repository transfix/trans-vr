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

#ifndef __SKELETONIZATION__DEGEN_H__
#define __SKELETONIZATION__DEGEN_H__

#include <Skeletonization/datastruct.h>
#include <Skeletonization/op.h>

namespace Skeletonization
{
bool is_degenerate_VF(const Triangulation& triang, 
                 const Cell_handle& c,
                 const int& fid,
                 const int& uid,
                 const int& vid,
                 const Point& d,
                 const char* prefix);
}
#endif // DEGEN_H

