/*
  Copyright 2007-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolMagick.

  VolMagick is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolMagick is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: BoundingBox.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_BOUNDINGBOX_H__
#define __VOLMAGICK_BOUNDINGBOX_H__

#include <CVC/BoundingBox.h>
#include <VolMagick/Dimension.h>
#include <VolMagick/Exceptions.h>
#include <VolMagick/Types.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace VolMagick {
VOLMAGICK_DEF_EXCEPTION(InvalidBoundingBox);

typedef CVC::BoundingBox BoundingBox;           // object space
typedef CVC::IndexBoundingBox IndexBoundingBox; // image space
}; // namespace VolMagick

#endif
