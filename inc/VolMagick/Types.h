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

/* $Id: Types.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_TYPES_H__
#define __VOLMAGICK_TYPES_H__

#include <CVC/Types.h>

#ifndef VOLMAGICK_VERSION_STRING
#define VOLMAGICK_VERSION_STRING "1.0.0"
#endif

namespace VolMagick {
typedef CVC::int64 int64;
typedef CVC::uint64 uint64;
typedef CVC::DataType VoxelType;

static const unsigned int *VoxelTypeSizes = CVC::DataTypeSizes;
static const char **VoxelTypeStrings = CVC::DataTypeStrings;

// transfix - 07/24/2011
// For compatibility with old VoxelType enums
const VoxelType Undefined = CVC::Undefined;
const VoxelType UChar = CVC::UChar;
const VoxelType UShort = CVC::UShort;
const VoxelType UInt = CVC::UInt;
const VoxelType Float = CVC::Float;
const VoxelType Double = CVC::Double;
const VoxelType UInt64 = CVC::UInt64;
const VoxelType Char = CVC::Char;
} // namespace VolMagick

#endif
