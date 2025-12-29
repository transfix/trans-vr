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

/* $Id: VoxelOperationStatusMessenger.cpp 4742 2011-10-21 22:09:44Z transfix $
 */

#include <VolMagick/VoxelOperationStatusMessenger.h>

namespace VolMagick {
const VoxelOperationStatusMessenger *vosmDefault = NULL;
void setDefaultMessenger(const VoxelOperationStatusMessenger *vosm) {
  vosmDefault = vosm;
}

const char *VoxelOperationStatusMessenger::opStrings[] = {
    "Calculating Min/Max",
    "Calculating Min",
    "Calculating Max",
    "Subvolume Extraction",
    "Fill",
    "Map",
    "Resize",
    "Composite",
    "Bilateral Filter",
    "Contrast Enhancement",
    "Anisotropic Diffusion",
    "CombineWith",
    "ReadVolumeFile",
    "WriteVolumeFile",
    "CreateVolumeFile",
    "CalcGradient",
    "GDTVFilter",
    "CalculatingHistogram"};
} // namespace VolMagick
