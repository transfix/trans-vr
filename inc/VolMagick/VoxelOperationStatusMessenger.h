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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: VoxelOperationStatusMessenger.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_VOSM_H__
#define __VOLMAGICK_VOSM_H__

#include <VolMagick/Types.h>

#include <cstdlib>

namespace VolMagick
{
  class Voxels;

  //this class is used to provide calling code a means of getting periodic control during long operations
  class VoxelOperationStatusMessenger
  {
  public:
    enum Operation { 
      CalculatingMinMax, 
      CalculatingMin, CalculatingMax, SubvolumeExtraction,
      Fill, Map, Resize, Composite, BilateralFilter, 
      ContrastEnhancement, AnisotropicDiffusion, CombineWith,
      ReadVolumeFile, WriteVolumeFile, CreateVolumeFile, CalcGradient,
      GDTVFilter, CalculatingHistogram,

      //no more after this comment.  When adding ops, don't forget to add the appropriate
      //string in opStrings
      NUM_OPERATIONS
    };
    static const char *opStrings[NUM_OPERATIONS];
    
    VoxelOperationStatusMessenger() {}
    virtual ~VoxelOperationStatusMessenger() {}

    virtual void start(const Voxels *vox, Operation op, uint64 numSteps) const = 0;
    virtual void step(const Voxels *vox, Operation op, uint64 curStep) const = 0;
    virtual void end(const Voxels *vox, Operation op) const = 0;
  };

  extern const VoxelOperationStatusMessenger* vosmDefault;

  //sets the default messenger for all voxel objects to use
  void setDefaultMessenger(const VoxelOperationStatusMessenger* vosm);
}

#endif
