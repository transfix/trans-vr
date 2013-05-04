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

/* $Id */

#ifndef __VOLMAGICK_STDERROPSTATUS_H__
#define __VOLMAGICK_STDERROPSTATUS_H__

#include <VolMagick/VoxelOperationStatusMessenger.h>

namespace VolMagick
{
  //Simple implementation of the status messenger interface for reference.  You don't need to use it.
  class StdErrOpStatus : public VoxelOperationStatusMessenger
  {
  public:
    void start(const Voxels *vox, Operation op, uint64 numSteps) const;
    void step(const Voxels *vox, Operation op, uint64 curStep) const;
    void end(const Voxels *vox, Operation op) const;
  private:
    mutable uint64 _numSteps;
  };
}

#endif
