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

/* $Id: CompositeFunction.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_COMPOSITEFUNCTION_H__
#define __VOLMAGICK_COMPOSITEFUNCTION_H__

#include <VolMagick/Voxels.h>

namespace VolMagick
{
  class CompositeFunction
  {
  public:
    CompositeFunction() {}
    virtual ~CompositeFunction() {}

    /*
      in_vox - input voxels
      in_i,j,k - input indices specifying the current input voxel being composited
      this_vox - the destination voxels object where the result of the composition will be stored
      this_i,j,k - the destination voxel indices
      returns - the result of the composition
    */
    virtual double operator()(const Voxels& in_vox, uint64 in_i, uint64 in_j, uint64 in_k,
			      const Voxels& this_vox,uint64 this_i, uint64 this_j, uint64 this_k) const = 0;
  };

  /*
    Replaces the destination voxel with the input voxel
  */
  class CopyFunc : public CompositeFunction
  {
  public:
    CopyFunc() {}
    virtual ~CopyFunc() {}

    virtual double operator()(const Voxels& in_vox, uint64 in_i, uint64 in_j, uint64 in_k,
			      const Voxels& this_vox,uint64 this_i, uint64 this_j, uint64 this_k) const
    {
      return in_vox(in_i,in_j,in_k); /* make compiler happy.... */ this_vox(0); this_i=0; this_j=0; this_k=0;
    }
  };

  /*
    Adds the input voxel to the destination voxel;
  */
  class AddFunc : public CompositeFunction
  {
  public:
    AddFunc() {}
    virtual ~AddFunc() {}

    virtual double operator()(const Voxels& in_vox, uint64 in_i, uint64 in_j, uint64 in_k,
			      const Voxels& this_vox,uint64 this_i, uint64 this_j, uint64 this_k) const
    {
      return this_vox(this_i,this_j,this_k) + in_vox(in_i,in_j,in_k);
    }
  };

  /*
    Subtracts the destination voxel with the input voxel
  */
  class SubtractFunc : public CompositeFunction
  {
  public:
    SubtractFunc() {}
    virtual ~SubtractFunc() {}

    virtual double operator()(const Voxels& in_vox, uint64 in_i, uint64 in_j, uint64 in_k,
			      const Voxels& this_vox,uint64 this_i, uint64 this_j, uint64 this_k) const
    {
      return this_vox(this_i,this_j,this_k) - in_vox(in_i,in_j,in_k);
    }
  };
}

#endif
