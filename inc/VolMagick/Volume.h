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

/* $Id: Volume.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_VOLUME_H__
#define __VOLMAGICK_VOLUME_H__

#include <VolMagick/Voxels.h>
#include <VolMagick/BoundingBox.h>

#include <string>

namespace VolMagick
{
  class Volume : public Voxels
  {
  public:
    Volume(const Dimension& d = Dimension(4,4,4), 
	   VoxelType vt = CVC::UChar, 
	   const BoundingBox& box = BoundingBox(-0.5,-0.5,-0.5,0.5,0.5,0.5)) 
      : Voxels(d,vt), _boundingBox(box), _desc("No Name") {}
    Volume(const unsigned char *v, 
	   const Dimension& d, 
	   VoxelType vt, 
	   const BoundingBox& box = BoundingBox(-0.5,-0.5,-0.5,0.5,0.5,0.5))
      : Voxels(v,d,vt), _boundingBox(box), _desc("No Name") {}
    Volume(const Voxels& vox,
	   const BoundingBox& box = BoundingBox(-0.5,-0.5,-0.5,0.5,0.5,0.5))
      : Voxels(vox), _boundingBox(box), _desc("No Name") {}
    Volume(const Volume& vol)
      : Voxels(vol), _boundingBox(vol.boundingBox()), _desc(vol.desc()) {}
    ~Volume() {}

    /*
      Bounding box in object space
     */
    BoundingBox& boundingBox() { return _boundingBox; }
    const BoundingBox& boundingBox() const { return _boundingBox; }
    void boundingBox(const BoundingBox& box) { _boundingBox = box; }

    double XMin() const { return boundingBox().minx; }
    double XMax() const { return boundingBox().maxx; }
    double YMin() const { return boundingBox().miny; }
    double YMax() const { return boundingBox().maxy; }
    double ZMin() const { return boundingBox().minz; }
    double ZMax() const { return boundingBox().maxz; }

    double XSpan() const { return XDim()-1 == 0 ? 0.0 : (boundingBox().maxx-boundingBox().minx)/(XDim()-1); }
    double YSpan() const { return YDim()-1 == 0 ? 0.0 : (boundingBox().maxy-boundingBox().miny)/(YDim()-1); }
    double ZSpan() const { return ZDim()-1 == 0 ? 0.0 : (boundingBox().maxz-boundingBox().minz)/(ZDim()-1); }

    /*
      Volume description (used when this object is being saved and the volume
      format supports volume descriptions)
    */
    const std::string& desc() const { return _desc; }
    void desc(const std::string& d) { _desc = d; }

    Volume& operator=(const Volume& vol) { copy(vol); return *this; }

    bool operator==(const Volume& vol)
    {
      return Voxels::operator==(vol) &&
        boundingBox() == vol.boundingBox();
    }

    bool operator!=(const Volume& vol)
    {
      return !(*this == vol);
    }

    /*
      Operations!
    */
    virtual Volume& copy(const Volume& vol); // makes this a copy of vol
    virtual Volume& sub(uint64 off_x, uint64 off_y, uint64 off_z,
			const Dimension& subvoldim
#ifdef _MSC_VER
			, int brain_damage = 1 //avoiding VC++ error C2555
#endif
			);
    /*
      compose volumes using object space coordinates.  Makes a duplicate of compVol and resizes it to match
      the grid resolution of this volume, then does normal voxel composition.
    */
    //virtual Volume& compositeObj(const Volume& compVol, double off_x, double off_y, double off_z, const CompositeFunction& func);

    virtual Volume& sub(const BoundingBox& subvolbox); //Gets a subvolume from a bounding box.
                                                       //Aims to keep the span of the subvolume
                                                       //as close as possible to the original.

    //Creates a subvolume with a bounding box == subvolbox, and a dimension == subvoldim
    virtual Volume& sub(const BoundingBox& subvolbox, const Dimension& subvoldim);

    //returns a linearly interpolated voxel value for the object coordinates supplied.  The coordinates must
    //be inside the bounding box, or an exception is thrown.
    double interpolate(double obj_x, double obj_y, double obj_z) const;

    //makes this volume into a new volume that contains both this volume and the volume specified, bounding box and all
    //If dimension is specified, this volume will be resized to that dimension
    Volume& combineWith(const Volume& vol, const Dimension& dim);
    Volume& combineWith(const Volume& vol);

  protected:
    BoundingBox _boundingBox;
    std::string _desc;
  };
}

#endif
