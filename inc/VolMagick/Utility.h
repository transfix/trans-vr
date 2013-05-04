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

/* $Id: Utility.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_UTILITY_H__
#define __VOLMAGICK_UTILITY_H__

#include <VolMagick/VolumeFileInfo.h>
#include <VolMagick/VolumeFile_IO.h>
#include <VolMagick/VoxelOperationStatusMessenger.h>

namespace VolMagick
{
  // arand: the mac doesn't like VERSION... so I changed it to VM_VERSION
  //        also, I moved this to the header... I have no idea why this was working
  //        in the .cpp file but the mac was complaining
  const uint64 VM_VERSION = 0x00010206;

  template <class T> const T& MIN(const T& a, const T& b) { return std::min(a,b); }
  template <class T> const T& MAX(const T& a, const T& b) { return std::max(a,b); }

  static inline uint64 upToPowerOfTwo(uint64 value)
  {
    uint64 c = 0;
    uint64 v = value;
      
    /* round down to nearest power of two */
    while (v>1)
      {
        v = v>>1;
        c++;
      }
      
    /* if that isn't exactly the original value */
    if ((v<<c)!=value)
      {
        /* return the next power of two */
        return (v<<(c+1));
      }
    else
      {
        /* return this power of two */
        return (v<<c);
      }
  }

  //trilinear interpolation function
  static inline double getTriVal(double val[8], double x, double y, double z,
                                 double resX, double resY, double resZ)
  {
    double x_ratio, y_ratio, z_ratio;
    double temp1,temp2,temp3,temp4,temp5,temp6;
    
    x_ratio=x/resX;
    y_ratio=y/resY;
    z_ratio=z/resZ;
    
    if( x_ratio == 1 ) x_ratio = 0;
    if( y_ratio == 1 ) y_ratio = 0;
    if( z_ratio == 1 ) z_ratio = 0;
    
    temp1 = val[0] + (val[1]-val[0])*x_ratio;
    temp2 = val[4] + (val[5]-val[4])*x_ratio;
    temp3 = val[2] + (val[3]-val[2])*x_ratio;
    temp4 = val[6] + (val[7]-val[6])*x_ratio;
    temp5 = temp1  + (temp3-temp1)*y_ratio;
    temp6 = temp2  + (temp4-temp2)*y_ratio;
    
    return temp5  + (temp6-temp5)*z_ratio;
  }

  /*
    Shortcut for creating a volume file based on a volume file info object
  */  
  static inline void createVolumeFile(const std::string& filename,
				      const VolumeFileInfo& volinfo)
  {
    createVolumeFile(filename,
		     volinfo.boundingBox(),
		     volinfo.dimension(),
		     volinfo.voxelTypes(),
		     volinfo.numVariables(),
		     volinfo.numTimesteps(),
		     volinfo.TMin(),volinfo.TMax());
  }

  // ----------------
  // createVolumeFile
  // ----------------
  // Purpose: 
  //   Shortcut for creating a volume file based on a volume object.  Also
  //   writes the volume data in the object to the specified file.
  // ---- Change History ----
  // 01/04/2010 -- Joe R. -- Initial implementation.
  static inline void createVolumeFile(const Volume& vol,
				      const std::string& filename)
  {
    createVolumeFile(filename,
		     vol.boundingBox(),
		     vol.dimension(),
		     std::vector<VoxelType>(1, vol.voxelType()));
    writeVolumeFile(vol,filename);
  }

  // ----------------
  // createVolumeFile
  // ----------------
  // Purpose: 
  //   Same as above, except with arguments in the order consistent with the full
  //   createVolumeFile call
  // ---- Change History ----
  // 01/04/2010 -- Joe R. -- Initial implementation.
  static inline void createVolumeFile(const std::string& filename,
				      const Volume& vol)
  {
    createVolumeFile(vol,filename);
  }

  /*
    Calculates the gradient vector field of the input voxels, and returns the xyz vector values as 3 volumes in 'grad'
    'vt' is the voxel type of the gradient volumes.  If 'vt' is of integral type (UChar, UShort, UInt), the first
    half of the set of integers maps to [-1.0,0) and the last half maps to (0,1.0].
  */
  void calcGradient(std::vector<Volume>& grad, const Volume& vol, VoxelType vt = CVC::Float);

  /*
    Copies a subvolume of vol to dest.
  */
  void sub(Volume& dest, const Volume& vol, 
	   uint64 off_x, uint64 off_y, uint64 off_z,
	   const Dimension& subvoldim);

  // ----------
  // volconvert
  // ----------
  // Purpose: 
  //   Converts (or copies) volume from one file or filetype to another.  Basically
  //   the same as the VolUtils cmd line program.
  // ---- Change History ----
  // 09/18/2011 -- Joe R. -- Initial implementation.
  void volconvert(const std::string& input_volume_file,
                  const std::string& output_volume_file);
}

#endif
