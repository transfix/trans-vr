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

#ifndef __VTK_H__
#define __VTK_H__

#include <VolMagick/VolumeFile_IO.h>
#include <VolMagick/Exceptions.h>

namespace VolMagick
{
  // ------
  // VTK_IO
  // ------
  // Purpose: 
  //   Provides VTK file support.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  struct VTK_IO : public VolumeFile_IO
  {
    // --------------
    // VTK_IO::VTK_IO
    // --------------
    // Purpose:
    //   Initializes the extension list and id.
    // ---- Change History ----
    // 11/20/2009 -- Joe R. -- Initial implementation.
    VTK_IO();

    // ----------
    // VTK_IO::id
    // ----------
    // Purpose:
    //   Returns a string that identifies this VolumeFile_IO object.  This should
    //   be unique, but is freeform.
    // ---- Change History ----
    // 11/20/2009 -- Joe R. -- Initial implementation.
    virtual const std::string& id() const;

    // ------------------
    // VTK_IO::extensions
    // ------------------
    // Purpose:
    //   Returns a list of extensions that this VolumeFile_IO object supports.
    // ---- Change History ----
    // 11/20/2009 -- Joe R. -- Initial implementation.
    virtual const ExtensionList& extensions() const;

    // -------------------------
    // VTK_IO::getVolumeFileInfo
    // -------------------------
    // Purpose:
    //   Writes to a structure containing all info that VolMagick needs
    //   from a volume file.
    // ---- Change History ----
    // 11/20/2009 -- Joe R. -- Initial implementation.
    virtual void getVolumeFileInfo(VolumeFileInfo::Data& data,
				   const std::string& filename) const;

    // ----------------------
    // VTK_IO::readVolumeFile
    // ----------------------
    // Purpose:
    //   Writes to a Volume object after reading from a volume file.
    // ---- Change History ----
    // 11/20/2009 -- Joe R. -- Initial implementation.
    virtual void readVolumeFile(Volume& vol,
				const std::string& filename, 
				unsigned int var, unsigned int time,
				uint64 off_x, uint64 off_y, uint64 off_z,
				const Dimension& subvoldim) const;

    // ------------------------
    // VTK_IO::createVolumeFile
    // ------------------------
    // Purpose:
    //   Creates an empty volume file to be later filled in by writeVolumeFile
    // ---- Change History ----
    // 11/20/2009 -- Joe R. -- Initial implementation.
    virtual void createVolumeFile(const std::string& filename,
				  const BoundingBox& boundingBox,
				  const Dimension& dimension,
				  const std::vector<VoxelType>& voxelTypes,
				  unsigned int numVariables, unsigned int numTimesteps,
				  double min_time, double max_time) const;

    // -----------------------
    // VTK_IO::writeVolumeFile
    // -----------------------
    // Purpose:
    //   Writes the volume contained in wvol to the specified volume file. Should create
    //   a volume file if the filename provided doesn't exist.  Else it will simply
    //   write data to the existing file.  A common user error arises when you try to
    //   write over an existing volume file using this function for unrelated volumes.
    //   If what you desire is to overwrite an existing volume file, first run
    //   createVolumeFile to replace the volume file.
    // ---- Change History ----
    // 11/20/2009 -- Joe R. -- Initial implementation.
    virtual void writeVolumeFile(const Volume& wvol, 
				 const std::string& filename,
				 unsigned int var, unsigned int time,
				 uint64 off_x, uint64 off_y, uint64 off_z) const;
  protected:
    std::string _id;
    std::list<std::string> _extensions;
  };
}

#endif
