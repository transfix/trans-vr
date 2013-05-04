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

#include <boost/current_function.hpp>
#include <boost/format.hpp>

#include <VolMagick/Null_IO.h>

namespace VolMagick
{
  // ----------------
  // Null_IO::Null_IO
  // ----------------
  // Purpose:
  //   Initializes the extension list and id.
  // ---- Change History ----
  // 12/04/2009 -- Joe R. -- Initial implementation.
  Null_IO::Null_IO()
    : _id("Null_IO : v1.0")
  {
    _extensions.push_back(".nothing");
  }

  // -----------
  // Null_IO::id
  // -----------
  // Purpose:
  //   Returns a string that identifies this VolumeFile_IO object.  This should
  //   be unique, but is freeform.
  // ---- Change History ----
  // 12/04/2009 -- Joe R. -- Initial implementation.
  const std::string& Null_IO::id() const
  {
    return _id;
  }

  // -------------------
  // Null_IO::extensions
  // -------------------
  // Purpose:
  //   Returns a list of extensions that this VolumeFile_IO object supports.
  // ---- Change History ----
  // 12/04/2009 -- Joe R. -- Initial implementation.
  const VolumeFile_IO::ExtensionList& Null_IO::extensions() const
  {
    return _extensions;
  }

  // --------------------------
  // Null_IO::getVolumeFileInfo
  // --------------------------
  // Purpose:
  //   Throws an error since this class is not intended for actual IO.
  // ---- Change History ----
  // 12/04/2009 -- Joe R. -- Initial implementation.
  void Null_IO::getVolumeFileInfo(VolumeFileInfo::Data& /*data*/,
				  const std::string& /*filename*/) const
  {
    throw ReadError(
      boost::str(
	boost::format("%1% unimplemented") % BOOST_CURRENT_FUNCTION
      )
    );
  }

  // -----------------------
  // Null_IO::readVolumeFile
  // -----------------------
  // Purpose:
  //   Throws an error since this class is not intended for actual IO.
  // ---- Change History ----
  // 12/04/2009 -- Joe R. -- Initial implementation.
  void Null_IO::readVolumeFile(Volume& /*vol*/,
			       const std::string& /*filename*/, 
			       unsigned int /*var*/, unsigned int /*time*/,
			       uint64 /*off_x*/, uint64 /*off_y*/, uint64 /*off_z*/,
			       const Dimension& /*subvoldim*/) const
  {
    throw ReadError(
      boost::str(
	boost::format("%1% unimplemented") % BOOST_CURRENT_FUNCTION
      )
    );
  }

  // -------------------------
  // Null_IO::createVolumeFile
  // -------------------------
  // Purpose:
  //   Throws an error since this class is not intended for actual IO.
  // ---- Change History ----
  // 12/04/2009 -- Joe R. -- Initial implementation.
  void Null_IO::createVolumeFile(const std::string& /*filename*/,
				 const BoundingBox& /*boundingBox*/,
				 const Dimension& /*dimension*/,
				 const std::vector<VoxelType>& /*voxelTypes*/,
				 unsigned int /*numVariables*/, unsigned int /*numTimesteps*/,
				 double /*min_time*/, double /*max_time*/) const
  {
    throw WriteError(
      boost::str(
	boost::format("%1% unimplemented") % BOOST_CURRENT_FUNCTION
      )
    );
  }

  // ------------------------
  // Null_IO::writeVolumeFile
  // ------------------------
  // Purpose:
  //   Throws an error since this class is not intended for actual IO.
  // ---- Change History ----
  // 12/04/2009 -- Joe R. -- Initial implementation.
  void Null_IO::writeVolumeFile(const Volume& /*wvol*/, 
				const std::string& /*filename*/,
				unsigned int /*var*/, unsigned int /*time*/,
				uint64 /*off_x*/, uint64 /*off_y*/, uint64 /*off_z*/) const
  {
    throw WriteError(
      boost::str(
	boost::format("%1% unimplemented") % BOOST_CURRENT_FUNCTION
      )
    );
  }
}
