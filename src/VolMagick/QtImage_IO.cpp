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

#include <qimage.h>

#include <VolMagick/QtImage_IO.h>

#include <CVC/App.h>

namespace VolMagick
{
  // ----------------------
  // QtImage_IO::QtImage_IO
  // ----------------------
  // Purpose:
  //   Initializes the extension list and id.
  // ---- Change History ----
  // 04/24/2010 -- Joe R. -- Initial implementation.
  QtImage_IO::QtImage_IO()
    : _id("QtImage_IO : v1.0")
  {
    _extensions.push_back(".tiff");
    _extensions.push_back(".tif");
    _extensions.push_back(".png");
    _extensions.push_back(".jpg");
  }

  // --------------
  // QtImage_IO::id
  // --------------
  // Purpose:
  //   Returns a string that identifies this VolumeFile_IO object.  This should
  //   be unique, but is freeform.
  // ---- Change History ----
  // 04/24/2010 -- Joe R. -- Initial implementation.
  const std::string& QtImage_IO::id() const
  {
    return _id;
  }

  // ----------------------
  // QtImage_IO::extensions
  // ----------------------
  // Purpose:
  //   Returns a list of extensions that this VolumeFile_IO object supports.
  // ---- Change History ----
  // 04/24/2010 -- Joe R. -- Initial implementation.
  const VolumeFile_IO::ExtensionList& QtImage_IO::extensions() const
  {
    return _extensions;
  }

  // -----------------------------
  // QtImage_IO::getVolumeFileInfo
  // -----------------------------
  // Purpose:
  //   Nothing yet.
  // ---- Change History ----
  // 04/24/2010 -- Joe R. -- Initial implementation.
  void QtImage_IO::getVolumeFileInfo(VolumeFileInfo::Data& data,
				  const std::string& filename) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    QImage img(filename.c_str());
    if(img.isNull())
      throw ReadError(
	 boost::str(
	    boost::format("QImage could not read %1%") % filename
	 )
      );

    data._filename = filename;
    data._dimension = Dimension(img.width(),img.height(),1);
    data._boundingBox = BoundingBox(0.0,0.0,0.0,
				    img.width()+1,
				    img.height()+1,
				    0.0);
    data._numVariables = 1; //just using grayscale for now, so 1 var
    data._numTimesteps = 1;
    data._voxelTypes = std::vector<VoxelType>(1,UChar);
    
    data._names.clear();
    data._names.push_back("QImage");
    data._tmin = 0.0;
    data._tmax = 0.0;

    data._minIsSet.clear();
    data._minIsSet.resize(data._numVariables); for(int i=0; i<data._minIsSet.size(); i++) data._minIsSet[i].resize(data._numTimesteps);
    data._min.clear();
    data._min.resize(data._numVariables); for(int i=0; i<data._min.size(); i++) data._min[i].resize(data._numTimesteps);
    data._maxIsSet.clear();
    data._maxIsSet.resize(data._numVariables); for(int i=0; i<data._maxIsSet.size(); i++) data._maxIsSet[i].resize(data._numTimesteps);
    data._max.clear();
    data._max.resize(data._numVariables); for(int i=0; i<data._max.size(); i++) data._max[i].resize(data._numTimesteps);
  }

  // --------------------------
  // QtImage_IO::readVolumeFile
  // --------------------------
  // Purpose:
  //   Nothing yet.
  // ---- Change History ----
  // 04/24/2010 -- Joe R. -- Initial implementation.
  void QtImage_IO::readVolumeFile(Volume& /*vol*/,
			       const std::string& /*filename*/, 
			       unsigned int /*var*/, unsigned int /*time*/,
			       uint64 /*off_x*/, uint64 /*off_y*/, uint64 /*off_z*/,
			       const Dimension& /*subvoldim*/) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    throw ReadError(
      boost::str(
	boost::format("%1% unimplemented") % BOOST_CURRENT_FUNCTION
      )
    );
  }

  // ----------------------------
  // QtImage_IO::createVolumeFile
  // ----------------------------
  // Purpose:
  //   Nothing yet.
  // ---- Change History ----
  // 04/24/2010 -- Joe R. -- Initial implementation.
  void QtImage_IO::createVolumeFile(const std::string& /*filename*/,
				 const BoundingBox& /*boundingBox*/,
				 const Dimension& /*dimension*/,
				 const std::vector<VoxelType>& /*voxelTypes*/,
				 unsigned int /*numVariables*/, unsigned int /*numTimesteps*/,
				 double /*min_time*/, double /*max_time*/) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    throw WriteError(
      boost::str(
	boost::format("%1% unimplemented") % BOOST_CURRENT_FUNCTION
      )
    );
  }

  // ---------------------------
  // QtImage_IO::writeVolumeFile
  // ---------------------------
  // Purpose:
  //   Nothing yet.
  // ---- Change History ----
  // 04/24/2010 -- Joe R. -- Initial implementation.
  void QtImage_IO::writeVolumeFile(const Volume& /*wvol*/, 
				const std::string& /*filename*/,
				unsigned int /*var*/, unsigned int /*time*/,
				uint64 /*off_x*/, uint64 /*off_y*/, uint64 /*off_z*/) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    throw WriteError(
      boost::str(
	boost::format("%1% unimplemented") % BOOST_CURRENT_FUNCTION
      )
    );
  }
}
