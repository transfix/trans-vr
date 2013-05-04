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

/* $Id: VolumeFile_IO.h 5355 2012-04-06 22:16:56Z transfix $ */

#ifndef __VOLMAGICK_VOLUMEFILE_IO_H__
#define __VOLMAGICK_VOLUMEFILE_IO_H__

#include <VolMagick/VolumeFileInfo.h>
#include <VolMagick/Volume.h>

#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/regex.hpp>

#include <string>
#include <list>
#include <map>
#include <vector>

namespace VolMagick
{
  // -------------
  // VolumeFile_IO
  // -------------
  // Purpose: 
  //   Provides functions for accessing and creating volume files. Meant
  //   to be inherited by clients of VolMagick whishing to add support
  //   for any particular volume file format.  The default implementation
  //   does nothing.
  // ---- Change History ----
  // 11/13/2009 -- Joe R. -- Initial implementation.
  // 09/05/2011 -- Joe R. -- Moved splitRawFilename and 2 const strings here
  //                         from HDF5_IO.
  // 09/10/2011 -- Joe R. -- Just using char* and constructing regexes locally to avoid
  //                         crashes if the main thread finishes before child threads.
  struct VolumeFile_IO
  {
    static const char* CVC_VOLUME_GROUP; // /cvc/volumes
    static const char* DEFAULT_VOLUME_NAME;
    static const char* FILE_EXTENSION_EXPR;

    // -------------------------------
    // VolumeFile_IO::splitRawFilename
    // -------------------------------
    // Purpose:
    //   Splits a filename into an actual file name and an object
    //   path.
    // ---- Change History ----
    // 07/24/2011 -- Joe R. -- Initial implementation.
    // 09/05/2011 -- Joe R. -- Moved here from HDF5_IO
    static boost::tuple<
      std::string, /* actual file name */
      std::string  /* hdf5 object name */
    > splitRawFilename(const std::string& filename);

    // -----------------
    // VolumeFile_IO::id
    // -----------------
    // Purpose:
    //   Returns a string that identifies this VolumeFile_IO object.  This should
    //   be unique, but is freeform.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    virtual const std::string& id() const = 0;

    typedef std::list<std::string> ExtensionList;

    // -------------------------
    // VolumeFile_IO::extensions
    // -------------------------
    // Purpose:
    //   Returns a list of extensions that this VolumeFile_IO object supports.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    virtual const ExtensionList& extensions() const = 0;

    // --------------------------------
    // VolumeFile_IO::getVolumeFileInfo
    // --------------------------------
    // Purpose:
    //   Writes to a structure containing all info that VolMagick needs
    //   from a volume file.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    virtual void getVolumeFileInfo(VolumeFileInfo::Data& /*data*/,
				   const std::string& /*filename*/) const = 0;

    // -----------------------------
    // VolumeFile_IO::readVolumeFile
    // -----------------------------
    // Purpose:
    //   Writes to a Volume object after reading from a volume file.  
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    virtual void readVolumeFile(Volume& /*vol*/, 
				const std::string& /*filename*/, 
				unsigned int /*var*/,
				unsigned int /*time*/,
				uint64 /*off_x*/,
				uint64 /*off_y*/,
				uint64 /*off_z*/,
				const Dimension& /*subvoldim*/) const = 0;

    // -----------------------------
    // VolumeFile_IO::readVolumeFile
    // -----------------------------
    // Purpose:
    //   Same as above except uses a bounding box for specifying the
    //   subvol.  A default implementation is provided. 
    // ---- Change History ----
    // 01/03/2010 -- Joe R. -- Initial implementation.
    virtual void readVolumeFile(Volume& vol, 
				const std::string& filename, 
				unsigned int var,
				unsigned int time,
				const BoundingBox& subvolbox) const;

    // -------------------------------
    // VolumeFile_IO::createVolumeFile
    // -------------------------------
    // Purpose:
    //   Creates an empty volume file to be later filled in by writeVolumeFile
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    virtual void createVolumeFile(const std::string& /*filename*/,
				  const BoundingBox& /*boundingBox*/,
				  const Dimension& /*dimension*/,
				  const std::vector<VoxelType>& /*voxelTypes*/,
				  unsigned int /*numVariables*/,
				  unsigned int /*numTimesteps*/,
				  double /*min_time*/,
				  double /*max_time*/) const = 0;

    // ------------------------------
    // VolumeFile_IO::writeVolumeFile
    // ------------------------------
    // Purpose:
    //   Writes the volume contained in wvol to the specified volume file. Should create
    //   a volume file if the filename provided doesn't exist.  Else it will simply
    //   write data to the existing file.  A common user error arises when you try to
    //   write over an existing volume file using this function for unrelated volumes.
    //   If what you desire is to overwrite an existing volume file, first run
    //   createVolumeFile to replace the volume file.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    virtual void writeVolumeFile(const Volume& /*wvol*/, 
				 const std::string& /*filename*/,
				 unsigned int /*var*/,
				 unsigned int /*time*/,
				 uint64 /*off_x*/,
				 uint64 /*off_y*/,
				 uint64 /*off_z*/) const = 0;

    // -------------------------------
    // VolumeFile_IO::writeBoundingBox
    // -------------------------------
    // Purpose:
    //   Writes the specified bounding box to the file.  The default implementation is slow
    //   because it has to read the entire file.  This can be sped up on an individual file type
    //   basis.
    // ---- Change History ----
    // 04/06/2012 -- Joe R. -- Initial implementation.
    virtual void writeBoundingBox(const BoundingBox& bbox, const std::string& filename) const;

    typedef boost::shared_ptr<VolumeFile_IO> Ptr;
    typedef std::vector<Ptr> Handlers;
    typedef std::map<
      std::string, /* file extension */
      Handlers > HandlerMap;

    // -------------------------
    // VolumeFile_IO::handlerMap
    // -------------------------
    // Purpose:
    //   Static initialization of handler map.  Clients use the HandlerMap
    //   to add themselves to the collection of objects that are to be used
    //   to perform volume file i/o operations.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    static HandlerMap& handlerMap();

    // ----------------------------
    // VolumeFile_IO::insertHandler
    // ----------------------------
    // Purpose:
    //   Convenience function for adding objects to the map.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    static void insertHandler(const Ptr& vfio);

    // ----------------------------
    // VolumeFile_IO::removeHandler
    // ----------------------------
    // Purpose:
    //   Convenience function for removing objects from the map.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    static void removeHandler(const Ptr& vfio);

    // ----------------------------
    // VolumeFile_IO::removeHandler
    // ----------------------------
    // Purpose:
    //   Convenience function for removing objects from the map.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    static void removeHandler(const std::string& name);

    // ----------------------------
    // VolumeFile_IO::getExtensions
    // ----------------------------
    // Purpose:
    //   Returns the list of supported file extensions.
    // ---- Change History ----
    // 09/18/2011 -- Joe R. -- Initial implementation.    
    static std::vector<std::string> getExtensions();

  private:
    // ----------------------------
    // VolumeFile_IO::initializeMap
    // ----------------------------
    // Purpose:
    //   Adds the standard VolumeFile_IO objects to a new HandlerMap object
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    static HandlerMap *initializeMap();

    // ----------------------------
    // VolumeFile_IO::insertHandler
    // ----------------------------
    // Purpose:
    //   Convenience function for adding objects to the specified map.
    // ---- Change History ----
    // 11/13/2009 -- Joe R. -- Initial implementation.
    static void insertHandler(HandlerMap& hm, const Ptr& vfio);
  };

  // ------------------------- Volume I/O API

    /*
    read the specified subvolume from the specified file and copy it to the object
    vol.
  */
  void readVolumeFile(Volume& vol,
		      const std::string& filename, 
		      unsigned int var, unsigned int time,
		      uint64 off_x, uint64 off_y, uint64 off_z,
		      const Dimension& subvoldim);

  /*
    read the specified subvolume from the specified file and copy it to the object
    vol.
  */
  void readVolumeFile(Volume& vol,
		      const std::string& filename, 
		      unsigned int var, unsigned int time,
		      const BoundingBox& subvolbox);

  /*
    read the entire volume from the specified file and copy it to the object vol.
  */
  void readVolumeFile(Volume& vol, 
		      const std::string& filename,
		      unsigned int var = 0, unsigned int time = 0);

  /*
    Read multi-volume file and add each volume to the vector vols.
  */
  void readVolumeFile(std::vector<Volume>& vols,
		      const std::string& filename);

  /*
    write the specified volume to the specified offset in file 'filename'
  */
  void writeVolumeFile(const Volume& vol, 
		       const std::string& filename,
		       unsigned int var = 0, unsigned int time = 0,
		       uint64 off_x = 0, uint64 off_y = 0, uint64 off_z = 0);

  /*
    write the specified volume to the specified subvolume bounding box in file 'filename'
  */
  void writeVolumeFile(const Volume& vol, 
		       const std::string& filename,
		       unsigned int var, unsigned int time,
		       const BoundingBox& subvolbox);

  /*
    Writes the vector 'vols' to the specified file.  Make sure that the file extension
    specified is for a volume file type that supports multi-volumes. Assumes 1 timestep.
  */
  void writeVolumeFile(const std::vector<Volume>& vols,
		       const std::string& filename);

  /*
    Creates a volume file using the specified information.
  */
  void createVolumeFile(const std::string& filename,
			const BoundingBox& boundingBox,
			const Dimension& dimension,
			const std::vector<VoxelType>& voxelTypes = std::vector<VoxelType>(1, CVC::UChar),
			unsigned int numVariables = 1, unsigned int numTimesteps = 1,
			double min_time = 0.0, double max_time = 0.0);

  //Functions for reading and writing volume bounding boxes
  BoundingBox readBoundingBox(const std::string& filename);
  void writeBoundingBox(const BoundingBox& bbox, const std::string& filename);
}

#endif
