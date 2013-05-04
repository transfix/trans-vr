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

/* $Id: IMOD_MRC_IO.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <cstdio>
#include <climits>

#include <VolMagick/VolMagick.h>
#include <VolMagick/IMOD_MRC_IO.h>
#include <VolMagick/endians.h>

#include <VolMagick/libiimod/iimage.h>

#include <boost/scoped_array.hpp>
#include <boost/current_function.hpp>

#include <CVC/App.h>

namespace VolMagick
{
  // ------------------------
  // IMOD_MRC_IO::IMOD_MRC_IO
  // ------------------------
  // Purpose:
  //   Initializes the extension list and id.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  IMOD_MRC_IO::IMOD_MRC_IO()
  {
    _id = "IMOD_MRC_IO : v1.0";
  }

  // ------------------------------
  // IMOD_MRC_IO::getVolumeFileInfo
  // ------------------------------
  // Purpose:
  //   Writes to a structure containing all info that VolMagick needs
  //   from a volume file.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  void IMOD_MRC_IO::getVolumeFileInfo(VolumeFileInfo::Data& data,
				      const std::string& filename) const
  {
    using namespace std;
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    VoxelType mrcTypes[] = { UChar, UShort, Float };
    MrcHeader header;
    ImodImageFile *iif = iiOpen(const_cast<char*>(filename.c_str()),"rb");
    if(iif == NULL)
      throw ReadError("Error opening MRC file via libiimod");

    data._dimension = Dimension(iif->nx,iif->ny,iif->nz);
    data._numTimesteps = 1;
    data._numVariables = 1;
    data._names.clear();
    data._names.push_back("No Name");
    data._voxelTypes.clear();
    if(iif->mode > 3)
      throw InvalidMRCFile("Invalid mode");
    data._voxelTypes.push_back(mrcTypes[iif->mode]);

    //read the header directly because it doesn't seem that bounding box info
    //is kept in the ImodImageFile struct.
    if(mrc_head_read(iif->fp,&header))
      {
	iiClose(iif);
	iiDelete(iif);
	throw ReadError("Error reading MRC header via libiimod");
      }

    data._boundingBox = BoundingBox(header.nxstart,
			       header.nystart,
			       header.nzstart,
			       header.nxstart + header.xlen,
			       header.nystart + header.ylen,
			       header.nzstart + header.zlen);

    //only one timestep
    data._tmin = data._tmax = 0.0;

    /* new volume, so min/max is now unset */
    data._minIsSet.clear();
    data._minIsSet.resize(data._numVariables);
    for(unsigned int i=0; i<data._minIsSet.size(); i++) data._minIsSet[i].resize(data._numTimesteps);
    data._min.clear();
    data._min.resize(data._numVariables);
    for(unsigned int i=0; i<data._min.size(); i++) data._min[i].resize(data._numTimesteps);
    data._maxIsSet.clear();
    data._maxIsSet.resize(data._numVariables);
    for(unsigned int i=0; i<data._maxIsSet.size(); i++) data._maxIsSet[i].resize(data._numTimesteps);
    data._max.clear();
    data._max.resize(data._numVariables);
    for(unsigned int i=0; i<data._max.size(); i++) data._max[i].resize(data._numTimesteps);

    /* the min and max values are in the header */
    data._min[0][0] = header.amin;
    data._max[0][0] = header.amax;
    data._minIsSet[0][0] = true;
    data._maxIsSet[0][0] = true;

    iiClose(iif);
    iiDelete(iif);
  }

  // ---------------------------
  // IMOD_MRC_IO::readVolumeFile
  // ---------------------------
  // Purpose:
  //   Writes to a Volume object after reading from a volume file.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  void IMOD_MRC_IO::readVolumeFile(Volume& vol,
				   const std::string& filename, 
				   unsigned int var, unsigned int time,
				   uint64 off_x, uint64 off_y, uint64 off_z,
				   const Dimension& subvoldim) const
  {
    using namespace std;
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    if(var > 0)
      throw IndexOutOfBounds("Variable index out of bounds.");
    if(time > 0)
      throw IndexOutOfBounds("Timestep index out of bounds.");
    if(subvoldim.isNull())
      throw IndexOutOfBounds("Specified subvolume dimension is null.");

    VolumeFileInfo vfi(filename);
    
    if((off_x + subvoldim[0] - 1 >= vfi.XDim()) ||
       (off_y + subvoldim[1] - 1 >= vfi.YDim()) ||
       (off_z + subvoldim[2] - 1 >= vfi.ZDim()))
      throw IndexOutOfBounds("Subvolume specified is outside volume dimensions");

    /** errors checked, now we can start modifying the volume object */
    vol.voxelType(vfi.voxelType());
    vol.dimension(subvoldim);
    vol.boundingBox(BoundingBox(vfi.XMin()+off_x*vfi.XSpan(),
				vfi.YMin()+off_y*vfi.YSpan(),
				vfi.ZMin()+off_z*vfi.ZSpan(),
				vfi.XMin()+(off_x+subvoldim[0]-1)*vfi.XSpan(),
				vfi.YMin()+(off_y+subvoldim[1]-1)*vfi.YSpan(),
				vfi.ZMin()+(off_z+subvoldim[2]-1)*vfi.ZSpan()));
    vol.min(vfi.min());
    vol.max(vfi.max());

    //Finally read in the data, one section at a time. Extract out the
    //part of the section we need as specified in the subvoldim.
    ImodImageFile *iif = iiOpen(const_cast<char*>(filename.c_str()),"rb");
    if(iif == NULL)
      throw ReadError("Error opening MRC file via libiimod");
    boost::scoped_array<char> section(new char[vfi.XDim()*vfi.YDim()*vfi.voxelSize()]);
    for(size_t k=off_z; k<=(off_z+subvoldim[2]-1); k++)
      {
	if(iiReadSection(iif,section.get(),k)==-1)
	  {
	    iiClose(iif);
	    iiDelete(iif);
	    throw ReadError("Error reading MRC file via libiimod");
	  }
	for(size_t j=off_y; j<=(off_y+subvoldim[1]-1); j++)
	  memcpy(*vol+
		 (k-off_z)*vol.XDim()*vol.YDim()*vol.voxelSize()+
		 (j-off_y)*vol.XDim()*vol.voxelSize(),
		 section.get()+(j*vfi.XDim()+off_x)*vfi.voxelSize(),
		 vol.XDim()*vol.voxelSize());
      }

    //convert signed values to unsigned since volmagick doesnt support signed
    switch(vol.voxelType())
      {
      case UChar:
	{
	  vol.min(((vol.min() - double(SCHAR_MIN))/(double(SCHAR_MAX) - double(SCHAR_MIN)))*double(UCHAR_MAX));
	  vol.max(((vol.max() - double(SCHAR_MIN))/(double(SCHAR_MAX) - double(SCHAR_MIN)))*double(UCHAR_MAX));
	  size_t len = vol.XDim()*vol.YDim()*vol.ZDim();
	  for(int i=0; i<len; i++)
	    {
	      char c = *((char*)(*vol+i*vol.voxelSize()));
	      *((unsigned char*)(*vol+i*vol.voxelSize())) =
		(unsigned char)(((double(c) - double(SCHAR_MIN))/(double(SCHAR_MAX) - double(SCHAR_MIN)))*double(UCHAR_MAX));
	    }
	}
	break;
      case UShort:
	{
	  vol.min(((vol.min() - double(SHRT_MIN))/(double(SHRT_MAX) - double(SHRT_MIN)))*double(USHRT_MAX));
	  vol.max(((vol.max() - double(SHRT_MIN))/(double(SHRT_MAX) - double(SHRT_MIN)))*double(USHRT_MAX));	  
	  size_t len = vol.XDim()*vol.YDim()*vol.ZDim();
	  for(int i=0; i<len; i++)
	    {
	      short c = *((short*)(*vol+i*vol.voxelSize()));
	      *((unsigned short*)(*vol+i*vol.voxelSize())) =
		(unsigned short)(((float(c) - double(SHRT_MIN))/(double(SHRT_MAX) - double(SHRT_MIN)))*double(USHRT_MAX));
	    }
	}
	break;
      default: break;
      }

    iiClose(iif);
    iiDelete(iif);
  }
}
