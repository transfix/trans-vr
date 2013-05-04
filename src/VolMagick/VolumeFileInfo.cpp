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

/* $Id: VolumeFileInfo.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/VolumeFileInfo.h>
#include <VolMagick/Exceptions.h>
#include <VolMagick/VolumeFile_IO.h>
#include <VolMagick/Utility.h>

#include <CVC/App.h>

#include <boost/regex.hpp>

namespace VolMagick
{
  // --------------------
  // VolumeFileInfo::read
  // --------------------
  // Purpose:
  //   Refers to the handler map to choose an appropriate IO object for reading
  //    the requested volume file.  Use this function to initialize the VolumeFileInfo
  //    object with info from a volume file.
  // ---- Change History ----
  // ??/??/2007 -- Joe R. -- Initially implemented.
  // 11/13/2009 -- Joe R. -- Re-implemented using VolumeFile_IO handler map
  // 12/28/2009 -- Joe R. -- Collecting exception error strings
  // 09/08/2011 -- Joe R. -- Using splitRawFilename to extract real filename
  //                         if the provided filename is a file|obj tuple.
  void VolumeFileInfo::read(const std::string& filename)
  {
    std::string errors;
    boost::regex file_extension("^(.*)(\\.\\S*)$");
    boost::smatch what;

    std::string actualFileName;
    std::string objectName;

    boost::tie(actualFileName, objectName) =
      VolumeFile_IO::splitRawFilename(filename);

    if(boost::regex_match(actualFileName, what, file_extension))
      {
	if(VolumeFile_IO::handlerMap()[what[2]].empty())
	  throw UnsupportedVolumeFileType(std::string(BOOST_CURRENT_FUNCTION) + 
					  std::string(": Cannot read ") + filename);
	VolumeFile_IO::Handlers& handlers = VolumeFile_IO::handlerMap()[what[2]];
	//use the first handler that succeds
	for(VolumeFile_IO::Handlers::iterator i = handlers.begin();
	    i != handlers.end();
	    i++)
	  try
	    {
	      if(*i)
		{
		  (*i)->getVolumeFileInfo(_data,filename);
		  return;
		}
	    }
	  catch(VolMagick::Exception& e)
	    {
	      errors += std::string(" :: ") + e.what();
	    }
      }
    throw UnsupportedVolumeFileType(
      boost::str(
	boost::format("%1% : Cannot read '%2%'%3%") % 
	BOOST_CURRENT_FUNCTION %
	filename %
	errors
      )
    );
  }

  void VolumeFileInfo::calcMinMax(unsigned int var, unsigned int time) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    Volume vol;
    const uint64 maxdim = 128; //read in 128^3 chunks
    for(unsigned int off_z = 0; off_z < ZDim(); off_z+=maxdim)
      for(unsigned int off_y = 0; off_y < YDim(); off_y+=maxdim)
	for(unsigned int off_x = 0; off_x < XDim(); off_x+=maxdim)
	  {
	    Dimension read_dim(std::min(XDim()-off_x,maxdim),
			       std::min(YDim()-off_y,maxdim),
			       std::min(ZDim()-off_z,maxdim));
	    readVolumeFile(vol,filename(),var,time,
			   off_x,off_y,off_z,read_dim);
	    if(off_x==0 && off_y==0 && off_z==0)
	      {
		_data._min[var][time] = vol.min();
		_data._max[var][time] = vol.max();
	      }
	    else
	      {
		if(_data._min[var][time] > vol.min())
		  _data._min[var][time] = vol.min();
		if(_data._max[var][time] < vol.max())
		  _data._max[var][time] = vol.max();
	      }
	  }

    _data._minIsSet[var][time] = true;
    _data._maxIsSet[var][time] = true;
  }
}
