/*
  Copyright 2005-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolUtils.

  VolUtils is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolUtils is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: volcache.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/StdErrOpStatus.h>
#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>
#include <algorithm>
#include <boost/array.hpp>
#include <boost/format.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define OUT_OF_CORE

using namespace std;

int main(int argc, char **argv) {
  if (argc < 11) {
    cerr << "Testing volume cache functionality in VolMagick...\n"
         << "Usage: " << argv[0]
         << " <volume file> <var> <time> <minx> <miny> <minz> "
         << " <maxx> <maxy> <maxz> <output volume file>" << endl;
    return 1;
  }

  try {
    VolMagick::StdErrOpStatus status;
    VolMagick::setDefaultMessenger(&status);

#if 0
      VolMagick::VolumeCache volcache;

      //create a cache using a simple in-core method... the real deal should be out of core
      //because volumes can be of arbitrary size
#ifndef OUT_OF_CORE
      {
	VolMagick::readVolumeFile(vol,argv[1],var,time);

	boost::array<VolMagick::uint64,3> dims = { vol.XDim(), vol.YDim(), vol.ZDim() };
	VolMagick::uint64 maxdim = *std::max_element(dims.begin(),dims.end());
	std::cout << maxdim << std::endl;

	for(VolMagick::uint64 dim = 2 << 3;
	    dim < maxdim;
	    dim <<= 1)
	  {
	    VolMagick::uint64 dimx = std::min(dim,vol.XDim());
	    VolMagick::uint64 dimy = std::min(dim,vol.YDim());
	    VolMagick::uint64 dimz = std::min(dim,vol.ZDim());

	    VolMagick::Volume subvol(vol);
	    std::cout << "Resizing to " <<dimx<<"x"<<dimy<<"x"<<dimz<< std::endl;
	    subvol.resize(VolMagick::Dimension(dimx,dimy,dimz));
	    std::string outfilename = boost::str(boost::format("%1%.%2%.rawiv")
						 % argv[10]
						 % dim);
	    VolMagick::writeVolumeFile(subvol,outfilename);
	    VolMagick::VolumeFileInfo vfi;
	    vfi.read(outfilename);
	    volcache.add(vfi);
	  }
      }
#else
      {
	using namespace std;
	using namespace boost;

	VolMagick::VolumeFileInfo volinfo(argv[1]);

	array<VolMagick::uint64,3> dims = {{ volinfo.XDim(), volinfo.YDim(), volinfo.ZDim() }};
	VolMagick::uint64 maxdim = *max_element(dims.begin(),dims.end());

	const VolMagick::uint64 max_chunk_dim_size = 128;
	const VolMagick::Dimension max_chunk_dim(max_chunk_dim_size,
						 max_chunk_dim_size,
						 max_chunk_dim_size);
	
	double xdim_ratio = ceil(max(double(volinfo.XDim())/double(max_chunk_dim_size),1.0));
	double ydim_ratio = ceil(max(double(volinfo.YDim())/double(max_chunk_dim_size),1.0));
	double zdim_ratio = ceil(max(double(volinfo.ZDim())/double(max_chunk_dim_size),1.0));
	double x_interval = (volinfo.XMax() - volinfo.XMin())/xdim_ratio;
	double y_interval = (volinfo.YMax() - volinfo.YMin())/ydim_ratio;
	double z_interval = (volinfo.ZMax() - volinfo.ZMin())/zdim_ratio;

	for(VolMagick::uint64 dim = 2 << 3;
	    dim < maxdim;
	    dim <<= 1)
	  {
	    VolMagick::Volume vol;
	    VolMagick::Dimension cur_dim(min(dim,volinfo.XDim()),
					 min(dim,volinfo.YDim()),
					 min(dim,volinfo.ZDim()));

	    string outfilename = str(format("%1%.%2%.rawiv")
				     % argv[10]
				     % dim);
	    
	    VolMagick::createVolumeFile(outfilename,
					volinfo.boundingBox(),
					cur_dim,
					std::vector<VolMagick::VoxelType>(1,VolMagick::UChar));
	    for(double z_split = volinfo.ZMin();
		z_split < volinfo.ZMax();
		z_split += z_interval)
	      for(double y_split = volinfo.YMin();
		  y_split < volinfo.YMax();
		  y_split += y_interval)
		for(double x_split = volinfo.XMin();
		    x_split < volinfo.XMax();
		    x_split += x_interval)
		  {
		    VolMagick::readVolumeFile(vol,volinfo.filename(),
					      var,time,
					      VolMagick::BoundingBox(x_split,y_split,z_split,
								     x_split+x_interval,
								     y_split+y_interval,
								     z_split+z_interval));
		    vol.min(volinfo.min());
		    vol.max(volinfo.max());
		    vol.map(0.0,255.0);
		    vol.voxelType(VolMagick::UChar);
		    VolMagick::writeVolumeFile(vol,outfilename,
					       var,time,
					       VolMagick::BoundingBox(x_split,y_split,z_split,
								      x_split+x_interval,
								      y_split+y_interval,
								      z_split+z_interval));
		  }

	    VolMagick::VolumeFileInfo vfi;
	    vfi.read(outfilename);
	    volcache.add(vfi);
	  } 
      }
#endif
      
      //now do a request to the cache and output the volume requested
      {
	VolMagick::BoundingBox requestRegion(atof(argv[4]),atof(argv[5]),atof(argv[6]),
					     atof(argv[7]),atof(argv[8]),atof(argv[9]));
	VolMagick::Volume requestVol(volcache.get(requestRegion));
	VolMagick::writeVolumeFile(requestVol,argv[10]);
      }
#endif

    VolMagick::VolumeFileInfo volinfo(argv[1]);
    unsigned int var = atoi(argv[2]);
    unsigned int time = atoi(argv[3]);

    VolMagick::Volume vol;
    VolMagick::BoundingBox requestRegion(atof(argv[4]), atof(argv[5]),
                                         atof(argv[6]), atof(argv[7]),
                                         atof(argv[8]), atof(argv[9]));
    VolMagick::readVolumeFile(vol, volinfo.filename(), var, time,
                              requestRegion);
    VolMagick::createVolumeFile(vol, argv[10]);
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
