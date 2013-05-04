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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: volresize.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>

#include <fstream>

using namespace std;

class VolMagickOpStatus : public VolMagick::VoxelOperationStatusMessenger
{
public:
  void start(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 numSteps) const
  {
    _numSteps = numSteps;
  }

  void step(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 curStep) const
  {
    const char *opStrings[] = { "CalculatingMinMax", "CalculatingMin", "CalculatingMax",
				"SubvolumeExtraction", "Fill", "Map", "Resize", "Composite",
				"BilateralFilter", "ContrastEnhancement"};

    fprintf(stderr,"%s: %5.2f %%\r",opStrings[op],(((float)curStep)/((float)((int)(_numSteps-1))))*100.0);
  }

  void end(const VolMagick::Voxels *vox, Operation op) const
  {
    printf("\n");
  }

private:
  mutable VolMagick::uint64 _numSteps;
};

int main(int argc, char **argv)
{
  if(argc < 6)
    {
      cerr << "Usage: " << argv[0] << " <input volume file> <output volume file> <target X dim> <target Y dim> <target Z dim> [ooc]" << endl;
      cerr << "If ooc is present, volresize will run in out-of-core mode." << endl;
      return 1;
    }

  try
    {
	bool ooc_mode = false;

	if(argc == 7)
	{
		std::string ooc_arg(argv[6]);
		if(ooc_arg == "ooc")
			ooc_mode = true;
	}

      VolMagickOpStatus status;
      VolMagick::setDefaultMessenger(&status);

      VolMagick::VolumeFileInfo volinfo;

      volinfo.read(argv[1]);

      VolMagick::Dimension targetDim(atoi(argv[3]),
					  atoi(argv[4]),
			  		  atoi(argv[5]));

      VolMagick::createVolumeFile(argv[2],
				  volinfo.boundingBox(),
				  targetDim,
				  volinfo.voxelTypes(),
				  volinfo.numVariables(),
				  volinfo.numTimesteps(),
				  volinfo.TMin(),volinfo.TMax());

      for(unsigned int var=0; var<volinfo.numVariables(); var++)
	for(unsigned int time=0; time<volinfo.numTimesteps(); time++)
	  {
	    if(ooc_mode)
	    {
	    const VolMagick::uint64 maxdim = 128; //read in 128^3 chunk
	    boost::array<double,3> theSize =
	      {
		maxdim*volinfo.XSpan(),
		maxdim*volinfo.YSpan(),
		maxdim*volinfo.ZSpan()
	      };
	    const VolMagick::BoundingBox &bbox = volinfo.boundingBox();
	    for(double off_z = bbox.minz;
		off_z < bbox.maxz;
		off_z += theSize[0])
	      for(double off_y = bbox.miny;
		  off_y < bbox.maxy;
		  off_y += theSize[1])
		for(double off_x = bbox.minx;
		    off_x < bbox.maxx;
		    off_x += theSize[2])
		  {
	    	    VolMagick::Volume vol;
                    VolMagick::BoundingBox subvolbox(off_x,off_y,off_z,
					  std::min(off_x+theSize[0],bbox.maxx),
					  std::min(off_y+theSize[1],bbox.maxy),
					  std::min(off_z+theSize[2],bbox.maxz));
		    VolMagick::Dimension subvolDim( 
                         VolMagick::uint64(double(targetDim.xdim)*((subvolbox.maxx-subvolbox.minx)/(bbox.maxx-bbox.minx))),
			 VolMagick::uint64(double(targetDim.ydim)*((subvolbox.maxy-subvolbox.miny)/(bbox.maxy-bbox.miny))),
			 VolMagick::uint64(double(targetDim.zdim)*((subvolbox.maxz-subvolbox.minz)/(bbox.maxz-bbox.minz))));
		    VolMagick::readVolumeFile(vol,argv[1],var,time,subvolbox);
		    vol.resize(subvolDim);
		    VolMagick::writeVolumeFile(vol,argv[2],var,time,subvolbox);
		  }
            }
	    else
	    {
	    VolMagick::Volume vol;
	    readVolumeFile(vol,argv[1],var,time);
	    vol.resize(VolMagick::Dimension(atoi(argv[3]),
					    atoi(argv[4]),
					    atoi(argv[5])));
	    vol.desc(volinfo.name(var));
	    writeVolumeFile(vol,argv[2],var,time);
	    }
	  }
    }
  catch(VolMagick::Exception &e)
    {
      cerr << e.what() << endl;
    }
  catch(std::exception &e)
    {
      cerr << e.what() << endl;
    }

  return 0;
}
