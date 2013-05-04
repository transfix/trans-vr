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

/* $Id: voltype.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <limits>

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
      cerr << "Usage: " << argv[0] << " <volume file> <var> <timestep> <volume type> <output volume file>" << endl;
      cerr << "<volume type> - one of 'uchar' 'ushort' 'uint' 'float' or 'double'" << endl;
      return 1;
    }

  try
    {
      VolMagickOpStatus status;
      VolMagick::setDefaultMessenger(&status);

      VolMagick::Volume vol;
      
      VolMagick::readVolumeFile(vol,argv[1],atoi(argv[2]),atoi(argv[3]));
      
      VolMagick::VoxelType voxtype;
      if(std::string(argv[4]) == "uchar") voxtype = VolMagick::UChar;
      else if(std::string(argv[4]) == "ushort") voxtype = VolMagick::UShort;
      else if(std::string(argv[4]) == "uint") voxtype = VolMagick::UInt;
      else if(std::string(argv[4]) == "float") voxtype = VolMagick::Float;
      else if(std::string(argv[4]) == "double") voxtype = VolMagick::Double;
      else throw VolMagick::UnsupportedVolumeFileType("Unknown volume type");

      if((vol.voxelType() == VolMagick::Float || vol.voxelType() == VolMagick::Double) &&
	     (voxtype == VolMagick::UChar))
        vol.map(0.0,double(std::numeric_limits<unsigned char>::max()));

      if((vol.voxelType() == VolMagick::Float || vol.voxelType() == VolMagick::Double) &&
	 (voxtype == VolMagick::UShort))
     vol.map(0.0,double(std::numeric_limits<unsigned short>::max()));

      if((vol.voxelType() == VolMagick::Float || vol.voxelType() == VolMagick::Double) &&
	 (voxtype == VolMagick::UInt))
     vol.map(0.0,double(std::numeric_limits<unsigned int>::max()));

      vol.voxelType(voxtype);

      VolMagick::createVolumeFile(argv[5],
                                  vol.boundingBox(),
                                  vol.dimension(),
                                  std::vector<VolMagick::VoxelType>(1,vol.voxelType()));	
      VolMagick::writeVolumeFile(vol,argv[5],atoi(argv[2]),atoi(argv[3]));
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
