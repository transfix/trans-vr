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

/* $Id: volmerge.cpp 5196 2012-02-28 16:18:50Z zqyork $ */

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
  VolMagickOpStatus();

  void start(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 numSteps) const;
  void step(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 curStep) const;
  void end(const VolMagick::Voxels *vox, Operation op) const;

private:
  mutable VolMagick::uint64 _numSteps;
  char *_opStrings[12];
};

VolMagickOpStatus::VolMagickOpStatus()
{
  _opStrings[0] = (char *)"CalculatingMinMax"; _opStrings[1] = (char *)"CalculatingMin";
  _opStrings[2] = (char *)"CalculatingMax"; _opStrings[3] = (char *)"SubvolumeExtraction";
  _opStrings[4] = (char *)"Fill"; _opStrings[5] = (char *)"Map"; _opStrings[6] = (char *)"Resize";
  _opStrings[7] = (char *)"Composite"; _opStrings[8] = (char *)"BilateralFilter";
  _opStrings[9] = (char *)"ContrastEnhancement"; _opStrings[10] = (char *)"AnisotropicDiffusion";
  _opStrings[11] = (char *)"CombineWith";
}

void VolMagickOpStatus::start(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 numSteps) const
{
  _numSteps = numSteps;
}

void VolMagickOpStatus::step(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 curStep) const
{
  fprintf(stderr,"%s: %5.2f %%\r",_opStrings[op],(((float)curStep)/((float)((int)(_numSteps-1))))*100.0);
}

void VolMagickOpStatus::end(const VolMagick::Voxels *vox, Operation op) const
{
  printf("\n");
}


int main(int argc, char **argv)
{
  VolMagickOpStatus status;
  VolMagick::setDefaultMessenger(&status);

  if(argc < 4)
    {
      cerr << "Usage: " 
	   << argv[0]
	   << " <copy | add | subtract>"
	   << " <input volume file> [<input volume file ...>] <output volume file> <dimx> <dimy> <dimz> " << endl
	   << "Example: " << argv[0] << " add vol1.rawiv vol2.rawiv outvol.rawiv 100 100 100" << endl;
      return 1;
    }

  try
    {
      VolMagick::Volume outvol,invol;

      VolMagick::readVolumeFile(invol,argv[2]); //read the first volume file

      outvol = invol;

      for(unsigned int i = 3; i < (unsigned int)(argc-4); i++)
	{
	  VolMagick::readVolumeFile(invol,argv[i]);
	  outvol.combineWith(invol,VolMagick::Dimension(atoi(argv[argc-3]),
							atoi(argv[argc-2]),
							atoi(argv[argc-1])));
	  outvol.desc(outvol.desc() + "&" + invol.desc());
	}

//      VolMagick::createVolumeFile(argv[argc-4],
//				  outvol.boundingBox(),
//				  outvol.dimension(),
//				  std::vector<VolMagick::VoxelType>(1,outvol.voxelType()));

      VolMagick::createVolumeFile(outvol,argv[argc-4]);
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
