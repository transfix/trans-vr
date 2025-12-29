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

/* $Id: volcombine.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

class VolMagickOpStatus : public VolMagick::VoxelOperationStatusMessenger {
public:
  VolMagickOpStatus();

  void start(const VolMagick::Voxels *vox, Operation op,
             VolMagick::uint64 numSteps) const;
  void step(const VolMagick::Voxels *vox, Operation op,
            VolMagick::uint64 curStep) const;
  void end(const VolMagick::Voxels *vox, Operation op) const;

private:
  mutable VolMagick::uint64 _numSteps;
  // char *_opStrings[12];
  string _opStrings[12];
};

VolMagickOpStatus::VolMagickOpStatus() {
  _opStrings[0] = "CalculatingMinMax";
  _opStrings[1] = "CalculatingMin";
  _opStrings[2] = "CalculatingMax";
  _opStrings[3] = "SubvolumeExtraction";
  _opStrings[4] = "Fill";
  _opStrings[5] = "Map";
  _opStrings[6] = "Resize";
  _opStrings[7] = "Composite";
  _opStrings[8] = "BilateralFilter";
  _opStrings[9] = "ContrastEnhancement";
  _opStrings[10] = "AnisotropicDiffusion";
  _opStrings[11] = "CombineWith";
}

void VolMagickOpStatus::start(const VolMagick::Voxels *vox, Operation op,
                              VolMagick::uint64 numSteps) const {
  _numSteps = numSteps;
}

void VolMagickOpStatus::step(const VolMagick::Voxels *vox, Operation op,
                             VolMagick::uint64 curStep) const {
  fprintf(stderr, "%s: %5.2f %%\r", _opStrings[op].c_str(),
          (((float)curStep) / ((float)((int)(_numSteps - 1)))) * 100.0);
}

void VolMagickOpStatus::end(const VolMagick::Voxels *vox,
                            Operation op) const {
  printf("\n");
}

int main(int argc, char **argv) {
  VolMagickOpStatus status;
  VolMagick::setDefaultMessenger(&status);

  if (argc < 6) {
    cerr << "Usage: " << argv[0]
         << " <input volume file> <input volume file ...> <output volume "
            "file> <target X dim> <target Y dim> <target Z dim>"
         << endl;
    return 1;
  }

  try {
    VolMagick::Volume outvol, invol;

    VolMagick::readVolumeFile(invol, argv[1]); // read the first volume file

    // DEBUG
#if 0
      {
	VolMagick::Volume testVol(VolMagick::Dimension(256,256,256),
				  VolMagick::Float,
				  invol.boundingBox());

	for(unsigned int k = 0; k<256; k++)
	  for(unsigned int j = 0; j<256; j++)
	    for(unsigned int i = 0; i<256; i++)
	      {
		testVol(i,j,k, invol.interpolate(i*testVol.XSpan()+invol.XMin(),
						 j*testVol.YSpan()+invol.YMin(),
						 k*testVol.ZSpan()+invol.ZMin()));
	      }

	VolMagick::writeVolumeFile(testVol,"test.rawiv");
      }
#endif

    outvol = invol;
    // if argc == 6, there is nothing to combine, so just resize and pretend
    // we did anything
    if (argc == 6)
      outvol.resize(VolMagick::Dimension(
          atoi(argv[argc - 3]), atoi(argv[argc - 2]), atoi(argv[argc - 1])));
    for (unsigned int i = 2; i < (unsigned int)(argc - 4); i++) {
      VolMagick::readVolumeFile(invol, argv[i]);
      outvol.combineWith(invol, VolMagick::Dimension(atoi(argv[argc - 3]),
                                                     atoi(argv[argc - 2]),
                                                     atoi(argv[argc - 1])));
      outvol.desc(outvol.desc() + "&" + invol.desc());
    }

    VolMagick::createVolumeFile(
        argv[argc - 4], outvol.boundingBox(), outvol.dimension(),
        std::vector<VolMagick::VoxelType>(1, outvol.voxelType()));

    VolMagick::writeVolumeFile(outvol, argv[argc - 4]);
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
