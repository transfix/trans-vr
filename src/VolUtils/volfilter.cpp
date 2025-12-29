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

/* $Id: volfilter.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

class VolMagickOpStatus : public VolMagick::VoxelOperationStatusMessenger {
public:
  void start(const VolMagick::Voxels *vox, Operation op,
             VolMagick::uint64 numSteps) const {
    _numSteps = numSteps;
  }

  void step(const VolMagick::Voxels *vox, Operation op,
            VolMagick::uint64 curStep) const {
    const char *opStrings[] = {"CalculatingMinMax",
                               "CalculatingMin",
                               "CalculatingMax",
                               "SubvolumeExtraction",
                               "Fill",
                               "Map",
                               "Resize",
                               "Composite",
                               "BilateralFilter",
                               "ContrastEnhancement",
                               "AnisotropicDiffusion"};

    fprintf(stderr, "%s: %5.2f %%\r", opStrings[op],
            (((float)curStep) / ((float)((int)(_numSteps - 1)))) * 100.0);
  }

  void end(const VolMagick::Voxels *vox, Operation op) const { printf("\n"); }

private:
  mutable VolMagick::uint64 _numSteps;
};

int main(int argc, char **argv) {
  if (argc < 5) {
    cerr << "Usage: " << argv[0]
         << " <input volume file> <var> <time> <output volume file> "
            "[radiometric sigma (default 200)] [spatial sigma (default 1.5)] "
            "[filter radius (default 2)]"
         << endl;
    return 1;
  }

  try {
    VolMagickOpStatus status;
    VolMagick::setDefaultMessenger(&status);

    unsigned int var, time;
    var = atoi(argv[2]);
    time = atoi(argv[3]);

    VolMagick::Volume vol;
    VolMagick::readVolumeFile(vol, argv[1], var, time);

    double radiometric = 200.0;
    double spatial = 1.5;
    unsigned int radius = 2;

    if (argc >= 6)
      radiometric = atof(argv[5]);
    if (argc >= 7)
      spatial = atof(argv[6]);
    if (argc >= 8)
      radius = atoi(argv[7]);

    vol.bilateralFilter(radiometric, spatial, radius);

    // VolMagick::createVolumeFile(argv[4], vol);
    VolMagick::createVolumeFile(vol, argv[4]);

    //      VolMagick::writeVolumeFile(vol,argv[4]);
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
