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

/* $Id: catvols.cpp 4742 2011-10-21 22:09:44Z transfix $ */

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
                               "ContrastEnhancement"};

    fprintf(stderr, "%s: %5.2f %%\r", opStrings[op],
            (((float)curStep) / ((float)((int)(_numSteps - 1)))) * 100.0);
  }

  void end(const VolMagick::Voxels *vox, Operation op) const { printf("\n"); }

private:
  mutable VolMagick::uint64 _numSteps;
};

int main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "Usage: " << argv[0]
         << " <volume file> <volume file> ... <multi timestep volume file>"
         << endl;
    return 1;
  }

  try {
    VolMagickOpStatus status;

    VolMagick::setDefaultMessenger(&status);

    double realmin, realmax;
    VolMagick::Volume vol;
    std::vector<VolMagick::VolumeFileInfo> volinfos(argc - 2);
    volinfos[0].read(argv[1]);
    realmin = volinfos[0].min();
    realmax = volinfos[0].max();
    for (unsigned int i = 0; i < argc - 2; i++) {
      volinfos[i].read(argv[i + 1]);
      if (realmin > volinfos[i].min())
        realmin = volinfos[i].min();
      if (realmax < volinfos[i].max())
        realmax = volinfos[i].max();
    }

    cout << "Realmin: " << realmin << endl;
    cout << "Realmax: " << realmax << endl;

    VolMagick::createVolumeFile(
        argv[argc - 1], volinfos[0].boundingBox(), volinfos[0].dimension(),
        std::vector<VolMagick::VoxelType>(1, VolMagick::UChar), 1, argc - 2,
        0.0, double(argc - 3));

    for (unsigned int i = 0; i < argc - 2; i++) {
      VolMagick::readVolumeFile(vol, argv[i + 1]);
      // so the mapping is correct across all timesteps, we must set the real
      // min and max across time
      vol.min(realmin);
      vol.max(realmax);
      vol.map(0.0, 255.0);
      vol.voxelType(VolMagick::UChar);
      VolMagick::writeVolumeFile(vol, argv[argc - 1], 0, i);
    }
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
