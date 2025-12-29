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

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <iostream>
#include <set>
#include <sstream>
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

typedef boost::tuple<double, double, double> Color;

int main(int argc, char **argv) {
  if (argc < 5) {
    cerr
        << "Usage: volSetExtents <volume> <min> <max> <output volume>" << endl
        << "This program sets the minum and maximum voxel values of a volume."
        << endl;

    return 1;
  }

  try {
    VolMagick::Volume inputVol;
    VolMagick::Volume outputVol;

    VolMagick::readVolumeFile(inputVol,
                              argv[1]); /// first argument is input volume

    double minval = atof(argv[2]);
    double maxval = atof(argv[3]);

    cout << inputVol.XDim() << endl;

    int xsize, ysize, zsize;
    xsize = inputVol.XDim();
    ysize = inputVol.YDim();
    zsize = inputVol.ZDim();

    cout << "Volume Size: " << xsize << " x " << ysize << " x " << zsize
         << endl;

    outputVol.voxelType(inputVol.voxelType());
    outputVol.dimension(inputVol.dimension());
    outputVol.boundingBox(inputVol.boundingBox());

    bool minachieved = false;
    bool maxachieved = false;

    for (unsigned int i = 0; i < xsize; i++)
      for (unsigned int j = 0; j < ysize; j++)
        for (unsigned int k = 0; k < zsize; k++) {

          if (inputVol(i, j, k) > maxval) {
            outputVol(i, j, k, maxval);
            maxachieved = true;
          } else if (inputVol(i, j, k) < minval) {
            outputVol(i, j, k, minval);
            minachieved = true;
          } else {
            outputVol(i, j, k, inputVol(i, j, k));
          }
        }

    if (!minachieved) {
      cout
          << "Minimum Value is never achieved.  Adding dummy pixel at (0,0,0)"
          << endl;
      outputVol(0, 0, 0, minval);
    }
    if (!maxachieved) {
      cout
          << "Maximum Value is never achieved.  Adding dummy pixel at (0,0,1)"
          << endl;
      outputVol(0, 0, 1, maxval);
    }

    VolMagick::createVolumeFile(argv[4], outputVol);

    VolMagick::writeVolumeFile(outputVol,
                               argv[4]); /// first argument is file name

  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
