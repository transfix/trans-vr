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
#include <iostream>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 8) {
    cerr << "Usage: inputfile, bb_xdim1, bb_xdim2, bb_ydim1, bb_ydim2, "
            "bb_zdim1, bb_zdim2"
         << " \n"
         << "The program gives the mean voxel value and the standard "
            "deviation, sigma, of voxels"
         << " \n"
         << " in the user provided bounding box \n"
         << argv[0];

    return 1;
  }

  try {
    VolMagick::Volume inputVol;

    VolMagick::Volume outputVol;

    VolMagick::readVolumeFile(inputVol,
                              argv[1]); /// first argument is input volume

    int bb_xdim1 = atoi(argv[2]);

    int bb_xdim2 = atoi(argv[3]);

    int bb_ydim1 = atoi(argv[4]);

    int bb_ydim2 = atoi(argv[5]);

    int bb_zdim1 = atoi(argv[6]);

    int bb_zdim2 = atoi(argv[7]);

    double sum_intensity = 0;

    double sum_square = 0;

    double countVox = 0;

    double countNZVox = 0;

    cout << "vol min: " << inputVol.min() << "\n";

    cout << "xdim1: " << bb_xdim1 << "\n";
    cout << "xdim1: " << bb_xdim2 << "\n";

    cout << "ydim1: " << bb_ydim1 << "\n";
    cout << "ydim1: " << bb_ydim2 << "\n";

    cout << "zdim1: " << bb_zdim1 << "\n";
    cout << "zdim1: " << bb_zdim2 << "\n";

    for (unsigned int i = bb_xdim1; i < bb_xdim2; i++)
      for (unsigned int j = bb_ydim1; j < bb_ydim2; j++)
        for (unsigned int k = bb_zdim1; k < bb_zdim2; k++)

        {

          sum_intensity = sum_intensity + inputVol(i, j, k);
          countVox++;

          if (inputVol(i, j, k))
            countNZVox++;
        }

    float mean = sum_intensity / countVox;

    cout << "sum_intensity: " << sum_intensity << "\n";
    cout << "count Vox: " << countVox << "\n";
    cout << "Non zero Voxels: " << countNZVox << "\n";
    cout << "mean value: " << mean << "\n";

    for (unsigned int i = bb_xdim1; i < bb_xdim2; i++)
      for (unsigned int j = bb_ydim1; j < bb_ydim2; j++)
        for (unsigned int k = bb_zdim1; k < bb_zdim2; k++)

        {

          float square = (inputVol(i, j, k) - mean);

          square = square * square;

          sum_square = sum_square + square;
        }

    float sigma = sqrt(sum_square / countVox);

    cout << "sigma: " << sigma;

    cout << endl;

  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
