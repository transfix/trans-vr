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

/* Program to switch values in the volume */
/* Uses VolMagick, use -L VolMagick.a while linking */

#include <VolMagick/VolMagick.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  using namespace std;

  if (argc != 2) {
    cout << "Usage: flipvol <filein>" << endl;
    cout << "Flips the sign of each voxel value." << endl;
    return 0;
  }

  try {
    VolMagick::Volume v;
    VolMagick::readVolumeFile(v, argv[1]);

    int x = v.XDim(), y = v.YDim(), z = v.ZDim();

    for (int i = 0; i < x; i++)
      for (int j = 0; j < y; j++)
        for (int k = 0; k < z; k++)
          v(i, j, k, v(i, j, k) * -1.0);

    VolMagick::writeVolumeFile(v, argv[1]);
  } catch (VolMagick::Exception &e) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
