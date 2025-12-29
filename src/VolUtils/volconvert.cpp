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

/* $Id: volconvert.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/StdErrOpStatus.h>
#include <VolMagick/VolMagick.h>
#include <VolMagick/endians.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

// #define OUT_OF_CORE

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0]
         << " <input volume file> <output volume file>" << endl;
    return 1;
  }

  try {
#ifndef OUT_OF_CORE
    cerr << "In-core convert" << endl;
    VolMagick::Volume vol;

    // TODO: read/write a slice at a time instead of reading the whole volume
    // in memory then writing it out...
    VolMagick::readVolumeFile(vol, argv[1] /*,var,time*/);
    // VolMagick::writeVolumeFile(vol,argv[2]/*,var,time*/);
    VolMagick::createVolumeFile(vol, argv[2]);
#else
    cerr << "Out-of-core convert" << endl;
    VolMagick::VolumeFileInfo volinfo;
    volinfo.read(argv[1]);

    // VolMagick::createVolumeFile in Utlity.h
    VolMagick::createVolumeFile(argv[2], volinfo);

    // cout<<"convert volinfo:" << volinfo.boundingBox().minx <<" " <<
    // volinfo.boundingBox().maxx<< endl;
    // read in slice by slice
    for (unsigned int k = 0; k < volinfo.ZDim(); k++) {
      for (unsigned int var = 0; var < volinfo.numVariables(); var++)
        for (unsigned int time = 0; time < volinfo.numTimesteps(); time++) {
          VolMagick::Volume vol;
          readVolumeFile(
              vol, argv[1], var, time, 0, 0, k,
              VolMagick::Dimension(volinfo.XDim(), volinfo.YDim(), 1));

          vol.desc(volinfo.name(var));
          writeVolumeFile(vol, argv[2], var, time, 0, 0, k);
        }
      fprintf(stderr, "Converting: %5.2f %%\r",
              (((float)k) / ((float)((int)(volinfo.ZDim() - 1)))) * 100.0);
    }
    fprintf(stderr, "\n");
#endif
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
