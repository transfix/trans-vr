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

/* $Id: vol2image.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <Magick++.h>
#include <VolMagick/StdErrOpStatus.h>
#include <VolMagick/VolMagick.h>
#include <boost/format.hpp>
#include <boost/scoped_array.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;
using namespace boost;

int main(int argc, char **argv) {
  VolMagick::StdErrOpStatus status;
  VolMagick::setDefaultMessenger(&status);

  if (argc < 5) {
    cerr << "Usage: " << argv[0]
         << " <input volume file> <variable index> <time index> <output "
            "image filename>"
         << endl;
    return EXIT_FAILURE;
  }

  try {
    VolMagick::Volume vol;
    Magick::Image image;
    VolMagick::readVolumeFile(vol, argv[1], atoi(argv[2]), atoi(argv[3]));

    // get the filename strings
    string filename(argv[4]);
    string basename(filename);
    string extension(filename);
    basename.erase(filename.rfind('.'));
    extension.erase(0, filename.rfind('.') + 1);

    vol.map(0.0, 255.0);
    vol.voxelType(VolMagick::UChar);

    for (VolMagick::uint64 k = 0; k < vol.ZDim(); k++) {
      VolMagick::Volume subvol;
      VolMagick::sub(subvol, vol, 0, 0, k,
                     VolMagick::Dimension(vol.XDim(), vol.YDim(), 1));

      // duplicate the slice 3 times to produce correct grayscale images
      scoped_array<unsigned char> subvolbuf(
          new unsigned char[vol.XDim() * vol.YDim() * 3]);
      for (VolMagick::uint64 i = 0; i < vol.XDim() * vol.YDim(); i++) {
        subvolbuf[i * 3 + 0] = subvol(i);
        subvolbuf[i * 3 + 1] = subvol(i);
        subvolbuf[i * 3 + 2] = subvol(i);
      }

      // We only need 1 value since after grayscale conversion R = G = B
      image.read(subvol.XDim(), subvol.YDim(), "RGB", Magick::CharPixel,
                 subvolbuf.get());
      image.modifyImage();
      image.type(Magick::GrayscaleType);
      image.write(str(format("%1%.%2$ 05d.%3%") % basename % k % extension));

      fprintf(stderr, "Progress: %5.2f %%\r",
              (((float)k) / ((float)((int)(vol.ZDim() - 1)))) * 100.0);
    }

    fprintf(stderr, "\n");
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  } catch (Magick::Exception &e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
