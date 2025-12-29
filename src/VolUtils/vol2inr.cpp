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

/* $Id: vol2inr.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>
#include <boost/format.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace VolMagick {
VOLMAGICK_DEF_EXCEPTION(InvalidINRHeader);
};

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <volume file> <timestep index> <output inr file>"
              << std::endl;
    return 1;
  }

  try {
    VolMagick::VolumeFileInfo volinfo;
    std::vector<VolMagick::Volume> vols;
    std::string filename(argv[1]);
    std::string out_filename(argv[3]);
    int timestep = atoi(argv[2]);

    volinfo.read(filename);

    if (timestep >= volinfo.numTimesteps())
      throw VolMagick::IndexOutOfBounds("No such timestep index");

    vols.resize(volinfo.numVariables());
    for (unsigned int i = 0; i < volinfo.numVariables(); i++)
      VolMagick::readVolumeFile(vols[i], filename, i, timestep);

    {
      char buf[256];
      char header[256];

      FILE *output;
      size_t i, j, k, v;

      memset(buf, 0, 256);
      memset(header, '\n', 256);

      std::string datatype, pixsize;
      switch (volinfo.voxelTypes()[0]) {
      default:
      case VolMagick::UChar:
        datatype = "unsigned fixed";
        pixsize = "8 bits";
        break;
      case VolMagick::UShort:
        datatype = "unsigned fixed";
        pixsize = "16 bits";
        break;
      case VolMagick::UInt:
        datatype = "unsigned fixed";
        pixsize = "32 bits";
        break;
      case VolMagick::Float:
        datatype = "float";
        pixsize = "32 bits";
        break;
      case VolMagick::Double:
        datatype = "float";
        pixsize = "64 bits";
        break;
        // case VolMagick::UInt64: datatype = "unsigned fixed"; pixsize = "64
        // bits"; break;
      }

      strcpy(header,
             boost::str(
                 boost::format("#INRIMAGE-4#{\n"
                               "XDIM=%1%\n"
                               "YDIM=%2%\n"
                               "ZDIM=%3%\n"
                               "VDIM=%4%\n"
                               "VX=%5%\n"
                               "VY=%6%\n"
                               "VZ=%7%\n"
                               "TYPE=%8%\n"
                               "PIXSIZE=%9%\n"
                               "CPU=%10%\n") %
                 volinfo.dimension()[0] % volinfo.dimension()[1] %
                 volinfo.dimension()[2] % volinfo.numVariables() %
                 ((volinfo.boundingBox().maxx - volinfo.boundingBox().minx) /
                  (volinfo.dimension()[0] - 1)) %
                 ((volinfo.boundingBox().maxy - volinfo.boundingBox().miny) /
                  (volinfo.dimension()[1] - 1)) %
                 ((volinfo.boundingBox().maxz - volinfo.boundingBox().minz) /
                  (volinfo.dimension()[2] - 1)) %
                 datatype % pixsize % (big_endian() ? "sun" : "pc"))
                 .c_str());
      memcpy(header + 256 - 4, "##}\n", 4);

      if ((output = fopen(out_filename.c_str(), "wb")) == NULL) {
        //	    geterrstr(errno,buf,256);
        std::string errStr =
            "Error opening file '" + out_filename + "': " + buf;
        throw VolMagick::WriteError(errStr);
      }

      if (fwrite(header, 256, 1, output) != 1) {
        //	    geterrstr(errno,buf,256);
        std::string errStr =
            "Error writing header to file '" + out_filename + "': " + buf;
        fclose(output);
        throw VolMagick::WriteError(errStr);
      }

      // write a scanline at a time
      for (v = 0; v < volinfo.numVariables(); v++) {
        if (fwrite(*(vols[v]), vols[v].voxelSize(),
                   vols[v].dimension().size(),
                   output) != vols[v].dimension().size()) {
          //		geterrstr(errno,buf,256);
          std::string errStr = "Error writing volume data to file '" +
                               out_filename + "': " + buf;
          fclose(output);
          throw VolMagick::WriteError(errStr);
        }
      }

      fclose(output);
    }

  } catch (VolMagick::Exception &e) {
    std::cerr << e.what() << std::endl;
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
