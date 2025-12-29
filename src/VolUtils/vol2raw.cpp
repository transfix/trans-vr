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

/* $Id: vol2raw.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <Contour/contour.h>
#include <Contour/datasetreg3.h>
#include <VolMagick/VolMagick.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>
// #include "CellQueue.h"

#include <boost/format.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;
using namespace boost;

static inline bool hasExt(const string &haystack, const string &needle) {
  return haystack.rfind(needle) == haystack.size() - needle.size();
}

static inline void writeRaw(const string &filename,
                            const vector<double> &points,
                            const vector<int> &tris,
                            const vector<double> &normal = vector<double>(),
                            const vector<double> &color = vector<double>()) {
  if ((points.size() % 3 != 0) || (tris.size() % 3 != 0))
    throw runtime_error(
        "points and tris vectors must have a size equal to a multiple of 3");

  ofstream outf(filename.c_str());
  if (!outf)
    throw runtime_error(string("Could not open ") + filename);

  if (!(outf << points.size() / 3 << " " << tris.size() / 3 << endl))
    throw runtime_error(
        "Could not write number of points or number of tris to file!");

  for (unsigned int i = 0; i < points.size(); i += 3) {
    outf << points[i + 0] << " " << points[i + 1] << " " << points[i + 2];
    if (normal.size() == points.size())
      outf << " " << normal[i + 0] << " " << normal[i + 1] << " "
           << normal[i + 2];
    if (color.size() == points.size())
      outf << " " << color[i + 0] << " " << color[i + 1] << " "
           << color[i + 2];
    outf << endl;
    if (!outf)
      throw runtime_error(str(format("Error writing vertex %1%") % i));
  }

  for (unsigned int i = 0; i < tris.size(); i += 3) {
    outf << tris[i + 0] << " " << tris[i + 1] << " " << tris[i + 2] << endl;
    if (!outf)
      throw runtime_error(str(format("Error writing triangle %1%") % i));
  }
}

int main(int argc, char **argv) {
  try {
    VolMagick::Volume vol;
    ConDataset *the_data;
    Contour3dData *contour3d;
    int dim[3];
    float span[3], orig[3];
    double isovalue = 0.0;

    int var, time, red, green, blue;
    string input_filename;
    string output_filename;
    int do_color;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "input,i", po::value<string>(&input_filename),
        "input file (volume: rawiv, rawv, mrc, pif, etc...)")(
        "output,o", po::value<string>(&output_filename),
        "output file (raw(c) triangle format)")(
        "isoval,n", po::value<double>(&isovalue)->default_value(1.0),
        "isovalue for contour extraction (some float between [0.0-255.0])")(
        "var,v", po::value<int>(&var)->default_value(0),
        "variable index of volume variable to extract isocontour from")(
        "time,t", po::value<int>(&time)->default_value(0),
        "timestep index of volume variable to extract isocontour from")(
        "color,c", po::value<int>(&do_color)->default_value(0),
        "Color mode: 0 - all vertices are white. 1 - enable colored vertices "
        "from color component volume variables. "
        "2 - arguments -r,-g,-b specify a color component directly.")(
        "red,r", po::value<int>(&red)->default_value(0),
        "Depending on color mode: variable index of red component of volume "
        "color, or red color component [0-255]")(
        "green,g", po::value<int>(&green)->default_value(0),
        "Depending on color mode: variable index of green component of "
        "volume color, or green color component [0-255]")(
        "blue,b", po::value<int>(&blue)->default_value(0),
        "Depending on color mode: variable index of blue component of volume "
        "color, or blue color component [0-255]");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cerr << desc << endl;
      cerr << "Example: " << argv[0]
           << " -i vol.rawiv -o contour.raw -v 0 -t 0 -n 1.0" << endl;
      return EXIT_FAILURE;
    } else if (!vm.count("input") || !vm.count("output")) {
      cerr << "error: Input and output files must be specified!" << endl;
      cerr << desc << endl;
      return EXIT_FAILURE;
    }

    VolMagick::readVolumeFile(vol, input_filename, var, time);

    dim[0] = vol.XDim();
    dim[1] = vol.YDim();
    dim[2] = vol.ZDim();
    span[0] = vol.XSpan();
    span[1] = vol.YSpan();
    span[2] = vol.ZSpan();
    orig[0] = vol.XMin();
    orig[1] = vol.YMin();
    orig[2] = vol.ZMin();

    vol.map(0.0, 255.0);
    vol.voxelType(VolMagick::UChar);

    the_data = newDatasetReg(CONTOUR_UCHAR, CONTOUR_REG_3D, 1, 1, dim, *vol);
    ((Datareg3 *)the_data->data->getData(0))->setOrig(orig);
    ((Datareg3 *)the_data->data->getData(0))->setSpan(span);

    if (isovalue < 0.0 || isovalue > 255.0)
      cerr << "warning: clamping isovalue " << isovalue << " to [0.0-255.0]"
           << endl;
    isovalue = isovalue > 255.0 ? 255.0
               : isovalue < 0.0 ? 0.0
                                : isovalue; // clamp to [0.0-255.0]

    cout << "Using isovalue: " << isovalue << endl;
    // saveContour3d(the_data,0,0,isovalue,NO_COLOR_VARIABLE,const_cast<char*>(output_filename.c_str()));
    contour3d = getContour3d(the_data, 0, 0, isovalue, NO_COLOR_VARIABLE);

    vector<double> points, color, normal;
    vector<int> tris;

    // collect points, normal, and triangle info into vectors for writing
    for (int i = 0; i < contour3d->nvert; i++) {
      points.push_back(contour3d->vert[i][0]);
      points.push_back(contour3d->vert[i][1]);
      points.push_back(contour3d->vert[i][2]);
      normal.push_back(contour3d->vnorm[i][0]);
      normal.push_back(contour3d->vnorm[i][1]);
      normal.push_back(contour3d->vnorm[i][2]);
    }

    for (int i = 0; i < contour3d->ntri; i++) {
      tris.push_back(contour3d->tri[i][0]);
      tris.push_back(contour3d->tri[i][1]);
      tris.push_back(contour3d->tri[i][2]);
    }

    // calculate color info
    switch (do_color) {
    case 1: {
      VolMagick::Volume red_vol, green_vol, blue_vol;
      VolMagick::readVolumeFile(red_vol, input_filename, red, time);
      VolMagick::readVolumeFile(green_vol, input_filename, green, time);
      VolMagick::readVolumeFile(blue_vol, input_filename, blue, time);

      // normalize to [0.0,255.0]
      red_vol.map(0.0, 255.0);
      green_vol.map(0.0, 255.0);
      blue_vol.map(0.0, 255.0);

      for (int i = 0; i < contour3d->nvert; i++) {
        // note that color components in raw files are within [0.0,1.0]
        color.push_back(red_vol.interpolate(contour3d->vert[i][0],
                                            contour3d->vert[i][1],
                                            contour3d->vert[i][2]) /
                        255.0);
        color.push_back(green_vol.interpolate(contour3d->vert[i][0],
                                              contour3d->vert[i][1],
                                              contour3d->vert[i][2]) /
                        255.0);
        color.push_back(blue_vol.interpolate(contour3d->vert[i][0],
                                             contour3d->vert[i][1],
                                             contour3d->vert[i][2]) /
                        255.0);
        if (!(i % 100))
          fprintf(stderr, "Calculating color info: %5.2f %%\r",
                  (float(i) / float(contour3d->nvert - 1)) * 100.0);
      }
      fprintf(stderr, "\n");
    } break;
    case 2: {
      // clamp color component values to [0-255]
      red = red > 255 ? 255 : red < 0 ? 0 : red;
      green = green > 255 ? 255 : green < 0 ? 0 : green;
      blue = blue > 255 ? 255 : blue < 0 ? 0 : blue;

      cout << str(format(
                      "Using color (%1%,%2%,%3%) for isocontour vertices.") %
                  red % green % blue)
           << endl;

      // set all vertices to this color
      for (int i = 0; i < contour3d->nvert; i++) {
        color.push_back(double(red) / double(255));
        color.push_back(double(green) / double(255));
        color.push_back(double(blue) / double(255));
      }
    } break;
    default: {
      color = vector<double>(points.size(), 1.0);
    } break;
    }

    if (hasExt(output_filename, ".raw"))
      writeRaw(output_filename, points, tris);
    else if (hasExt(output_filename, ".rawn"))
      writeRaw(output_filename, points, tris, normal);
    else if (hasExt(output_filename, ".rawc"))
      writeRaw(output_filename, points, tris, vector<double>(), color);
    else if (hasExt(output_filename, ".rawnc"))
      writeRaw(output_filename, points, tris, normal, color);

    delete the_data;
    delete contour3d;
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
  } catch (...) {
    cerr << "Unknown exception!" << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
