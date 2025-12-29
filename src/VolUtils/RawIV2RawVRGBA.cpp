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

/* $Id: RawIV2RawVRGBA.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// using GSL library for interpolation
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

namespace po = boost::program_options;

typedef boost::tuple<double, /* opacity position */
                     double  /* opacity value */
                     >
    opacity_node;

/*
   note, in the following data structure, RGB values
   are in the set [0.0-1.0]
*/
typedef boost::tuple<double, /* color position */
                     double, /* red value */
                     double, /* green value */
                     double  /* blue value */
                     >
    color_node;

typedef boost::tuple<std::vector<opacity_node>, std::vector<color_node>>
    trans_func;

static inline trans_func read_transfer_function(const std::string &filename) {
  trans_func tf;

  std::ifstream inf(filename.c_str());
  if (!inf)
    throw std::runtime_error(std::string("Could not open ") + filename);

  std::string line;
  getline(inf, line);
  if (!inf || line != "Anthony and Vinay are Great.")
    throw std::runtime_error("Not a proper vinay file!");
  getline(inf, line);
  if (!inf || line != "Alphamap")
    throw std::runtime_error("Not a proper vinay file!");
  getline(inf, line);
  if (!inf || line != "Number of nodes")
    throw std::runtime_error("Not a proper vinay file!");

  int num_alpha;
  inf >> num_alpha;
  if (!inf)
    throw std::runtime_error("Could not read number of alpha nodes!");
  getline(inf, line); // extract the \n

  if (num_alpha < 1)
    throw std::runtime_error("No alpha nodes!");

  getline(inf, line);
  if (!inf || line != "Position and opacity")
    throw std::runtime_error("Not a proper vinay file!");

  for (int i = 0; i < num_alpha; i++) {
    double pos, op;

    inf >> pos >> op;
    if (!inf)
      throw std::runtime_error("Could not read position and opacity!");
    getline(inf, line); // extract the \n

    tf.get<0>().push_back(opacity_node(pos, op));
  }

  getline(inf, line);
  if (!inf || line != "ColorMap")
    throw std::runtime_error("Not a proper vinay file!");
  getline(inf, line);
  if (!inf || line != "Number of nodes")
    throw std::runtime_error("Not a proper vinay file!");

  int num_color;
  inf >> num_color;
  if (!inf)
    throw std::runtime_error("Could not read number of color nodes!");
  getline(inf, line); // extract the \n

  if (num_color < 1)
    throw std::runtime_error("No color nodes!");

  getline(inf, line);
  if (!inf || line != "Position and RGB")
    throw std::runtime_error("Not a proper vinay file!");

  for (int i = 0; i < num_color; i++) {
    double pos, r, g, b;

    inf >> pos >> r >> g >> b;
    if (!inf)
      throw std::runtime_error("Could not read position and RGB!");
    getline(inf, line); // extract the \n

    tf.get<1>().push_back(color_node(pos, r, g, b));
  }

  // sort the nodes such that they're in ascending position order
  std::sort(tf.get<0>().begin(), tf.get<0>().end());
  std::sort(tf.get<1>().begin(), tf.get<1>().end());

  // ignore the rest of the file!

  return tf;
}

using namespace std;
using namespace boost;

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
  /*
  if(argc != 6)
    {
      cerr << "Usage: " << argv[0] << " <input volume file> <output volume
  file> <target X dim> <target Y dim> <target Z dim>" << endl; return 1;
    }
  */

  try {
    VolMagickOpStatus status;
    VolMagick::setDefaultMessenger(&status);

    unsigned int var, time;
    string input_filename, output_filename, function_filename;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "input,i", po::value<string>(&input_filename),
        "input file (volume: rawiv, rawv, mrc, pif, etc...)")(
        "var,v", po::value<unsigned int>(&var)->default_value(0),
        "variable index of volume variable to create output from")(
        "time,t", po::value<unsigned int>(&time)->default_value(0),
        "timestep index of volume variable to create output from")(
        "output,o", po::value<string>(&output_filename),
        "output multi-variable volume file (typically rawv format)")(
        "trans,f", po::value<string>(&function_filename),
        "transfer function to apply to the input file to produce the output "
        "(.vinay format)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cerr << desc << endl;
      return EXIT_FAILURE;
    } else if (!vm.count("input") || !vm.count("output") ||
               !vm.count("trans")) {
      cerr << "Error: Input, output, and transfer function files must be "
              "specified!"
           << endl
           << endl;
      cerr << desc << endl;
      return EXIT_FAILURE;
    }

    vector<VolMagick::Volume> rgba_vols;
    VolMagick::Volume vol;
    VolMagick::readVolumeFile(vol, input_filename, var, time);
    rgba_vols = vector<VolMagick::Volume>(
        4, VolMagick::Volume(vol.dimension(), VolMagick::UChar,
                             vol.boundingBox()));
    rgba_vols[3] = vol;

    trans_func tf = read_transfer_function(function_filename);

    vector<VolMagick::VoxelType> voxtypes(4, VolMagick::UChar);
    voxtypes[3] = vol.voxelType();
    VolMagick::createVolumeFile(output_filename, vol.boundingBox(),
                                vol.dimension(), voxtypes, 4, 1);

    rgba_vols[0].desc("red");
    rgba_vols[1].desc("green");
    rgba_vols[2].desc("blue");
    rgba_vols[3].desc("alpha");

    {
      const gsl_interp_type *interp = gsl_interp_linear; // interpolation type
      gsl_interp_accel *acc_op, *acc_col_r, *acc_col_g, *acc_col_b;
      gsl_spline *spline_op, *spline_col_r, *spline_col_g, *spline_col_b;

      vector<double> op_pos, op_val;
      vector<double> col_pos, col_r_val, col_g_val, col_b_val;

      /* copy our trans_func object into the above vectors */
      for (vector<opacity_node>::iterator i = tf.get<0>().begin();
           i != tf.get<0>().end(); i++) {
        op_pos.push_back(i->get<0>());
        op_val.push_back(i->get<1>());
      }

      for (vector<color_node>::iterator i = tf.get<1>().begin();
           i != tf.get<1>().end(); i++) {
        col_pos.push_back(i->get<0>());
        col_r_val.push_back(i->get<1>());
        col_g_val.push_back(i->get<2>());
        col_b_val.push_back(i->get<3>());
      }

      acc_op = gsl_interp_accel_alloc();
      acc_col_r = gsl_interp_accel_alloc();
      acc_col_g = gsl_interp_accel_alloc();
      acc_col_b = gsl_interp_accel_alloc();
      spline_op = gsl_spline_alloc(interp, op_pos.size());
      spline_col_r = gsl_spline_alloc(interp, col_pos.size());
      spline_col_g = gsl_spline_alloc(interp, col_pos.size());
      spline_col_b = gsl_spline_alloc(interp, col_pos.size());

      gsl_spline_init(spline_op, &(op_pos[0]), &(op_val[0]), op_pos.size());
      gsl_spline_init(spline_col_r, &(col_pos[0]), &(col_r_val[0]),
                      col_pos.size());
      gsl_spline_init(spline_col_g, &(col_pos[0]), &(col_g_val[0]),
                      col_pos.size());
      gsl_spline_init(spline_col_b, &(col_pos[0]), &(col_b_val[0]),
                      col_pos.size());

      for (unsigned int k = 0; k < vol.ZDim(); k++) {
        for (unsigned int j = 0; j < vol.YDim(); j++)
          for (unsigned int i = 0; i < vol.XDim(); i++) {
            rgba_vols[0](i, j, k,
                         int(gsl_spline_eval(spline_col_r,
                                             (vol(i, j, k) - vol.min()) /
                                                 (vol.max() - vol.min()),
                                             acc_col_r) *
                             255.0));
            rgba_vols[1](i, j, k,
                         int(gsl_spline_eval(spline_col_g,
                                             (vol(i, j, k) - vol.min()) /
                                                 (vol.max() - vol.min()),
                                             acc_col_g) *
                             255.0));
            rgba_vols[2](i, j, k,
                         int(gsl_spline_eval(spline_col_b,
                                             (vol(i, j, k) - vol.min()) /
                                                 (vol.max() - vol.min()),
                                             acc_col_b) *
                             255.0));
            rgba_vols[3](i, j, k,
                         gsl_spline_eval(spline_op,
                                         (vol(i, j, k) - vol.min()) /
                                             (vol.max() - vol.min()),
                                         acc_op) *
                                 (vol.max() - vol.min()) +
                             vol.min());
          }
        fprintf(stderr, "Generating RGBA volume: %5.2f %%\r",
                (float(k + 1) / float(vol.ZDim())) * 100.0);
      }
      fprintf(stderr, "\n");

      /* clean up our GSL mess */
      gsl_spline_free(spline_op);
      gsl_spline_free(spline_col_r);
      gsl_spline_free(spline_col_g);
      gsl_spline_free(spline_col_b);
      gsl_interp_accel_free(acc_op);
      gsl_interp_accel_free(acc_col_r);
      gsl_interp_accel_free(acc_col_g);
      gsl_interp_accel_free(acc_col_b);
    }

    VolMagick::writeVolumeFile(rgba_vols, output_filename);
  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}
