/*
  Copyright 2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/*
  This program is a simple program that outputs a transfer function
  ramp in the format that the old CVC raycaster likes using an input
  *.vinay transfer function file used by VolRover and TexMol and the like.

  Useless for most people except me!!! -Joe R. 01/22/2008
*/

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <set>
#include <iterator>

// using GSL library for interpolation
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>

using namespace std;

typedef boost::tuple<
  double, /* opacity position */
  double  /* opacity value */
  > opacity_node;

/* 
   note, in the following data structure, RGB values 
   are in the set [0.0-1.0] 
*/
typedef boost::tuple<
  double, /* color position */
  double, /* red value */
  double, /* green value */
  double  /* blue value */
  > color_node;

typedef boost::tuple<
  vector<opacity_node>,
  vector<color_node>
  > trans_func;

typedef boost::tuple<
  int,  /* ramp position */
  int,  /* red [0-255] */
  int,  /* green [0-255] */
  int,  /* blue [0-255] */
  int   /* alpha [0-255] */
  > color_ramp_node;

typedef vector<color_ramp_node> color_ramp;

trans_func read_transfer_function(const string& filename)
{
  trans_func tf;

  ifstream inf(filename.c_str());
  if(!inf)
    throw runtime_error(string("Could not open ") + filename);

  string line;
  getline(inf, line);
  if(!inf || line != "Anthony and Vinay are Great.")
    throw runtime_error("Not a proper vinay file!");
  getline(inf, line);
  if(!inf || line != "Alphamap")
    throw runtime_error("Not a proper vinay file!");
  getline(inf, line);
  if(!inf || line != "Number of nodes")
    throw runtime_error("Not a proper vinay file!");

  int num_alpha;
  inf >> num_alpha;
  if(!inf)
    throw runtime_error("Could not read number of alpha nodes!");
  getline(inf, line); //extract the \n

  if(num_alpha < 1)
    throw runtime_error("No alpha nodes!");

  getline(inf, line);
  if(!inf || line != "Position and opacity")
    throw runtime_error("Not a proper vinay file!");

  for(int i=0; i<num_alpha; i++)
    {
      double pos, op;

      inf >> pos >> op;
      if(!inf)
	throw runtime_error("Could not read position and opacity!");
      getline(inf, line); //extract the \n

      tf.get<0>().push_back(opacity_node(pos,op));
    }
  
  getline(inf, line);
  if(!inf || line != "ColorMap")
    throw runtime_error("Not a proper vinay file!");
  getline(inf, line);
  if(!inf || line != "Number of nodes")
    throw runtime_error("Not a proper vinay file!");

  int num_color;
  inf >> num_color;
  if(!inf)
    throw runtime_error("Could not read number of color nodes!");
  getline(inf, line); //extract the \n

  if(num_color < 1)
    throw runtime_error("No color nodes!");

  getline(inf, line);
  if(!inf || line != "Position and RGB")
    throw runtime_error("Not a proper vinay file!");

  for(int i=0; i<num_color; i++)
    {
      double pos, r, g, b;

      inf >> pos >> r >> g >> b;
      if(!inf)
	throw runtime_error("Could not read position and RGB!");
      getline(inf, line); //extract the \n

      tf.get<1>().push_back(color_node(pos,r,g,b));
    }

  //sort the nodes such that they're in ascending position order
  sort(tf.get<0>().begin(), tf.get<0>().end());
  sort(tf.get<1>().begin(), tf.get<1>().end());

  //ignore the rest of the file!

  return tf;
}

int main(int argc, char **argv)
{
  trans_func tf;
  color_ramp cr;

  map<string,int> vox_type_max;
  vox_type_max["uchar"] = 255;
  vox_type_max["ushort"] = 65535;
  vox_type_max["float"] = 4096; //the raycaster maps float valued volumes to [0-4096]

  if(argc != 3 || vox_type_max.find(argv[2]) == vox_type_max.end())
    {
      cout << "Usage: " << argv[0] << " <vinay file> <voxel type>" << endl;
      cout << "<voxel type> - one of 'uchar' 'ushort' or 'float'" << endl;
      return EXIT_FAILURE;
    }

  tf = read_transfer_function(argv[1]);
  
  {
    const gsl_interp_type *interp = gsl_interp_linear; //interpolation type
    gsl_interp_accel *acc_op, *acc_col_r, *acc_col_g, *acc_col_b;
    gsl_spline *spline_op, *spline_col_r, *spline_col_g, *spline_col_b;

    vector<double> op_pos, op_val;
    vector<double> col_pos, col_r_val, col_g_val, col_b_val;
    

    /* copy our trans_func object into the above vectors */
    for(vector<opacity_node>::iterator i = tf.get<0>().begin();
	i != tf.get<0>().end();
	i++)
      {
	op_pos.push_back(i->get<0>());
	op_val.push_back(i->get<1>());
      }

    for(vector<color_node>::iterator i = tf.get<1>().begin();
	i != tf.get<1>().end();
	i++)
      {
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
    gsl_spline_init(spline_col_r, &(col_pos[0]), &(col_r_val[0]), col_pos.size());
    gsl_spline_init(spline_col_g, &(col_pos[0]), &(col_g_val[0]), col_pos.size());
    gsl_spline_init(spline_col_b, &(col_pos[0]), &(col_b_val[0]), col_pos.size());

    //merge opacity node and color node positions and output an RGBA value for each position
    //suitable for the raycaster's cnf files... 
    /*
    vector<double> comb_pos(op_pos.size() + col_pos.size());
    merge(op_pos.begin(), op_pos.end(),
	  col_pos.begin(), col_pos.end(),
	  comb_pos.begin());
    */
    set<double> comb_pos;
    comb_pos.insert(op_pos.begin(), op_pos.end());
    comb_pos.insert(col_pos.begin(), col_pos.end());
    
    //now sample at each position
    for(//vector<double>::iterator i = comb_pos.begin();
	set<double>::iterator i = comb_pos.begin();
	i != comb_pos.end();
	i++)
      cr.push_back(color_ramp_node(int((*i)*vox_type_max[argv[2]]),
				   int(gsl_spline_eval(spline_col_r, *i, acc_col_r)*255.0),
				   int(gsl_spline_eval(spline_col_g, *i, acc_col_g)*255.0),
				   int(gsl_spline_eval(spline_col_b, *i, acc_col_b)*255.0),
				   int(gsl_spline_eval(spline_op, *i, acc_op)*255.0)));
    
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

  //finally, output the color ramp
  copy(cr.begin(), cr.end(),
       ostream_iterator<color_ramp_node>(cout,"\n"));

  return EXIT_SUCCESS;
}
