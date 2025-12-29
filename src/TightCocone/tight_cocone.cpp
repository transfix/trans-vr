/*
  Copyright 2007-2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// -- Samrat --
// Date started: 2nd April, 2007.

/* $Id: tight_cocone.cpp 5434 2012-04-18 17:53:49Z zqyork $ */

#include <TightCocone/datastruct.h>
#include <TightCocone/init.h>
#include <TightCocone/medax.h>
#include <TightCocone/op.h>
#include <TightCocone/rcocone.h>
#include <TightCocone/robust_cc.h>
#include <TightCocone/segment.h>
#include <TightCocone/tcocone.h>
#include <TightCocone/tight_cocone.h>
#include <TightCocone/util.h>
#include <VolMagick/VolMagick.h>
#include <boost/scoped_array.hpp>
#include <iostream>

namespace TightCocone {
void segment(float, float);
void Diffuse();
void read_data(const VolMagick::Volume &vol);
void write_data(char *out_seg);
void GVF_Compute();

int XDIM;
int YDIM;
int ZDIM;

Data_3DS *dataset;
VECTOR *velocity;
unsigned char *bin_img;
float *ImgGrad;

vector<double> bounding_box;

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

// -- bounding box ---
const double BB_SCALE = 1.0;

// -----------------------------------------------------------------------
// main
// ----
// Parses the command line and initites the computation.
// -----------------------------------------------------------------------

CVCGEOM_NAMESPACE::cvcgeom_t
surfaceReconstruction(const VolMagick::Volume &vol) {

  string shape_filename;
  string output_file_prefix;

  // robust cocone parameters.
  bool b_robust = false;
  double bb_ratio = DEFAULT_BIGBALL_RATIO;
  double theta_ff = M_PI / 180.0 * DEFAULT_THETA_FF_d;
  double theta_if = M_PI / 180.0 * DEFAULT_THETA_IF_d;

  // for flatness marking (in cocone)
  double flatness_ratio = DEFAULT_RATIO;
  double cocone_phi = DEFAULT_ANGLE;
  double flat_phi = DEFAULT_FLAT;

  // For medial axis
  double theta = DEFAULT_MED_THETA;
  double medial_ratio = DEFAULT_MED_RATIO;
  int biggest_medax_comp_id = 0;

  // Check commandline options.
  bool help = false;

  // zeyuns
  float tlow = 0;
  float thigh = 1;

  dataset = (Data_3DS *)malloc(sizeof(Data_3DS));

  printf("Loading dataset...\n");
  read_data(vol);
  printf("Dataset loaded\n");

  velocity = (VECTOR *)malloc(sizeof(VECTOR) * XDIM * YDIM * ZDIM);

  printf("XIDM: %d", XDIM);

  printf("Begin GVF computation....\n");
  GVF_Compute();

  bin_img =
      (unsigned char *)malloc(sizeof(unsigned char) * XDIM * YDIM * ZDIM);

  printf("Begin Segmentation....\n");
  segment(tlow, thigh);
  printf("end segmentation .. \n");

  // for(int i=0; i<51; i++)
  // printf(".. %d", bin_img[i]);

  printf("\n end zeyuns \n");

  //  unsigned char* bin_img = new unsigned char[
  // CCVSeg(argc, argv)

#if 0
    if(argc == 1)
      help = true;
    for (int i = 1; i < argc; i++) 
      {
	if ( argc < 3 ) 
	  {
	    help = true;
	    break;
	  }
	if ( (strcmp( "-h", argv[i]) == 0) || 
	     (strcmp( "-help", argv[i]) == 0)) {
	  help = true;
	  break;
	}
	else if (strcmp("-r", argv[i]) == 0)  
	  {
	    b_robust = true;
	  }
	else if ( strcmp( "-bbr", argv[i]) == 0) {
	  ++i;
	  if ( i >= argc) 
	    {
	      cerr << "Error: option -bbr requires "
		   << "a second argument." << endl;
	      help = true;
	    } 
	  else
	    bb_ratio = atof(argv[i]) * atof(argv[i]);
	}
	else if ( strcmp( "-thif", argv[i]) == 0) {
	  ++i;
	  if ( i >= argc) 
	    {
	      cerr << "Error: option -thif requires "
		   << "a second argument." << endl;
	      help = true;
	    } 
	  else
	    theta_if = M_PI/180.*atof(argv[i]);
	}
	else if ( strcmp( "-thff", argv[i]) == 0) {
	  ++i;
	  if ( i >= argc) 
	    {
	      cerr << "Error: option -thff requires "
		   << "a second argument." << endl;
	      help = true;
	    } 
	  else
	    theta_ff = M_PI/180.*atof(argv[i]);
	}
	else if ( strcmp( "-medr", argv[i]) == 0) {
	  ++i;
	  if ( i >= argc) 
	    {
	      cerr << "Error: option -medr requires "
		   << "a second argument." << endl;
	      help = true;
	    } 
	  else
	    medial_ratio = atof(argv[i]) * atof(argv[i]);
	}
	else if ( strcmp( "-medth", argv[i]) == 0) {
	  ++i;
	  if ( i >= argc) 
	    {
	      cerr << "Error: option -medth requires "
		   << "a second argument." << endl;
	      help = true;
	    } 
	  else
	    theta = M_PI/180.*atof(argv[i]);
	}
	else if ( i+1 >= argc) {
	  help = true;
	} 
	else 
	  {
	    // strcpy(shape_filename, argv[i]); 
	    // string shape_filename(argv[i] );
	    shape_filename = shape_filename.insert(0,argv[i]);
	    // strcpy( output_file_prefix, argv[i+1]);
	    //      string output_file_prefix(argv[i+1]);
	    output_file_prefix = output_file_prefix.insert(0,argv[i+1]);
	    i++;
	  }
      }

    if ( help) {
      cerr << "Usage: " << argv[0] 
	   << " <infile> <outfile>" << endl;
      exit( 1);
    }
    cerr << endl << "Shape name : " << shape_filename << endl << endl;
#endif

  CGAL::Timer timer;
  timer.start();
  int cnt = 0;
  Triangulation triang;

#if 0
    if (b_robust)
      {
	ifstream fin;
	cout<<shape_filename.c_str();
	fin.open(shape_filename.c_str());
   
	// Build the triangulation data structure.
	cerr << "DT 1 " << flush;  
	istream_iterator<double> input( fin);
	istream_iterator<double> beyond;
	while ( input != beyond) {
	  double x = *input; 
	  ++input;
	  if ( input == beyond) {
	    cerr << "Error: inconsistent triangulation file." << endl;
	    exit( 1);
	  }
	  double y = *input; 
	  ++input;
	  if ( input == beyond) {
	    cerr << "Error: inconsistent triangulation file." << endl;
	    exit( 1);
	  }
	  double z = *input; 
	  ++input;
	  cnt++;
	  if(cnt >= 1000){
	    cerr<<"." << flush;
	    cnt = 0;
	  }

 
	  Triangulation::Point p = Point( x, y, z);
	  triang.insert( p);
	  printf("%d %d %d \n", x, y,z);

	}
	fin.close();
	cerr << ". done." << endl;
	cerr << "Time : " << timer.time() <<" sec(s)."<< endl <<endl;
	timer.reset();
   




	// Initialization
	cerr << "Init 1 ";
	initialize(triang);
	cerr << ".";
	// Computation of voronoi vertex and cell radius.
	compute_voronoi_vertex_and_cell_radius(triang);
	cerr << ". done." << endl;
	cerr << "Time : " << timer.time() << endl << endl; 
	timer.reset();
   
	cerr << "RC ";
	robust_cocone(bb_ratio, theta_ff, theta_if, triang, output_file_prefix.c_str());
	cerr << " done." << endl;
	cerr << "Time : " << timer.time() << endl << endl; 
	timer.reset();

#ifdef DEBUG_OP
	write_iobdy(triang, output_file_prefix);
#endif
	// Create a new triangulation from the pointset taken 
	// from "output_file_prefix.tcip".
	// Delete the earlier triangulation.
	triang.clear();
      }
    char robust_shape_filename[100];
    //	string robust_shape_filename;
    if (b_robust)
      strcat(strcpy(robust_shape_filename,output_file_prefix.c_str()),".tcip");
    else 
      strcpy(robust_shape_filename, shape_filename.c_str());

    ifstream robust_fin;
    robust_fin.open(robust_shape_filename);

    if(! robust_fin)
      {
	cerr << "Cannot open the file " << robust_shape_filename 
	     << " for reading. " << endl;
	exit(1);
      }

    cerr << "DT 2 ";
#endif

  // Maintain the min-max span of the pointset in 3 directions.
  double x_min = DBL_MAX, x_max = -DBL_MAX, y_min = DBL_MAX, y_max = -DBL_MAX,
         z_min = DBL_MAX, z_max = -DBL_MAX;

  // istream_iterator<double> r_input(robust_fin);
  // istream_iterator<double> r_beyond;
  int total_pt_cnt = 0;
  /*
    while ( r_input != r_beyond) {
    double x = *r_input;
    ++r_input;
    if ( r_input == r_beyond) {
    cerr << "Error: inconsistent triangulation file." << endl;
    exit( 1);
    }
    double y = *r_input;
    ++r_input;
    if ( r_input == r_beyond) {
    cerr << "Error: inconsistent triangulation file." << endl;
    exit( 1);
    }
    double z = *r_input;
    ++r_input;
    cnt++;
    total_pt_cnt ++;
    if(cnt >= 1000){
    cerr<<"." << flush;
    cnt = 0;
    }

    //      printf("%f %f %f", x, y, z);
    // cout<<"Point: "<<Point(x,y,z);
    printf("\n");
    Vertex_handle new_vh = triang.insert( Point(x,y,z) );

    // check x-span
    if(CGAL::to_double(new_vh->point().x()) < x_min)
    x_min = CGAL::to_double(new_vh->point().x());
    if(CGAL::to_double(new_vh->point().x()) > x_max)
    x_max = CGAL::to_double(new_vh->point().x());
    // check y-span
    if(CGAL::to_double(new_vh->point().y()) < y_min)
    y_min = CGAL::to_double(new_vh->point().y());
    if(CGAL::to_double(new_vh->point().y()) > y_max)
    y_max = CGAL::to_double(new_vh->point().y());
    // check z-span
    if(CGAL::to_double(new_vh->point().z()) < z_min)
    z_min = CGAL::to_double(new_vh->point().z());
    if(CGAL::to_double(new_vh->point().z()) > z_max)
    z_max = CGAL::to_double(new_vh->point().z());
    }
  */

  for (int k = 0; k < ZDIM; k++) {
    for (int j = 0; j < YDIM; j++)
      for (int i = 0; i < XDIM; i++) {

        if (bin_img[IndexVect(i, j, k)] == 0) {

          Vertex_handle new_vh =
              triang.insert(Point(float(i) * vol.XSpan() + vol.XMin(),
                                  float(j) * vol.YSpan() + vol.YMin(),
                                  float(k) * vol.ZSpan() + vol.ZMin()));

          total_pt_cnt++;

          // check x-span
          if (CGAL::to_double(new_vh->point().x()) < x_min)
            x_min = CGAL::to_double(new_vh->point().x());
          if (CGAL::to_double(new_vh->point().x()) > x_max)
            x_max = CGAL::to_double(new_vh->point().x());
          // check y-span
          if (CGAL::to_double(new_vh->point().y()) < y_min)
            y_min = CGAL::to_double(new_vh->point().y());
          if (CGAL::to_double(new_vh->point().y()) > y_max)
            y_max = CGAL::to_double(new_vh->point().y());
          // check z-span
          if (CGAL::to_double(new_vh->point().z()) < z_min)
            z_min = CGAL::to_double(new_vh->point().z());
          if (CGAL::to_double(new_vh->point().z()) > z_max)
            z_max = CGAL::to_double(new_vh->point().z());
        }
      }

    fprintf(stderr, "%5.2f %%\r",
            (((float)k) / ((float)((int)(ZDIM - 1)))) * 100.0);
  }
  fprintf(stderr, "\n");

  cerr << " done." << endl;
  cerr << "Total point count: " << total_pt_cnt << endl;
  cerr << "Del Time : " << timer.time() << endl << endl;
  timer.reset();
  // robust_fin.close();
  //  Bounding box of the point set.
  bounding_box.push_back(x_min - BB_SCALE * (x_max - x_min));
  bounding_box.push_back(x_max + BB_SCALE * (x_max - x_min));

  bounding_box.push_back(y_min - BB_SCALE * (y_max - y_min));
  bounding_box.push_back(y_max + BB_SCALE * (y_max - y_min));

  bounding_box.push_back(z_min - BB_SCALE * (z_max - z_min));
  bounding_box.push_back(z_max + BB_SCALE * (z_max - z_min));

  // --- Init ----
  cerr << "Init 2 ";
  initialize(triang);
  cerr << ".";
  // compute voronoi vertex
  compute_voronoi_vertex_and_cell_radius(triang);
  cerr << ". done." << endl;
  cerr << "Time : " << timer.time() << endl << endl;
  timer.reset();

  // ---- Reconstruction -----
  cerr << "TC ";
  tcocone(cocone_phi, DEFAULT_SHARP, flat_phi, flatness_ratio, triang);
  cerr << " done." << endl;
  cerr << "Time : " << timer.time() << endl << endl;
  timer.reset();
  // write_wt(triang, output_file_prefix.c_str());

  CVCGEOM_NAMESPACE::cvcgeom_t result = wt_to_geometry(triang);

#if 0
    // Medial Axis
    timer.reset();
    cerr << "Medial axis " << flush;
#ifdef NO_TIGHTCOCONE
    compute_poles( triang );
    mark_flat_vertices( triang, flatness_ratio, cocone_phi, flat_phi);
#endif
    compute_medial_axis(triang, theta, medial_ratio, biggest_medax_comp_id);
    cerr << " done." << endl;
    cerr << "TIME: "<<timer.time()<<" sec(s)." << endl << endl;  
    timer.reset();
    timer.stop();

    // writing the medial axis in OFF format.
    write_axis(triang, biggest_medax_comp_id, output_file_prefix.c_str());
#endif

#if 0
    //find new bounding box if it changed
    if(result->m_NumTriVerts > 1)
      {
	x_min = x_max = result->m_TriVerts[0];
	y_min = y_max = result->m_TriVerts[1];
	z_min = z_max = result->m_TriVerts[2];
      }
    else return result;

    for(int i = 0; i < result->m_NumTriVerts; i++)
      {
	if(result->m_TriVerts[3*i+0] > x_max) x_max = result->m_TriVerts[3*i+0];
	if(result->m_TriVerts[3*i+0] < x_min) x_min = result->m_TriVerts[3*i+0];
	if(result->m_TriVerts[3*i+1] > y_max) y_max = result->m_TriVerts[3*i+1];
	if(result->m_TriVerts[3*i+1] < y_min) y_min = result->m_TriVerts[3*i+1];
	if(result->m_TriVerts[3*i+2] > z_max) z_max = result->m_TriVerts[3*i+2];
	if(result->m_TriVerts[3*i+2] < z_min) z_min = result->m_TriVerts[3*i+2];
      }

    //scale to original volume's bounding box
    for(int i = 0; i < result->m_NumTriVerts; i++)
      {
	result->m_TriVerts[3*i+0] = vol.XMin()+((result->m_TriVerts[3*i+0] - x_min)/(x_max - x_min))*(vol.XMax()-vol.XMin());
	result->m_TriVerts[3*i+1] = vol.YMin()+((result->m_TriVerts[3*i+1] - y_min)/(y_max - y_min))*(vol.YMax()-vol.YMin());
	result->m_TriVerts[3*i+2] = vol.ZMin()+((result->m_TriVerts[3*i+2] - z_min)/(z_max - z_min))*(vol.ZMax()-vol.ZMin());
      }
#endif

  return result;
}

CVCGEOM_NAMESPACE::cvcgeom_t
generateBoundaryPointCloud(const VolMagick::Volume &vol, float tlow,
                           float thigh) {
  // temp comment

  // dataset=(Data_3DS*)malloc(sizeof(Data_3DS));
  Data_3DS dataset_local;
  dataset = &dataset_local; // this is safe because dataset will not be used
                            // outside of this function call

  fprintf(stderr, "Loading dataset...\n");
  read_data(vol);
  fprintf(stderr, "Dataset loaded\n");

  // velocity = (VECTOR*)malloc(sizeof(VECTOR)*XDIM*YDIM*ZDIM);
  boost::scoped_array<VECTOR> velocity_local(new VECTOR[XDIM * YDIM * ZDIM]);
  velocity = velocity_local.get();

  fprintf(stderr, "XDIM: %d\n", XDIM);
  fprintf(stderr, "YDIM: %d\n", YDIM);
  fprintf(stderr, "ZDIM: %d\n", ZDIM);

  fprintf(stderr, "Begin GVF computation....\n");
  GVF_Compute();

  // bin_img = (unsigned char*)malloc(sizeof(unsigned char)*XDIM*YDIM*ZDIM);
  boost::scoped_array<unsigned char> bin_img_local(
      new unsigned char[XDIM * YDIM * ZDIM]);
  bin_img = bin_img_local.get();

  fprintf(stderr, "Begin Segmentation....\n");
  segment(tlow, thigh);
  fprintf(stderr, "end segmentation .. \n");

  std::vector<int> boundary_indices;

  for (int k = 0; k < ZDIM; k++)
    for (int j = 0; j < YDIM; j++)
      for (int i = 0; i < XDIM; i++) {
        if (bin_img[IndexVect(i, j, k)] == 0) {
          boundary_indices.push_back(i);
          boundary_indices.push_back(j);
          boundary_indices.push_back(k);
        }
      }

  CVCGEOM_NAMESPACE::cvcgeom_t result;
  //    result->AllocatePoints(boundary_indices.size()/3);
  for (unsigned int i = 0; i < boundary_indices.size() / 3; i++) {
    CVCGEOM_NAMESPACE::cvcgeom_t::point_t newVertex;

    newVertex[0] = boundary_indices[i * 3 + 0] * vol.XSpan() + vol.XMin();
    newVertex[1] = boundary_indices[i * 3 + 1] * vol.YSpan() + vol.YMin();
    newVertex[2] = boundary_indices[i * 3 + 2] * vol.ZSpan() + vol.ZMin();
    result.points().push_back(newVertex);
  }

  return result;
}

CVCGEOM_NAMESPACE::cvcgeom_t
surfaceReconstruction(const CVCGEOM_NAMESPACE::cvcgeom_t &pointCloud,
                      const Parameters &params) {
  // string shape_filename;
  // string output_file_prefix;

  // robust cocone parameters.
  bool b_robust = params.b_robust();   // false;
  double bb_ratio = params.bb_ratio(); // DEFAULT_BIGBALL_RATIO;
  double theta_ff = params.theta_ff(); // M_PI/180.0*DEFAULT_THETA_FF_d;
  double theta_if = params.theta_if(); // M_PI/180.0*DEFAULT_THETA_IF_d;

  // for flatness marking (in cocone)
  double flatness_ratio = params.flatness_ratio(); // DEFAULT_RATIO;
  double cocone_phi = params.cocone_phi();         // DEFAULT_ANGLE;
  double flat_phi = params.flat_phi();             // DEFAULT_FLAT;

  // For medial axis
  // double theta = DEFAULT_MED_THETA;
  // double medial_ratio = DEFAULT_MED_RATIO;
  // int biggest_medax_comp_id = 0;

  CGAL::Timer timer;
  timer.start();
  int cnt = 0;
  Triangulation triang;

  std::vector<Point> robust_points;

  cerr << "Delaunay Triangulation ";

  // Maintain the min-max span of the pointset in 3 directions.
  double x_min = DBL_MAX, x_max = -DBL_MAX, y_min = DBL_MAX, y_max = -DBL_MAX,
         z_min = DBL_MAX, z_max = -DBL_MAX;

  int numvert = pointCloud.points().size();
  for (int i = 0; i < numvert; i++) {
    Triangulation::Point p =
        Point(pointCloud.points()[i][0], pointCloud.points()[i][1],
              pointCloud.points()[i][2]);
    triang.insert(p);

    if (CGAL::to_double(p.x()) < x_min)
      x_min = CGAL::to_double(p.x());
    if (CGAL::to_double(p.x()) > x_max)
      x_max = CGAL::to_double(p.x());
    // check y-span
    if (CGAL::to_double(p.y()) < y_min)
      y_min = CGAL::to_double(p.y());
    if (CGAL::to_double(p.y()) > y_max)
      y_max = CGAL::to_double(p.y());
    // check z-span
    if (CGAL::to_double(p.z()) < z_min)
      z_min = CGAL::to_double(p.z());
    if (CGAL::to_double(p.z()) > z_max)
      z_max = CGAL::to_double(p.z());

    if (i % 1000 == 0)
      cerr << "." << flush;
  }
  cerr << "done" << endl;
  cerr << "Time: " << timer.time() << endl;
  timer.reset();

  // #if 0
  if (b_robust) {

    // Initialization
    cerr << "Init 1 ";
    initialize(triang);
    cerr << ".";
    // Computation of voronoi vertex and cell radius.
    compute_voronoi_vertex_and_cell_radius(triang);
    cerr << ". done." << endl;
    cerr << "Time : " << timer.time() << endl << endl;
    timer.reset();

    cerr << "Robust Cocone ";
    robust_points = robust_cocone(bb_ratio, theta_ff, theta_if, triang);
    cerr << " done." << endl;
    cerr << "Time : " << timer.time() << endl << endl;
    timer.reset();
  }

  int total_pt_cnt = 0;

  if (robust_points.size() > 0) // if we used robust cocone, use those points
                                // instead of the original points
  {
    cerr << "DT 2 ";

    triang.clear();
    for (std::vector<Point>::iterator i = robust_points.begin();
         i != robust_points.end(); i++) {
      Vertex_handle new_vh = triang.insert(*i);
      total_pt_cnt++;

      // check x-span
      if (CGAL::to_double(new_vh->point().x()) < x_min)
        x_min = CGAL::to_double(new_vh->point().x());
      if (CGAL::to_double(new_vh->point().x()) > x_max)
        x_max = CGAL::to_double(new_vh->point().x());
      // check y-span
      if (CGAL::to_double(new_vh->point().y()) < y_min)
        y_min = CGAL::to_double(new_vh->point().y());
      if (CGAL::to_double(new_vh->point().y()) > y_max)
        y_max = CGAL::to_double(new_vh->point().y());
      // check z-span
      if (CGAL::to_double(new_vh->point().z()) < z_min)
        z_min = CGAL::to_double(new_vh->point().z());
      if (CGAL::to_double(new_vh->point().z()) > z_max)
        z_max = CGAL::to_double(new_vh->point().z());

      if (std::distance(robust_points.begin(), i) %
          (pointCloud.points().size() - 1) / 100)
        fprintf(stderr, "%5.2f %%\r",
                float(std::distance(robust_points.begin(), i)) /
                    (float(robust_points.size() - 1) * 100.0));
    }
    fprintf(stderr, "\n");

    cerr << " done." << endl;
    cerr << "Total point count: " << total_pt_cnt << endl;
    cerr << "Del Time : " << timer.time() << endl << endl;
    timer.reset();
  }

  // robust_fin.close();
  //  Bounding box of the point set.
  bounding_box.push_back(x_min - BB_SCALE * (x_max - x_min));
  bounding_box.push_back(x_max + BB_SCALE * (x_max - x_min));

  bounding_box.push_back(y_min - BB_SCALE * (y_max - y_min));
  bounding_box.push_back(y_max + BB_SCALE * (y_max - y_min));

  bounding_box.push_back(z_min - BB_SCALE * (z_max - z_min));
  bounding_box.push_back(z_max + BB_SCALE * (z_max - z_min));

  // --- Init ----
  cerr << "Init 2 ";
  initialize(triang);
  cerr << ".";
  // compute voronoi vertex
  compute_voronoi_vertex_and_cell_radius(triang);
  cerr << ". done." << endl;
  cerr << "Time : " << timer.time() << endl << endl;
  timer.reset();

  // ---- Reconstruction -----
  cerr << "TC ";
  tcocone(cocone_phi, DEFAULT_SHARP, flat_phi, flatness_ratio, triang);
  cerr << " done." << endl;
  cerr << "Time : " << timer.time() << endl << endl;
  timer.reset();
  //   write_wt(triang, "temp.off");

  return wt_to_geometry(triang);
}
}; // namespace TightCocone
