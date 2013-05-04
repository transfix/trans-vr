/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Curation.

  Curation is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Curation is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: main.cpp 4741 2011-10-21 21:22:06Z transfix $ */

#include <Curation/datastruct.h>
#include <Curation/handle.h>
#include <Curation/curat.h>
#include <Curation/mds.h>
#include <Curation/mesh_io.h>
#include <Curation/am.h>
#include <Curation/tcocone.h>
//#include "rcocone.h"
#include <Curation/util.h>
#include <Curation/op.h>
#include <Curation/init.h>
#include <Curation/smax.h>
#include <Curation/medax.h>

#include <Curation/Curation.h>

namespace Curation
{
  std::vector<double> bounding_box;


  // -----------------------------------------------------------------------
  // Constants
  // -----------------------------------------------------------------------

  // ------- cocone -----


  const double DEFAULT_ANGLE = M_PI / 8.0;      // Half of the co-cone angle. 
  const double DEFAULT_SHARP = 2 * M_PI / 3.0;  // Angle of sharp edges.
  const double DEFAULT_RATIO = 1.2 * 1.2;       // Squared thinness factor. 
  const double DEFAULT_FLAT  = M_PI / 3.0;      // Angle for flatness Test

  // -- medial axis ---

  //const double DEFAULT_MED_THETA = M_PI*22.5/180.0; // original: M_PI*22.5/180.0;
  //const double DEFAULT_MED_RATIO = 8.0*8.0; // original: 8.0*8.0;


  

  // -- robust cocone ---

  //const double DEFAULT_BIGBALL_RATIO  = 4.*4.;  // parameter to choose big balls.
  //const double DEFAULT_THETA_IF_d  = 5.0;       // parameter for infinite-finite deep intersection.
  //const double DEFAULT_THETA_FF_d  = 10.0;      // parameter for finite-finite deep intersection.
  //const double DEFAULT_THETA_II_d  = 30.0;      // parameter for infinite-infinite deep intersection.

//  std::vector<boost::shared_ptr<Geometry> > curate(const boost::shared_ptr<Geometry>& geom,
 // std::vector<CVCGEOM_NAMESPACE::cvcgeom_t> curate(const CVCGEOM_NAMESPACE::cvcgeom_t & geom,
  CVCGEOM_NAMESPACE::cvcgeom_t curate(const CVCGEOM_NAMESPACE::cvcgeom_t & geom,
						   float mr,
						   int keep_pockets_count,
						   int keep_tunnels_count)
  {
//    boost::shared_ptr<Geometry> result;
//	CVCGEOM_NAMESPACE::cvcgeom_t result;
  
  	map<int, cell_cluster> cluster_set;

    // segmentation parameters.
    //float mr = DEFAULT_MERGE_RATIO;
    // parameter to see the K biggest segments.
    //int output_seg_count = DEFAULT_OUTPUT_SEG_COUNT;
    
    // segmentation and matching filenames.
    //char shape_filename[10000];
    //char op_prefix[90];
    
    // Check commandline options.
    //bool help = false;
    // To output the copyright information etc.
    //bool long_banner = false;
    
    bool flip = false;
    //bool read_color_opacity = false;
    
    CGAL::Timer timer;
    timer.start();
    // read mesh.
    Mesh mesh;
    cerr << "Reading mesh ";
    read_mesh_from_geom(mesh, geom);
    cerr << "done." << endl;
    // align mesh.
    cerr << "Alignment ";
    am(mesh);
    cerr << "done." << endl;
    cerr << "Vertex normals ";
    compute_mesh_vertex_normal(mesh, flip);
    cerr << "computed." << endl;
    //write_mesh(mesh, op_prefix);

    // Compute Delaunay of the vertex set.
    Triangulation triang;
    cerr << "Triangulation ";
    double x_min = DBL_MAX, y_min = DBL_MAX, z_min = DBL_MAX,
      x_max = -DBL_MAX, y_max = -DBL_MAX, z_max = -DBL_MAX;
    for(int i = 0; i < mesh.get_nv(); i ++) 
      {
	Point p = mesh.vert_list[i].point();
	double x = p[0], y = p[1], z = p[2];
	if( x < x_min ) x_min = x;
	if( x > x_max ) x_max = x;
	if( y < y_min ) y_min = y;
	if( y > y_max ) y_max = y;
	if( z < z_min ) z_min = z;
	if( z > z_max ) z_max = z;
	Vertex_handle vh = triang.insert(p);
	if(i%1000 == 0) cerr << ".";
      }
    cerr << "done." << endl;
    bounding_box.push_back(x_min);
    bounding_box.push_back(x_max);
    bounding_box.push_back(y_min);
    bounding_box.push_back(y_max);
    bounding_box.push_back(z_min);
    bounding_box.push_back(z_max);

  //  cout<<"triangle vertices number= " << triang.number_of_vertices() << endl;
    // Initialization of Triangulation datastructure.
    cerr << "Init ";
    initialize(triang);
    cerr << ".";
    // compute voronoi vertex
    compute_voronoi_vertex_and_cell_radius(triang);
    cerr << ". done." << endl;
	cerr << "Time : " << timer.time() << endl;
    timer.reset();


    // surface reconstruction using Tight Cocone
    cerr << "Surface Reconstruction ";
    tcocone(DEFAULT_ANGLE, DEFAULT_SHARP, DEFAULT_FLAT, DEFAULT_RATIO, triang);
    cerr << " done." << endl;
    cerr << "Reconstruction took "<< timer.time() << " seconds." << endl;


    cerr << "Computing S_MAX ";
    cerr << " mr = " << mr << " ";
   // vector<int> sorted_smax_index_vector = compute_smax(triang, mesh, cluster_set, mr);
    vector<int> sorted_smax_index_vector = compute_smax(triang,  cluster_set, mr);

    cerr << " done." << endl;


	// detect pocket, tunnel, void.
   cerr << "Detecting Handles ";
   detect_handle(triang, cluster_set);
   cerr << " done." << endl;
   cerr << "Handle detection took " << timer.time() << " seconds." << endl;
   timer.reset();

 //  cerr << "Writing handles ";
  // write the outside pockets of the dataset
 //  write_handle(triang, cluster_set, sorted_smax_index_vector, 
   //            output_tunnel_count, output_pocket_count, op_prefix.c_str());
  // cerr << " done." << endl;
 //  cerr << "Handle detection took " << timer.time() << " seconds." << endl;

  // build curated surface from the triangulation.
//   timer.reset();
   cerr << "Curation ";
   curate_tr(triang, cluster_set, sorted_smax_index_vector, keep_pockets_count, keep_tunnels_count);
   cerr << " done." << endl;
   cerr << "Delaunay based Curation took " << timer.time() << " seconds." << endl;

   cerr << "Writing curated surfaces ";


 //   cerr << "Number of segments " << (int)sorted_smax_index_vector.size() << endl;

    timer.stop();

    // ---------- End of Segmentation ------------

    // write the outside pockets of the dataset
    return write_smax_to_geom(triang, cluster_set, sorted_smax_index_vector); //,
		//		  keep_pockets_count, 
		//	      keep_tunnels_count);
  }
}

#if 0
// -----------------------------------------------------------------------
// main
// ----
// Parses commandline arguments and calls cocone and writes the output.
// -----------------------------------------------------------------------

int 
main( int argc, char **argv) {

  using namespace Curation;

  map<int, cell_cluster> cluster_set;

  // segmentation parameters.
  float mr = DEFAULT_MERGE_RATIO;
  // parameter to see the K biggest segments.
  int output_seg_count = DEFAULT_OUTPUT_SEG_COUNT;
  
  // segmentation and matching filenames.
  char shape_filename[10000];
  char op_prefix[90];

  // Check commandline options.
  bool help = false;
  // To output the copyright information etc.
  bool long_banner = false;

  bool flip = false;
  bool read_color_opacity = false;

  if(argc == 1)
    help = true;
  for (int i = 1; i < argc; i++) {
    if ( argc < 3 ) {
      help = true;
      break;
    }
    
    if ( (strcmp( "-h", argv[i]) == 0) || 
	 (strcmp( "-help", argv[i]) == 0)) {
      help = true;
      break;
    }
    else if ( strcmp( "-L", argv[i]) == 0) {
      long_banner = true;
    }
    else if ( strcmp( "-mr", argv[i]) == 0) {
      ++i;
      if ( i >= argc) 
	{
	  cerr << "Error: option -mr requires "
	       << "a second argument." << endl;
	  help = true;
	} 
      else
	mr = atof( argv[i]) * atof( argv[i]);
    }
    else if ( strcmp( "-opc", argv[i]) == 0) {
      ++i;
      if ( i >= argc) 
	{
	  cerr << "Error: option -opc requires "
	       << "a second argument." << endl;
	  help = true;
	} 
      else
	output_seg_count = atoi(argv[i]);
    }
    else if(strcmp("-ca", argv[i]) == 0)
      {
	read_color_opacity = true;
      }
    else if(strcmp("-f", argv[i]) == 0 ||
            strcmp("-F", argv[i]) == 0)
      {
	flip = true;
      }

    else if ( i+1 >= argc) {
      help = true;
    } 
    else 
      {
	strcpy(shape_filename, argv[i]);
	strcpy(op_prefix, argv[i+1]);
	    
	// when input filename is seen the output file prefix 
	// is also seen .. so skip the next two parameters.
	i++;
      }
  }

  if ( help) {
    cerr << "Usage: " << argv[0] 
	 << " [-L | -rc | -thif <angle> | -thff <angle> | -mr <value> | -opc <int value> ]"
	 << " infile outfile_prefix " << endl;
    exit( 1);
  }
  
  cerr << endl << "shape name : " << shape_filename << endl << endl;
  CGAL::Timer timer;
  timer.start();
  // read mesh.
  Mesh mesh;
  cerr << "Reading mesh ";
  read_mesh(mesh, shape_filename, read_color_opacity);
  cerr << "done." << endl;
  // align mesh.
  cerr << "Alignment ";
  am(mesh);
  cerr << "done." << endl;
  cerr << "Vertex normals ";
  compute_mesh_vertex_normal(mesh, flip);
  cerr << "computed." << endl;
  write_mesh(mesh, op_prefix);

  // Compute Delaunay of the vertex set.
  Triangulation triang;
  cerr << "Triangulation ";
  double x_min = DBL_MAX, y_min = DBL_MAX, z_min = DBL_MAX,
    x_max = -DBL_MAX, y_max = -DBL_MAX, z_max = -DBL_MAX;
  for(int i = 0; i < mesh.get_nv(); i ++) 
    {
      Point p = mesh.vert_list[i].point();
      double x = p[0], y = p[1], z = p[2];
      if( x < x_min ) x_min = x;
      if( x > x_max ) x_max = x;
      if( y < y_min ) y_min = y;
      if( y > y_max ) y_max = y;
      if( z < z_min ) z_min = z;
      if( z > z_max ) z_max = z;
      Vertex_handle vh = triang.insert(p);
      vh->mesh_vid = i;
      if(i%1000 == 0) cerr << ".";
    }
  cerr << "done." << endl;
  bounding_box.push_back(x_min);
  bounding_box.push_back(x_max);
  bounding_box.push_back(y_min);
  bounding_box.push_back(y_max);
  bounding_box.push_back(z_min);
  bounding_box.push_back(z_max);

  // Initialization of Triangulation datastructure.
  cerr << "Init ";
  initialize(triang);
  cerr << ".";
  // compute voronoi vertex
  compute_voronoi_vertex_and_cell_radius(triang);
  cerr << ". done." << endl;

  cerr << "Computing S_MAX ";
  cerr << " mr = " << mr << " ";
  vector<int> sorted_smax_index_vector = compute_smax(triang, mesh, cluster_set, mr);
  cerr << " done." << endl;

  cerr << "Number of segments " << (int)sorted_smax_index_vector.size() << endl;

  timer.stop();

  // ---------- End of Segmentation ------------

  // write the outside pockets of the dataset
  write_smax(triang, cluster_set, sorted_smax_index_vector, 
             output_seg_count, op_prefix);

  return 0;
}

#endif
