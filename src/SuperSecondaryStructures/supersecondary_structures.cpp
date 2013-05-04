


// -------------------------------------------------------------------
// Samrat Goswami.
// Date : 17th June, 2002
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Header files.
// -------------------------------------------------------------------

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/tcocone.h>
#include <SuperSecondaryStructures/rcocone.h>
#include <SuperSecondaryStructures/util.h>
#include <SuperSecondaryStructures/op.h>
#include <SuperSecondaryStructures/init.h>
#include <SuperSecondaryStructures/smax.h>
#include <SuperSecondaryStructures/medax.h>
#include <iostream>

#include <SuperSecondaryStructures/supersecondary_structures.h>
#include <boost/scoped_array.hpp>

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

// ------- cocone -----
/*

const double DEFAULT_ANGLE = M_PI / 8.0;      // Half of the co-cone angle. 
const double DEFAULT_SHARP = 2 * M_PI / 3.0;  // Angle of sharp edges.
const double DEFAULT_RATIO = 1.2 * 1.2;       // Squared thinness factor. 
const double DEFAULT_FLAT  = M_PI / 3.0;      // Angle for flatness Test

// -- medial axis ---

const double DEFAULT_MED_THETA = M_PI*22.5/180.0; // original: M_PI*22.5/180.0;
const double DEFAULT_MED_RATIO = 8.0*8.0; // original: 8.0*8.0;


// --- segmentation ---

const double DEFAULT_MERGE_RATIO = .3*.3;   // ratio to merge two segments.

// --- segmentation output ---

const int DEFAULT_OUTPUT_SEG_COUNT = 30;

// -- robust cocone ---

const double DEFAULT_BIGBALL_RATIO  = 4.*4.;  // parameter to choose big balls.
const double DEFAULT_THETA_IF_d  = 5.0;       // parameter for infinite-finite deep intersection.
const double DEFAULT_THETA_FF_d  = 10.0;      // parameter for finite-finite deep intersection.
const double DEFAULT_THETA_II_d  = 30.0;      // parameter for infinite-infinite deep intersection.



// -----------------------------------------------------------------------
// main
// ----
// Parses commandline arguments and calls cocone and writes the output.
// -----------------------------------------------------------------------

int 
main( int argc, char **argv) {

*/

namespace SuperSecondaryStructures
{

  map<int, cell_cluster> cluster_set;

  // segmentation parameters.
  float mr = DEFAULT_MERGE_RATIO;
  // parameter to see the K biggest segments.
  int output_seg_count = DEFAULT_OUTPUT_SEG_COUNT;
  

  // robust cocone parameters.
  double bb_ratio = DEFAULT_BIGBALL_RATIO;
  double theta_ff = M_PI/180.0*DEFAULT_THETA_FF_d;
  double theta_if = M_PI/180.0*DEFAULT_THETA_IF_d;

  //For medial axis
  double theta = DEFAULT_MED_THETA;
  double medial_ratio = DEFAULT_MED_RATIO;

  // degeneracy indicator.
  // bool degenerate_vector[4] = {0, 0, 0, 0};

  // segmentation and matching filenames.
  char shape_filename[90];
  char output_file_prefix[90];

  // Check commandline options.
  bool help = false;
  // To output the copyright information etc.
  bool long_banner = false;

  // If do_robust_cocone flag is true
  // then it does a pre-processing stage
  bool do_robust_cocone = false;
 
 /*
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
    else if(strcmp( "-rc", argv[i]) == 0)
    {
	do_robust_cocone = true;
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
    else if ( i+1 >= argc) {
      help = true;
    } 
    else 
    {
	    strcpy(shape_filename, argv[i]);
	    strcpy(output_file_prefix, argv[i+1]);
	    
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

  // open the file containing the current shape.
  ifstream fin;
  fin.open(shape_filename);
  if ( ! fin) 
  { 
	cerr << "Error: cannot open file '" << shape_filename
	     << "' for reading." <<endl;
	exit(1);
  }

  cerr << endl << "shape name : " << shape_filename << endl << endl;
*/
  void surfaceReconstruction(const CVCGEOM_NAMESPACE::cvcgeom_t & pointCloud, const Parameters& params)
  {
    bool b_robust = params.b_robust();//false;
    double bb_ratio = params.bb_ratio();//DEFAULT_BIGBALL_RATIO;
    double theta_ff = params.theta_ff();//M_PI/180.0*DEFAULT_THETA_FF_d;
    double theta_if = params.theta_if();//M_PI/180.0*DEFAULT_THETA_IF_d;

    // for flatness marking (in cocone)
    double flatness_ratio = params.flatness_ratio();//DEFAULT_RATIO;
    double cocone_phi = params.cocone_phi();//DEFAULT_ANGLE;
    double flat_phi = params.flat_phi();//DEFAULT_FLAT;
    
	double mr = params.merge_ratio();
	mr *= mr;
	int output_seg_count = params.seg_number();
	strcpy(output_file_prefix, params.out_prefix().c_str());
  
   CGAL::Timer timer;
   timer.start();
 


  // read the points and do the delaunay triangulation.
   cerr << "Delaunay triangulation " << flush;  
   int cnt = 0;
   Triangulation triang;
  
   std::vector<Point> robust_points;

  int numvert = pointCloud.points().size();
  for(int i = 0; i<numvert; i++)
  {
		Triangulation::Point p = Point(pointCloud.points()[i][0],
								pointCloud.points()[i][1],
								pointCloud.points()[i][2]);
	    triang.insert(p);
		if(i%1000 == 0){
			cerr<<"."<< flush;
		}

	}

  

  cerr << " done." << endl; 
  
  cerr << "Time : " << timer.time() << endl;

  timer.reset();


  if(b_robust)
  {
       // -------------------------------------------
       // Robust Cocone.
       // When we deal with noisy data we need a tool
       // to pre-process the data before doing
       // the surface reconstruction. Robust Cocone
       // is a tool to do that. 
       // -------------------------------------------

       // Robust cocone will take the triangulation and
       // will output a subset of points which it thinks
       // lies on a smooth underlying surface.
       // The pointset will be output in a file called
       // "output_file_prefix.tcip".
       cerr << "\t" << "Robust Cocone ";
	   robust_points = robust_cocone(bb_ratio, theta_ff, theta_if, triang);
       cerr << " done." << endl;

       cerr << "\t" << "Time : " << timer.time() << endl;

       timer.reset();

       if(robust_points.size() > 0)
	   {
	    
       // Create a new triangulation from the pointset taken 
       // from "output_file_prefix.tcip".
       // Delete the earlier triangulation.
       triang.clear();

    
       cerr << "\t" << "Rebuilding Delaunay Triangulation ";
    	for(std::vector<Point>::iterator i = robust_points.begin();
	    	i != robust_points.end(); i++)
	    {
           cnt++;
           if(cnt >= 1000){
              cerr<<"." << flush;
              cnt = 0;
           }
           triang.insert( *i);
       }

       cerr << " done." << endl;

       cerr << "\t" << "Time : " << timer.time() << endl;

       timer.reset();
	  }

  }

  // ------------------------------------------
  // Initialization of all the required fields
  // needed for Tight Cocone and Segmentation
  // ------------------------------------------
  cerr << "Initialization ";
  initialize(triang);
  cerr << ".";

  // compute voronoi vertex
  compute_voronoi_vertex_and_cell_radius(triang);
  cerr << ". done." << endl;

  cerr << "Time : " << timer.time() << endl;

  timer.reset();

  // ------------------------------------------
  // Surface Reconstruction using Tight Cocone
  // ------------------------------------------

  cerr << "Surface Reconstruction ";
 
  tcocone(DEFAULT_ANGLE, DEFAULT_SHARP, DEFAULT_FLAT, DEFAULT_RATIO, triang);
  cerr << " done." << endl;
  cerr << "Time : " << timer.time() << endl;

  cerr << "Medial axis computation ";
  // compute_medial_axis(triang, theta, medial_ratio);
  cerr << "done." << endl;

  // write the water-tight surface.
  write_wt(triang, output_file_prefix);

  timer.reset();

  // compute the stable manifold of the index-2 saddles.

  cerr << "Computing S_MAX ";
  vector<int> sorted_smax_index_vector = compute_smax(triang, cluster_set, mr);
  cerr << " done." << endl;

  cerr << "Number of segments " << (int)sorted_smax_index_vector.size() << endl;

  timer.stop();

  // ---------- End of Segmentation ------------

  cerr << "Writing meshes ..." << endl; 
  // write the outside pockets of the dataset
  write_smax(triang, cluster_set, sorted_smax_index_vector, output_seg_count, output_file_prefix);

  cerr << " done." << endl;
}

};
