/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PocketTunnel.

  PocketTunnel is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PocketTunnel is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <PocketTunnel/pocket_tunnel.h>

// #define _DEBUG_OUTPUT_

namespace PocketTunnel {

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------
// ------- cocone -----
const double DEFAULT_ANGLE = M_PI / 8.0;     // Half of the co-cone angle.
const double DEFAULT_SHARP = 2 * M_PI / 3.0; // Angle of sharp edges.
const double DEFAULT_RATIO = 1.2 * 1.2;      // Squared thinness factor.
const double DEFAULT_FLAT = M_PI / 3.0;      // Angle for flatness Test

// --- segmentation output ---
const int DEFAULT_OUTPUT_SEG_COUNT = 5; // 20;

// -- robust cocone ---
const double DEFAULT_BIGBALL_RATIO =
    4. * 4.; // parameter to choose big balls.
const double DEFAULT_THETA_IF_d =
    5.0; // parameter for infinite-finite deep intersection.
const double DEFAULT_THETA_FF_d =
    10.0; // parameter for finite-finite deep intersection.
const double DEFAULT_THETA_II_d =
    30.0; // parameter for infinite-infinite deep intersection.

CVCGEOM_NAMESPACE::cvcgeom_t *
pocket_tunnel_fromsurf(const CVCGEOM_NAMESPACE::cvcgeom_t *molsurf,
                       int num_pockets,
                       int num_tunnels) // input surface data.
{

  map<int, cell_cluster> cluster_set;
  //  int output_seg_count = DEFAULT_OUTPUT_SEG_COUNT;

  // robust cocone parameters.
  double bb_ratio = DEFAULT_BIGBALL_RATIO;
  double theta_ff = M_PI / 180.0 * DEFAULT_THETA_FF_d;
  double theta_if = M_PI / 180.0 * DEFAULT_THETA_IF_d;

  vector<Point> pts_list;
  for (int i = 0; i < molsurf->points().size(); i++) {
    float x = molsurf->points()[i][0], y = molsurf->points()[i][1],
          z = molsurf->points()[i][2];
    pts_list.push_back(Point(x, y, z));
  }

  // CGAL::Timer timer;
  // timer.start();

  // cout <<"list size:" <<  pts_list.size() << endl;
  cerr << "Delaunay ";
  Triangulation triang;
  triang.insert(pts_list.begin(), pts_list.end());
  assert(triang.is_valid());
  cerr << "done." << endl;
  // cerr << "Time: " << timer.time() << endl; timer.reset();

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
  // cerr << "Time: " << timer.time() << endl; timer.reset();

  // ------------------------------------------
  // Surface Reconstruction using Tight Cocone
  // ------------------------------------------
  cerr << "Surface Reconstruction ";
  tcocone(DEFAULT_ANGLE, DEFAULT_SHARP, DEFAULT_FLAT, DEFAULT_RATIO, triang);
  cerr << " done." << endl;
  // cerr << "Time: " << timer.time() << endl; timer.reset();

#ifdef _DEBUG_OUTPUT_
  write_wt(triang, "debug_output/temp_recon");
#endif

  cerr << "Computing S_MAX ";
  double mr = 1.3; // not used in this routine.
  vector<int> sorted_smax_index_vector =
      compute_smax(triang, cluster_set, mr);
  cerr << " done." << endl;

  // detect pocket, tunnel, void.
  cerr << "Computing Pocket-Tunnels ";
  detect_handle(triang, cluster_set);
  cerr << " done." << endl;
  // cerr << "Time: " << timer.time() << endl; timer.reset();

#ifdef _DEBUG_OUTPUT_
  // write_handle(triang, cluster_set, sorted_smax_index_vector,
  // output_seg_count, "debug_output/temp_PTV");
#endif

  // convert the handles into rawc geometries to be viewed by TexMol.
  CVCGEOM_NAMESPACE::cvcgeom_t *PTV;
  convert_pocket_tunnel_to_rawc_geometry(&PTV, triang, cluster_set,
                                         sorted_smax_index_vector,
                                         num_pockets, num_tunnels);

  return PTV;
}

}; // namespace PocketTunnel
