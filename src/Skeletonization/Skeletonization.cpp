/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Skeletonization.

  Skeletonization is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Skeletonization is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// Samrat

#include <Skeletonization/datastruct.h>
#include <Skeletonization/graph.h>
#include <Skeletonization/hfn_util.h>
#include <Skeletonization/init.h>
#include <Skeletonization/intersect.h>
#include <Skeletonization/medax.h>
#include <Skeletonization/op.h>
#include <Skeletonization/rcocone.h>
#include <Skeletonization/robust_cc.h>
#include <Skeletonization/skel.h>
#include <Skeletonization/tcocone.h>
#include <Skeletonization/u1.h>
#include <Skeletonization/u2.h>
#include <Skeletonization/util.h>
#include <stdlib.h>
// #include "helix.h"

#include <Skeletonization/Skeletonization.h>

namespace Skeletonization {
// -- bounding box ---
const double BB_SCALE = 1.0;
vector<double> bounding_box;

static inline Simple_skel build_simple_skel(const Skel &skel) {
  Line_strip_set line_strips;
  Polygon_set polygons;

  // srand48(0);
  srand(0);
  for (int i = 0; i < (int)skel.sorted_pl_id.size(); i++) {
    int cl_id = skel.sorted_pl_id[i];
    if (!skel.is_big_pl[cl_id]) {
      // draw the star.
      // fout << "# --- " << endl;
      Point c = skel.pl_C[cl_id];
      for (int j = 0; j < (int)skel.active_bdy[cl_id].size(); j++) {
        Segment seg(c, skel.active_bdy[cl_id][j]);
        Simple_line_strip line_strip;
        line_strip.push_back(
            Simple_vertex(seg.point(0), Simple_color(0, 0, 1, 0.5)));
        line_strip.push_back(
            Simple_vertex(seg.point(1), Simple_color(0, 0, 1, 0.5)));
        line_strips.insert(line_strip);
      }
      // fout << "# --- " << endl;
    } else {
      // fout << "# --- " << endl;
      // double r = drand48(), g = drand48(), b = drand48(), a = 1;
      double r = double(rand()) / double(RAND_MAX),
             g = double(rand()) / double(RAND_MAX),
             b = double(rand()) / double(RAND_MAX), a = 1;
      for (int j = 0; j < (int)skel.pl[cl_id].size(); j++) {
        // double scale_color = skel.pl[cl_id][j].width/skel.max_pgn_width;
        //  draw_poly(skel.pl[cl_id][j].ordered_v_list,
        //  r*scale_color, g*scale_color, b*scale_color, a, fout);
        Simple_polygon poly;
        for (std::vector<Point>::const_iterator p =
                 skel.pl[cl_id][j].ordered_v_list.begin();
             p != skel.pl[cl_id][j].ordered_v_list.end(); p++)
          poly.push_back(Simple_vertex(*p, Simple_color(r, g, b, a)));
        polygons.insert(poly);
      }
      // fout << "# --- " << endl;
    }
  }

  // fout << "# LINEAR " << endl << endl;
  // write the linear part.
  for (int i = 0; i < (int)skel.L.size(); i++) {
    Polyline l = skel.L[i];
    bool is_far = false;
    for (int j = 0; j < (int)l.ordered_v_list.size(); j++)
      if (is_outside_bounding_box(l.ordered_v_list[j], bounding_box))
        is_far = true;
    if (is_far)
      continue;
    // draw the polyline with a random color.
    // double r = drand48(), g = drand48(), b = drand48(), a = 1;
    double r = double(rand()) / double(RAND_MAX),
           g = double(rand()) / double(RAND_MAX),
           b = double(rand()) / double(RAND_MAX), a = 1;
    r = 0;
    g = 1;
    b = 0;
    if (skel.L_invalid[i])
      continue; // a = 0.1;

    for (int j = 0; j < (int)l.ordered_v_list.size() - 1; j++) {
      Segment seg(l.ordered_v_list[j], l.ordered_v_list[j + 1]);
      Simple_line_strip line_strip;
      line_strip.push_back(
          Simple_vertex(seg.point(0), Simple_color(r, g, b, a)));
      line_strip.push_back(
          Simple_vertex(seg.point(1), Simple_color(r, g, b, a)));
      line_strips.insert(line_strip);
    }
  }

  for (int i = 0; i < (int)skel.L.size(); i++) {
    if (skel.L_invalid[i])
      continue;
    Simple_line_strip line_strip;
    for (int j = 0; j < (int)skel.L[i].cell_list.size(); j++) {
      Cell_handle c = skel.L[i].cell_list[j];
      if (c->big_pl()) {
        continue;
      }
      if (sqrt(c->cell_radius()) >= 2.5 + 0.5 ||
          sqrt(c->cell_radius()) <= 2.5 - 0.5)
        continue;

      Simple_color color;
      if (is_maxima(c))
        color = Simple_color(1.0, 0.0, 0.0, 1.0);
      else
        color = Simple_color(0.0, 1.0, 0.0, 1.0);

      line_strip.push_back(Simple_vertex(skel.L[i].ordered_v_list[j], color));
    }
    line_strips.insert(line_strip);
  }

  return Simple_skel(line_strips, polygons);
}

// -----------------------------------------------------------------------
// main skeletonization call
// -----------------------------------------------------------------------
Simple_skel skeletonize(const boost::shared_ptr<Geometry> &geom,
                        const Parameters &params) {
  // robust cocone parameters.
  bool b_robust = params.b_robust();   // false;
  double bb_ratio = params.bb_ratio(); // DEFAULT_BIGBALL_RATIO;
  double theta_ff = params.theta_ff(); // M_PI/180.0*DEFAULT_THETA_FF_d;
  double theta_if = params.theta_if(); // M_PI/180.0*DEFAULT_THETA_IF_d;

  // double mr = 1.05*1.05;
  // int helix_cnt = 20;

  // for flatness marking (in cocone)
  double flatness_ratio = params.flatness_ratio(); // DEFAULT_RATIO;
  double cocone_phi = params.cocone_phi();         // DEFAULT_ANGLE;
  double flat_phi = params.flat_phi();             // DEFAULT_FLAT;

  // For medial axis
  double theta = params.theta();               // DEFAULT_MED_THETA;
  double medial_ratio = params.medial_ratio(); // DEFAULT_MED_RATIO;

  // For selection of big planar clusters.
  double threshold = params.threshold();                     // 0.1;
  int pl_cnt = params.pl_cnt();                              // 2;
  bool discard_by_threshold = params.discard_by_threshold(); // false;

  CGAL::Timer timer;

  int cnt = 0;
  Triangulation triang;
  timer.start();

  std::vector<Point> robust_points;

  // #if 0
  if (b_robust) {
    int numvert = geom->m_NumTriVerts;

    for (int i = 0; i < numvert; i++) {
      Triangulation::Point p =
          Point(geom->m_TriVerts[3 * i + 0], geom->m_TriVerts[3 * i + 1],
                geom->m_TriVerts[3 * i + 2]);
      triang.insert(p);
    }

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
    robust_points = robust_cocone(bb_ratio, theta_ff, theta_if, triang);
    cerr << " done." << endl;
    cerr << "Time : " << timer.time() << endl << endl;
    timer.reset();

#ifdef DEBUG_OP
    // write_iobdy(triang, output_file_prefix);
#endif

    // Create a new triangulation from the pointset taken
    // from "output_file_prefix.tcip".
    // Delete the earlier triangulation.
    triang.clear();
  }
  // #endif

  cerr << "DT 2 ";

  // Maintain the min-max span of the pointset in 3 directions.
  double x_min = DBL_MAX, x_max = -DBL_MAX, y_min = DBL_MAX, y_max = -DBL_MAX,
         z_min = DBL_MAX, z_max = -DBL_MAX;

  int total_pt_cnt = 0;

  if (robust_points.size() > 0) // if we used robust cocone, use those points
                                // instead of the original points
  {
    for (std::vector<Point>::iterator i = robust_points.begin();
         i != robust_points.end(); i++) {
      cnt++;
      total_pt_cnt++;
      if (cnt >= 1000) {
        cerr << "." << flush;
        cnt = 0;
      }

      Vertex_handle new_vh = triang.insert(*i);
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
  } else {
    for (unsigned int i = 0; i < geom->m_NumTriVerts; i++) {
      double x = geom->m_TriVerts[3 * i + 0];
      double y = geom->m_TriVerts[3 * i + 1];
      double z = geom->m_TriVerts[3 * i + 2];

      cnt++;
      total_pt_cnt++;
      if (cnt >= 1000) {
        cerr << "." << flush;
        cnt = 0;
      }

      Vertex_handle new_vh = triang.insert(Point(x, y, z));
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
  cerr << " done." << endl;

  cerr << "Total point count: " << total_pt_cnt << endl;
  cerr << "Del Time : " << timer.time() << endl << endl;
  timer.reset();

  // Bounding box of the point set.
  bounding_box.push_back(x_min - BB_SCALE * (x_max - x_min));
  bounding_box.push_back(x_max + BB_SCALE * (x_max - x_min));

  bounding_box.push_back(y_min - BB_SCALE * (y_max - y_min));
  bounding_box.push_back(y_max + BB_SCALE * (y_max - y_min));

  bounding_box.push_back(z_min - BB_SCALE * (z_max - z_min));
  bounding_box.push_back(z_max + BB_SCALE * (z_max - z_min));

  // ------------------------------------------
  // Initialization of all the required fields
  // needed for Tight Cocone
  // ------------------------------------------
  cerr << "Init 2 ";
  initialize(triang);
  cerr << ".";

  // compute voronoi vertex
  compute_voronoi_vertex_and_cell_radius(triang);
  cerr << ". done." << endl;
  cerr << "Time : " << timer.time() << endl << endl;
  timer.reset();
  // ------------ End Intialization -------------

  // ------------------------------------------
  // Surface Reconstruction using Tight Cocone
  // ------------------------------------------
  cerr << "TC ";
  tcocone(cocone_phi, DEFAULT_SHARP, flat_phi, flatness_ratio, triang);
  cerr << " done." << endl;
  cerr << "Time : " << timer.time() << endl << endl;
  timer.reset();
  // write the water-tight surface.
  // write_wt(triang, output_file_prefix);

  // Medial Axis
  timer.reset();
  cerr << "Medial axis " << flush;
#ifdef NO_TIGHTCOCONE
  compute_poles(triang);
  mark_flat_vertices(triang, flatness_ratio, cocone_phi, flat_phi);
#endif
  compute_medial_axis(triang, theta, medial_ratio);
  cerr << " done." << endl;
  cerr << "TIME: " << timer.time() << " sec(s)." << endl << endl;
  timer.reset();

#ifdef DEBUG_OP
  // writing the medial axis in OFF format.
  write_axis(triang, "medax");
#endif

  // build skeleton from u1 and u2.
  // skeleton has two parts - planar and linear.
  cerr << "Skeleton building starts." << endl << endl;
  Skel skel = Skel();

  cerr << "\tU1";
  compute_u1(triang);
  cerr << " done." << endl;
#ifdef DEBUG_OP
  // write_u1(triang, output_file_prefix);
#endif
  // cluster planar patches, compute area and center.
  cerr << "\tProcessing planar patches";
  cluster_planar_patches(triang, skel);
  cerr << " done." << endl;
  cerr << "\tTIME: " << timer.time() << " sec(s)." << endl << endl;
  timer.reset();

  cerr << "\tU2";
  Graph graph;
  compute_u2(triang, graph);
  cerr << " done." << endl;
#ifdef DEBUG_OP
  // write_u2(triang, graph, output_file_prefix);
#endif
  // organize the linear portion into a network of polylines.
  cerr << "\tPorting Graph to Skel ";
  port_graph_to_skel(graph, skel);
  cerr << " done." << endl;
  cerr << "\tTIME: " << timer.time() << " sec(s)." << endl << endl;
  timer.stop();

  cerr << "\tRemoval of small flat components ";
  // sort and star the clusters.
  // sorting is done by area.
  star(triang, graph, skel);
  cerr << ".";
  sort_cluster_by_area(skel);
  cerr << ".";
  filter_small_clusters(skel, pl_cnt, threshold, discard_by_threshold);
  cerr << ".";
  cerr << " done." << endl;
  cerr << "Skeleton building ends." << endl << endl;

  /////// output result goes here!!..need to fix write_skel to return a
  ///geometry

  // write_skel(skel, output_file_prefix);

#if 0
      // Selection of Alpha Helix and Beta Sheet Candidates.
      cerr << "Helix candidate selection ";
      vector<Point> alpha_helix_cand;
      for(int i = 0; i < (int)skel.L.size(); i ++)
	{
	  if( skel.L_invalid[i] ) continue;
	  for(int j = 0; j < (int)skel.L[i].cell_list.size(); j ++)
	    {
	      Cell_handle c = skel.L[i].cell_list[j];
	      if( c->big_pl() ) continue;
	      if( sqrt(c->cell_radius()) < 1 ||
		  sqrt(c->cell_radius()) > 4 )
		continue;
	      alpha_helix_cand.push_back(c->voronoi());
	    }
	}
      cerr << "done." << endl;
      // write the candidates in a file.
      string s (output_file_prefix);
      s += ".HELIX_cand";
      ofstream fout;
      fout.open(s.c_str());
      for(int i = 0; i < (int)alpha_helix_cand.size(); i ++)
	fout << alpha_helix_cand[i] << endl;
#endif

  return build_simple_skel(skel);
}

} // namespace Skeletonization
