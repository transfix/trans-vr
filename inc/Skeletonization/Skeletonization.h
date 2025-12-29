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

#ifndef __SKELETONIZATION_H__
#define __SKELETONIZATION_H__

/*
  Main header!! Include only this!
*/

#include <Skeletonization/skel.h>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <cvcraw_geometry/Geometry.h>
#include <set>
#include <vector>

namespace Skeletonization {
// -- flatness marking ---
const double DEFAULT_ANGLE = M_PI / 8.0;     // Half of the co-cone angle.
const double DEFAULT_SHARP = 2 * M_PI / 3.0; // Angle of sharp edges.
const double DEFAULT_RATIO = 1.2 * 1.2;      // Squared thinness factor.
const double DEFAULT_FLAT = M_PI / 3.0;      // Angle for flatness Test

// -- robust cocone ---
const double DEFAULT_BIGBALL_RATIO =
    (1. / 4.) * (1. / 4.); // parameter to choose big balls.
const double DEFAULT_THETA_IF_d =
    5.0; // parameter for infinite-finite deep intersection.
const double DEFAULT_THETA_FF_d =
    10.0; // parameter for finite-finite deep intersection.

// -- medial axis ---
const double DEFAULT_MED_THETA =
    M_PI * 22.5 / 180.0;                    // original: M_PI*22.5/180.0;
const double DEFAULT_MED_RATIO = 8.0 * 8.0; // original: 8.0*8.0;

class Parameters {
public:
  Parameters()
      : _b_robust(false), _bb_ratio(DEFAULT_BIGBALL_RATIO),
        _theta_ff(M_PI / 180.0 * DEFAULT_THETA_FF_d),
        _theta_if(M_PI / 180.0 * DEFAULT_THETA_IF_d),
        _flatness_ratio(DEFAULT_RATIO), _cocone_phi(DEFAULT_ANGLE),
        _flat_phi(DEFAULT_FLAT), _theta(DEFAULT_MED_THETA),
        _medial_ratio(DEFAULT_MED_RATIO), _threshold(0.1), _pl_cnt(2),
        _discard_by_threshold(false) {}

  bool b_robust() const { return _b_robust; }
  Parameters &b_robust(bool b) {
    _b_robust = b;
    return *this;
  }
  double bb_ratio() const { return _bb_ratio; }
  Parameters &bb_ratio(double r) {
    _bb_ratio = r;
    return *this;
  }
  double theta_ff() const { return _theta_ff; }
  Parameters &theta_ff(double f) {
    _theta_ff = f;
    return *this;
  }
  double theta_if() const { return _theta_if; }
  Parameters &theta_if(double f) {
    _theta_if = f;
    return *this;
  }
  double flatness_ratio() const { return _flatness_ratio; }
  Parameters &flatness_ratio(double r) {
    _flatness_ratio = r;
    return *this;
  }
  double cocone_phi() const { return _cocone_phi; }
  Parameters &cocone_phi(double p) {
    _cocone_phi = p;
    return *this;
  }
  double flat_phi() const { return _flat_phi; }
  Parameters &flat_phi(double p) {
    _flat_phi = p;
    return *this;
  }
  double theta() const { return _theta; }
  Parameters &theta(double t) {
    _theta = t;
    return *this;
  }
  double medial_ratio() const { return _medial_ratio; }
  Parameters &medial_ratio(double r) {
    _medial_ratio = r;
    return *this;
  }
  double threshold() const { return _threshold; }
  Parameters &threshold(double t) {
    _threshold = t;
    return *this;
  }
  int pl_cnt() const { return _pl_cnt; }
  Parameters &pl_cnt(int p) {
    _pl_cnt = p;
    return *this;
  }
  bool discard_by_threshold() const { return _discard_by_threshold; }
  Parameters &discard_by_threshold(bool d) {
    _discard_by_threshold = d;
    return *this;
  }

private:
  // robust cocone parameters
  bool _b_robust;
  double _bb_ratio;
  double _theta_ff;
  double _theta_if;

  // for flatness marking (in cocone)
  double _flatness_ratio;
  double _cocone_phi;
  double _flat_phi;

  // for medial axis;
  double _theta;
  double _medial_ratio;

  // for selection of big planar clusters
  double _threshold;
  int _pl_cnt;
  bool _discard_by_threshold;
};

typedef boost::tuple<double, double, double, double> Simple_color;
typedef boost::tuple<Point, Simple_color> Simple_vertex;
typedef std::vector<Simple_vertex> Simple_line_strip;
typedef std::set<Simple_line_strip> Line_strip_set;
typedef std::vector<Simple_vertex> Simple_polygon;
typedef std::set<Simple_polygon> Polygon_set;
typedef std::vector<Simple_polygon> Polygon_vec;
typedef boost::tuple<Line_strip_set, Polygon_set> Simple_skel;

Simple_skel skeletonize(const boost::shared_ptr<Geometry> &,
                        const Parameters &params = Parameters());
} // namespace Skeletonization

#include <Skeletonization/PolyTess.h>

#endif
