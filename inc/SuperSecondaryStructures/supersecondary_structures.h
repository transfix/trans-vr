/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of SuperSecondaryStructures.

  SuperSecondaryStructures is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  SuperSecondaryStructures is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: tight_cocone.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef __SUPERSECONDARY_STRUCTURES_H__
#define __SUPERSECONDARY_STRUCTURES_H__

/*
  Main header!! Include only this!
*/

#include <VolMagick/VolMagick.h>
//#include <cvcraw_geometry/Geometry.h>

#include <cvcraw_geometry/cvcgeom.h>
#include <boost/shared_ptr.hpp>
#include <string>

namespace SuperSecondaryStructures
{
  // -- flatness marking ---
  const double DEFAULT_ANGLE = M_PI / 8.0;      // Half of the co-cone angle. 
  const double DEFAULT_SHARP = 2 * M_PI / 3.0;  // Angle of sharp edges.
  const double DEFAULT_RATIO = 1.2 * 1.2;       // Squared thinness factor. 
  const double DEFAULT_FLAT  = M_PI / 3.0;      // Angle for flatness Test

  // -- robust cocone ---
  const double DEFAULT_BIGBALL_RATIO  = (1./4.)*(1./4.);  // parameter to choose big balls.
  const double DEFAULT_THETA_IF_d  = 5.0;       // parameter for infinite-finite deep intersection.
  const double DEFAULT_THETA_FF_d  = 10.0;      // parameter for finite-finite deep intersection.

  // -- medial axis ---
  const double DEFAULT_MED_THETA = M_PI*22.5/180.0; // original: M_PI*22.5/180.0;
  // lowering the ratio makes medial axis more hairy.
  const double DEFAULT_MED_RATIO = 8.0*8.0; // original: 8.0*8.0;

  // --- segmentation ---

  const double DEFAULT_MERGE_RATIO = .3*.3;   // ratio to merge two segments.

   // --- segmentation output ---

   const int DEFAULT_OUTPUT_SEG_COUNT = 30;

 
  const std::string DEFAULT_OUTPUT_PREFIX = "seg";
 
  class Parameters
  {
  public:
    Parameters() : 
      _b_robust(false),
      _bb_ratio(DEFAULT_BIGBALL_RATIO),
      _theta_ff(M_PI/180.0*DEFAULT_THETA_FF_d),
      _theta_if(M_PI/180.0*DEFAULT_THETA_IF_d),
      _flatness_ratio(DEFAULT_RATIO),
      _cocone_phi(DEFAULT_ANGLE),
      _flat_phi(DEFAULT_FLAT),
	  _merge_ratio(DEFAULT_MERGE_RATIO),
	  _seg_number(DEFAULT_OUTPUT_SEG_COUNT),
	  _out_prefix(DEFAULT_OUTPUT_PREFIX)
      //_theta(DEFAULT_MED_THETA),
      //_medial_ratio(DEFAULT_MED_RATIO),
	{}

      bool b_robust() const { return _b_robust; }
      Parameters& b_robust(bool b) { _b_robust = b; return *this; }
      double bb_ratio() const { return _bb_ratio; }
      Parameters& bb_ratio(double r) { _bb_ratio = r; return *this; }
      double theta_ff() const { return _theta_ff; }
      Parameters& theta_ff(double f) { _theta_ff = f; return *this; }
      double theta_if() const { return _theta_if; }
      Parameters& theta_if(double f) { _theta_if = f; return *this; }
      double flatness_ratio() const { return _flatness_ratio; }
      Parameters& flatness_ratio(double r) { _flatness_ratio = r; return *this; }
      double cocone_phi() const { return _cocone_phi; }
      Parameters& cocone_phi(double p) { _cocone_phi = p; return *this; }
      double flat_phi() const { return _flat_phi; }
      Parameters& flat_phi(double p) { _flat_phi = p; return *this; }
	  double merge_ratio() const {return _merge_ratio;}
	  Parameters& merge_ratio(double p){ _merge_ratio = p; return *this; }
	  int seg_number() const {return _seg_number;}
	  Parameters& seg_number(int p){ _seg_number = p; return *this; }
	  std::string out_prefix() const {return _out_prefix;}
	  Parameters& out_prefix(std::string p){ _out_prefix = p; return *this;}
/*       double theta() const { return _theta; } */
/*       Parameters& theta(double t) { _theta = t; return *this; } */
/*       double medial_ratio() const { return _medial_ratio; } */
/*       Parameters& medial_ratio(double r) { _medial_ratio = r; return *this; } */

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

	// for segmentation
	double _merge_ratio;
	int _seg_number;
    
	std::string _out_prefix;
    // for medial axis;
/*     double _theta; */
/*     double _medial_ratio; */
  };

  CVCGEOM_NAMESPACE::cvcgeom_t surfaceReconstruction(const VolMagick::Volume& vol); //don't use this one!
  CVCGEOM_NAMESPACE::cvcgeom_t generateBoundaryPointCloud(const VolMagick::Volume& vol, float tlow = 0, float thigh = 1);
  void surfaceReconstruction(const CVCGEOM_NAMESPACE::cvcgeom_t& pointCloud,
						    const Parameters& params = Parameters());
}

#endif
