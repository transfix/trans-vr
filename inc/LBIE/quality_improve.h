/*
  Copyright 2006 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of LBIE.

  LBIE is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  LBIE is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __VOLUME__QUALITY_IMPROVE_H__
#define __VOLUME__QUALITY_IMPROVE_H__

#include <boost/shared_ptr.hpp>

#include <VolMagick/VolMagick.h>
#include <cvcraw_geometry/Geometry.h>
#include <LBIE/octree.h>

/*
  This is a simple interface to the LBIE library's mesh refinement functionality.
*/

namespace LBIE
{
  namespace DEFAULT
  {
    const float ERR = 1.2501f;	    //-0.0001f    //99.99f		//0.0001f
    const float ERR_IN = 0.0001f; 
    const float IVAL = 0.5001f; //-0.0001f    //-0.5000f	//-0.0001f  //-1.0331		0.0001
    const float IVAL_IN = 9.5001f;    //10.000f
  }

  void copyGeoframeToGeometry(const geoframe& input, boost::shared_ptr<Geometry>& output);
  void copyGeoframeToGeometry(const geoframe& input, Geometry& output);
  void copyGeometryToGeoframe(const boost::shared_ptr<Geometry>& input, geoframe& output);
  void copyGeometryToGeoframe(Geometry& input, geoframe& output);
  boost::shared_ptr<Geometry> mesh(const VolMagick::Volume& vol,
				   float iso_val = DEFAULT::IVAL,
				   float iso_val_in = DEFAULT::IVAL_IN,
				   float err_tol = DEFAULT::ERR);
  boost::shared_ptr<Geometry> refineMesh(const boost::shared_ptr<Geometry>& input);
}

#endif
