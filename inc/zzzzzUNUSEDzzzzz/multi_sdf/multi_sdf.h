/* $Id: multi_sdf.h 2273 2010-07-09 23:34:41Z transfix $ */

#ifndef __MULTI_SDF_H__
#define __MULTI_SDF_H__

/*
  Main header, include only this!

  Note: this code has some functionality for applying weights to certain vertices.  However
  for now we are just using uniform weights.
*/

#include <boost/shared_ptr.hpp>
#include <cvcraw_geometry/Geometry.h>
#include <VolMagick/VolMagick.h>

namespace multi_sdf
{
  VolMagick::Volume signedDistanceFunction(/*
					     Input geometry for sdf.
					   */
					   const boost::shared_ptr<Geometry>& geom,
					   /*
					     Dimension of output sdf vol.
					   */
					   const VolMagick::Dimension& dim,
					   /*
					     Bounding box of output vol. If default initialized,
					     use extents of Geometry.
					   */
					   const VolMagick::BoundingBox& bbox = VolMagick::BoundingBox());
}

#endif
