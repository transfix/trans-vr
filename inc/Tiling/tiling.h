#ifndef __TILING_H__
#define __TILING_H__

/*
  This is VolumeRover's interface into the tiling library... 
  do not include any other headers in external sources
*/

#include <VolumeGridRover/SurfRecon.h>
#include <cvcraw_geometry/Geometry.h>
#include <VolMagick/VolMagick.h>

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

namespace Tiling
{
  boost::shared_ptr<Geometry> surfaceFromContour(const SurfRecon::ContourPtr& contour,
						 const VolMagick::VolumeFileInfo& volinfo,
						 const std::string& tmpdir = std::string("."));
  
  std::vector<boost::shared_ptr<Geometry> > surfacesFromContours(const SurfRecon::ContourPtrArray& contours, 
								 const VolMagick::VolumeFileInfo& volinfo,
								 unsigned int var = 0, unsigned int time = 0,
								 const std::string& tmpdir = std::string("."));
};

#endif
