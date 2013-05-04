#ifndef __SMOOTHING_H__
#define __SMOOTHING_H__

#include <cvcraw_geometry/Geometry.h>

namespace Smoothing
{
  void smoothGeometry(Geometry *geo, float delta = 0.1f, bool fix_boundary = false);
};

#endif
