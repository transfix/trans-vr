// Author: Justin Kinney
// Date: Feb 2009

#ifndef CF_RECONSTRUCT2CONTOURTILER_H
#define CF_RECONSTRUCT2CONTOURTILER_H 1

#include "common.h"

#include <vector>

ContourFilter_BEGIN_NAMESPACE class Contour;
class Histogram;
class Object;
class Point;
class SplinePoint;
ContourFilter_END_NAMESPACE

#include "container.h"
#include "controls.h"

    ContourFilter_BEGIN_NAMESPACE

    // bool distinguishable (double a,double b,double epsilon);
    // bool distinguishable (double a,double b);

    inline bool
    distinguishable(double a, double b, double epsilon) {
  double c;
  c = a - b;
  if (c < 0)
    c = -c;
  if (a < 0)
    a = -a;
  if (a < 1)
    a = 1;
  if (b < 0)
    b = -b;
  if (b < a)
    return (c > a * epsilon);
  else
    return (c > b * epsilon);
}

/** Determine if two floating-point precision numbers
 * are equivalent in value within MY_DOUBLE_EPSILON.
 * \param[in] a First number.
 * \param[in] b Second number.
 * \return 1 if Inputs are different; 0 otherwise.
 */

inline bool distinguishable(double a, double b) {
  return distinguishable(a, b, Controls::instance().getEpsilon());
}
ContourFilter_END_NAMESPACE

#endif
