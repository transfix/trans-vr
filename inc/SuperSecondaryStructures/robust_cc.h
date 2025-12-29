#ifndef ROBUST_CC_H
#define ROBUST_CC_H

#include <SuperSecondaryStructures/datastruct.h>

namespace SuperSecondaryStructures {

Point nondg_voronoi_point(const Point &a, const Point &b, const Point &c,
                          const Point &d, bool &is_correct_computation);

Point dg_voronoi_point(const Point &a, const Point &b, const Point &c,
                       const Point &d, bool &is_correct_computation);

Point nondg_cc_tr_3(const Point &a, const Point &b, const Point &c,
                    bool &is_correct_computation);

Point cc_tr_3(const Point &a, const Point &b, const Point &c);

double sq_cr_tr_3(const Point &a, const Point &b, const Point &c);

Point circumcenter(const Facet &f);

double circumradius(const Facet &f);

} // namespace SuperSecondaryStructures

#endif // ROBUST_CC_H
