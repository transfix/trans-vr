#ifndef INTERSECT_H
#define INTERSECT_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/util.h>

namespace SuperSecondaryStructures {
bool does_intersect_ray3_seg3_in_plane(const Ray &r, const Segment &s);

Point intersect_ray3_seg3(const Ray &r, const Segment &s,
                          bool &is_correct_intersection);

bool does_intersect_convex_polygon_segment_3_in_3d(
    const vector<Point> &conv_poly, const Segment &s);

} // namespace SuperSecondaryStructures
#endif // INTERSECT_H
