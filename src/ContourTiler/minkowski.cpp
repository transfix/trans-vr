//! \file examples/Minkowski_sum_2/approx_offset.cpp
// Computing the approximated offset of a polygon.

#include <CGAL/basic.h>

#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Cartesian.h>
#include <CGAL/approximated_offset_2.h>
#include <CGAL/offset_polygon_2.h>
#include <CGAL/Timer.h>
#include <CGAL/Gmpz.h>
#include <iostream>

#include <ContourTiler/test_common.h>

CONTOURTILER_BEGIN_NAMESPACE

typedef CGAL::Lazy_exact_nt<CGAL::Gmpq>           Lazy_exact_nt;

struct Offset_kernel : public CGAL::Cartesian<Lazy_exact_nt> {};
typedef CGAL::Polygon_2<Offset_kernel>                    In_polygon_2;

typedef CGAL::Gps_circle_segment_traits_2<Offset_kernel>  Gps_traits_2;
typedef Gps_traits_2::Polygon_2                    Offset_polygon_2;
typedef Gps_traits_2::Polygon_with_holes_2         Offset_polygon_with_holes_2;
typedef Offset_polygon_with_holes_2::General_polygon_2 Outer_polygon_2;

// In_polygon_2 to_mink(const Polygon_2& p)
// {
//   In_polygon_2 P;
//   for (Polygon_2::Vertex_iterator it = p.vertices_begin(); it != p.vertices_end(); ++it) {
//     P.push_back(In_polygon_2::Point_2(it->x(), it->y()));
//   }
//   return P;
// }

// Polygon_2 from_mink(const Outer_polygon_2& outer)
// {
//   Polygon_2 ret;
//   for (Outer_polygon_2::Curve_iterator it = outer.curves_begin(); it != outer.curves_end(); ++it) {
//     Outer_polygon_2::X_monotone_curve_2& c = *it;
//     Outer_polygon_2::Point_2& source = c.source();
//     ret.push_back(Point_2(to_double(source.x()), to_double(source.y())));
//   }
//   return ret;
// }

// Polygon_2 offset_minkowski(const Polygon_2& p, const Number_type radius)
// {
//   In_polygon_2 P = to_mink(p);

//   // Approximate the offset polygon.
// //   const Number_type            radius = 5;
//   const double                 err_bound = 0.00001;
//   Offset_polygon_with_holes_2  offset;
//   Outer_polygon_2 outer;

//   if (radius > 0) {
//     offset = approximated_offset_2 (P, radius, err_bound);
//     outer = offset.outer_boundary();
//   }
//   else {
//     std::list<Offset_polygon_2>            inset_polygons;
//     std::list<Offset_polygon_2>::iterator  iit;
//     approximated_inset_2(P, -radius, err_bound,
// 			 std::back_inserter(inset_polygons));
//     outer = *inset_polygons.begin();
//   }


//   std::cout << "The offset polygon has "
//             << offset.outer_boundary().size() << " vertices, "
//             << offset.number_of_holes() << " holes." << std::endl;

//   return from_mink(outer);
// }

// Polygon_2 close_minkowski(const Polygon_2& p, const Number_type radius)
// {
//   In_polygon_2 P = to_mink(p);

//   // Approximate the offset polygon.
//   const double                 err_bound = 0.00001;
//   Offset_polygon_with_holes_2  offset;
//   Outer_polygon_2 outer;

//   // dilate
//   offset = approximated_offset_2 (P, radius, err_bound);
//   outer = offset.outer_boundary();

//   // erode
//   std::list<Offset_polygon_2>            inset_polygons;
//   std::list<Offset_polygon_2>::iterator  iit;
//   approximated_inset_2(P, -radius, err_bound,
// 		       std::back_inserter(inset_polygons));
//     outer = *inset_polygons.begin();
//   }


//   std::cout << "The offset polygon has "
//             << offset.outer_boundary().size() << " vertices, "
//             << offset.number_of_holes() << " holes." << std::endl;

//   Polygon_2 ret;
//   for (Outer_polygon_2::Curve_iterator it = outer.curves_begin(); it != outer.curves_end(); ++it) {
//     Outer_polygon_2::X_monotone_curve_2 c = *it;
// //     ret.push_back(c);
//     Outer_polygon_2::Point_2 source = c.source();
//     ret.push_back(Point_2(to_double(source.x()), to_double(source.y())));
//   }

// //   for (Polygon_2::Vertex_iterator it = p.vertices_begin(); it != p.vertices_end(); ++it) {
// //     ret.push_back(Point_2(it->x(), it->y()));
// //   }
//   return ret;
// }

CONTOURTILER_END_NAMESPACE
