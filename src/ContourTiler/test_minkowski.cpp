//! \file examples/Minkowski_sum_2/approx_offset.cpp
// Computing the approximated offset of a polygon.

#include <CGAL/basic.h>

// #ifdef CGAL_USE_GMP

//   // GMP is installed. Use the GMP rational number-type.
//   #include <CGAL/Gmpq.h>

//   typedef CGAL::Gmpq                                    Number_type;

// #else

//   // GMP is not installed. Use CGAL's exact rational number-type.
//   #include <CGAL/MP_Float.h>
//   #include <CGAL/Quotient.h>

//   typedef CGAL::Quotient<CGAL::MP_Float>                Number_type;

// #endif

#include <CGAL/Cartesian.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Timer.h>
#include <CGAL/approximated_offset_2.h>
#include <CGAL/offset_polygon_2.h>
#include <ContourTiler/test_common.h>
#include <iostream>

// typedef CGAL::Lazy_exact_nt<Number_type>           Lazy_exact_nt;
typedef CGAL::Lazy_exact_nt<CGAL::Gmpq> Lazy_exact_nt;

struct Offset_kernel : public CGAL::Cartesian<Lazy_exact_nt> {};
typedef CGAL::Polygon_2<Offset_kernel> In_Polygon_2;

typedef CGAL::Gps_circle_segment_traits_2<Offset_kernel> Gps_traits_2;
typedef Gps_traits_2::Polygon_2 Offset_polygon_2;
typedef Gps_traits_2::Polygon_with_holes_2 Offset_polygon_with_holes_2;

TEST(mink1) {
  // Open the input file.
  std::ifstream in_file("../test_data/spiked.dat");

  if (!in_file.is_open()) {
    std::cerr << "Failed to open the input file." << std::endl;
    return;
  }

  // Read the input polygon.
  In_Polygon_2 P;

  in_file >> P;
  in_file.close();

  std::cout << "Read an input polygon with " << P.size() << " vertices."
            << std::endl;

  // Approximate the offset polygon.
  const Number_type radius = 5;
  const double err_bound = 0.00001;
  Offset_polygon_with_holes_2 offset;
  CGAL::Timer timer;

  timer.start();
  offset = approximated_offset_2(P, radius, err_bound);
  timer.stop();

  std::cout << "The offset polygon has " << offset.outer_boundary().size()
            << " vertices, " << offset.number_of_holes() << " holes."
            << std::endl;
  std::cout << "Offset computation took " << timer.time() << " seconds."
            << std::endl;
  return;
}
