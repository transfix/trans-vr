#include <string>
#include <vector>

#include <ContourTiler/test_common.h>
#include <ContourTiler/reader_gnuplot.h>
#include <ContourTiler/contour_utils.h>
#include <ContourTiler/tiler_operations.h>
#include <ContourTiler/polygon_utils.h>
#include <ContourTiler/Hierarchy.h>

TEST (simple1)
{
  CHECK(Point_2(3, 4) != Point_2(4, 5));
}

TEST (distance1)
{
  Polygon_2 P;
  P.push_back(Point_2(0, 0));
  P.push_back(Point_2(1, 0));
  P.push_back(Point_2(2, 0));
  P.push_back(Point_2(2, 1));
  P.push_back(Point_2(2, 2));
  P.push_back(Point_2(1, 2));
  P.push_back(Point_2(0, 2));
  P.push_back(Point_2(0, 1));
  CHECK_EQUAL(0, distance(P[0], P[0], P));
  CHECK_EQUAL(1, distance(P[1], P[0], P));
  CHECK_EQUAL(1, distance(P[0], P[1], P));
  CHECK_EQUAL(1, distance(P[0], P[7], P));
  CHECK_EQUAL(1, distance(P[7], P[0], P));
  CHECK_EQUAL(4, distance(P[0], P[4], P));
  CHECK_EQUAL(-1, distance(P[0], Point_2(3,3), P));
  CHECK_EQUAL(-1, distance(Point_2(3,3), Point_2(3,3), P));
}

TEST (intersects_boundary1)
{
  Polygon_2 polygon;
  polygon.push_back(Point_2(0,  0));
  polygon.push_back(Point_2(10, 0));
  polygon.push_back(Point_2(10, 10));
  polygon.push_back(Point_2(0,  10));
  CHECK(intersects_boundary(Point_2(0, 0), polygon));
  CHECK(intersects_boundary(Point_2(5, 0), polygon));
  CHECK(!intersects_boundary(Point_2(1, 1), polygon));
}

TEST (intersects_boundary2)
{
  vector<Contour_handle> contours;
  read_contours_gnuplot2(data_dir+"/test7.dat", back_inserter(contours), 1);

  Contour_handle contour = contours[0];
  const Polygon_2& polygon = contour->polygon();

  CHECK(!intersects_proper(Segment_2(Point_2(0,0),Point_2(1,0)), polygon));
  CHECK(!intersects_proper(Segment_2(Point_2(2,5),Point_2(7,0)), polygon));
  CHECK( intersects_proper(Segment_2(Point_2(1,0),Point_2(1,2)), polygon));
  CHECK(!intersects_proper(Segment_2(Point_2(0,4),Point_2(2,4)), polygon));
  CHECK(!intersects_proper(Segment_2(Point_2(0,4),Point_2(4,4)), polygon));
  CHECK(!intersects_proper(Segment_2(Point_2(0,2),Point_2(1,2)), polygon));
  CHECK(!intersects_proper(Segment_2(Point_2(1,2),Point_2(3,2)), polygon));

  CHECK(intersects_proper(Segment_2(Point_2(0,3),Point_2(4,-1)), polygon));
  CHECK(intersects_proper(Segment_2(Point_2(0,3),Point_2(1,2)), polygon));
  CHECK(intersects_proper(Segment_2(Point_2(4,-1),Point_2(0,3)), polygon));
  CHECK(!intersects_proper(Segment_2(Point_2(0,0),Point_2(2,2)), polygon));
  CHECK(!intersects_proper(Segment_2(Point_2(0,0),Point_2(5,0)), polygon));
}
