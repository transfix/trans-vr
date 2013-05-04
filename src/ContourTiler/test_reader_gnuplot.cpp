#include <list>
#include <vector>
#include <string>

#include <ContourTiler/test_common.h>
#include <ContourTiler/reader_gnuplot.h>

TEST (reader_gnuplot1)
{
  list<Polygon_2> polygons;
  read_polygons_gnuplot2(data_dir+"/test1.dat", back_inserter(polygons), 1);
  CHECK_EQUAL(polygons.size(), 1);

  Polygon_2 polygon = *polygons.begin();
  CHECK_EQUAL(4, polygon.size());
  CHECK_EQUAL(Point_2(0, 0), polygon[0]);
  CHECK_EQUAL(Point_2(10, 0), polygon[1]);
  CHECK_EQUAL(Point_2(10, 10), polygon[2]);
  CHECK_EQUAL(Point_2(0, 10), polygon[3]);
}

TEST (reader_gnuplot2)
{
  vector<Polygon_2> polygons;
  read_polygons_gnuplot2(data_dir+"/test2.dat", back_inserter(polygons), 1);
  CHECK_EQUAL(4, polygons.size());

  CHECK_EQUAL(4, polygons[0].size());
  CHECK_EQUAL(3, polygons[1].size());
  CHECK_EQUAL(3, polygons[2].size());
  CHECK_EQUAL(6, polygons[3].size());

  Polygon_2 polygon = polygons[3];
  CHECK_EQUAL(6, polygon.size());
  CHECK_EQUAL(Point_2(7, 4), polygon[0]);
  CHECK_EQUAL(Point_2(7, 6), polygon[1]);
  CHECK_EQUAL(Point_2(5, 8), polygon[2]);
  CHECK_EQUAL(Point_2(8, 8), polygon[3]);
  CHECK_EQUAL(Point_2(8, 2), polygon[4]);
  CHECK_EQUAL(Point_2(5, 2), polygon[5]);
}
