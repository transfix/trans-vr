#include <ContourTiler/Hierarchy.h>
#include <ContourTiler/contour_utils.h>
#include <ContourTiler/reader_gnuplot.h>
#include <ContourTiler/test_common.h>
#include <ContourTiler/tiler_operations.h>
#include <string>
#include <vector>

struct Basic_fixture {
  Basic_fixture() {
    read_contours_gnuplot2(data_dir + "/test1.dat", back_inserter(bottom), 1);
    read_contours_gnuplot2(data_dir + "/test2.dat", back_inserter(top), 2);
  }

  vector<Contour_handle> top, bottom;
};

TEST_F(Basic_fixture, Hierarchy1) {
  Hierarchy hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
  for (vector<Contour_handle>::iterator it = top.begin(); it != top.end();
       ++it)
    CHECK((*it)->is_counterclockwise_oriented());

  CHECK(hierarchy.is_CCW(top[0]));
  CHECK(hierarchy.is_CW(top[1]));
  CHECK(hierarchy.is_CCW(top[2]));
  CHECK(hierarchy.is_CW(top[3]));

  CHECK_EQUAL(Contour_handle(), hierarchy.NEC(top[0]));
  CHECK_EQUAL(top[0], hierarchy.NEC(top[1]));
  CHECK_EQUAL(top[1], hierarchy.NEC(top[2]));
  CHECK_EQUAL(top[0], hierarchy.NEC(top[3]));
}

TEST_F(Basic_fixture, NEC1) {
  using namespace boost;

  Hierarchy hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
  CHECK(get<0>(hierarchy.is_overlapping(Point_2(1, 1))));
  CHECK(get<0>(hierarchy.is_overlapping(Point_2(2, 2))));
  CHECK(!get<0>(hierarchy.is_overlapping(Point_2(1.5, 1.5))));
  CHECK(!get<0>(hierarchy.is_overlapping(Point_2(1, 0))));
}
