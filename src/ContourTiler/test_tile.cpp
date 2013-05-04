#include <string>
#include <vector>
#include <sstream>

#include <ContourTiler/test_common.h>
#include <ContourTiler/reader_gnuplot.h>
#include <ContourTiler/contour_utils.h>
#include <ContourTiler/tiler_operations.h>
#include <ContourTiler/Hierarchy.h>
#include <ContourTiler/print_utils.h>

log4cplus::Logger logger_test_tile = log4cplus::Logger::getInstance("tiler.test_tile");

struct Tile_fixture
{
  Tile_fixture(const string& file1, const string& file2)
  {
    vector<Contour_handle> temp1, temp2;
    read_contours_gnuplot2(file1, back_inserter(temp1), 1);
    read_contours_gnuplot2(file2, back_inserter(temp2), 2);

    bottom_orig = temp1;
    top_orig = temp2;

    workspace = pre_tile(bottom_orig.begin(), bottom_orig.end(), top_orig.begin(), top_orig.end(), true);
  }

  vector<Contour_handle> top_orig, bottom_orig;
  boost::shared_ptr<Tiler_workspace> workspace;
};

struct Fixture12 : Tile_fixture
{
  Fixture12() : Tile_fixture(data_dir+"/test1.dat", data_dir+"/test2.dat") {}
};

struct Fixture34 : Tile_fixture
{
  Fixture34() : Tile_fixture(data_dir+"/test3.dat", data_dir+"/test4.dat") {}
};

struct Fixture56 : Tile_fixture
{
  Fixture56() : Tile_fixture(data_dir+"/test5.dat", data_dir+"/test6.dat") {}
};

struct Fixture9_10 : Tile_fixture
{
  Fixture9_10() : Tile_fixture(data_dir+"/test9.dat", data_dir+"/test10.dat") {}
};

struct Fixture11_12 : Tile_fixture
{
  Fixture11_12() : Tile_fixture(data_dir+"/test11.dat", data_dir+"/test12.dat") {}
};

struct Fixture13_14 : Tile_fixture
{
  Fixture13_14() : Tile_fixture(data_dir+"/test13.dat", data_dir+"/test14.dat") {}
};

TEST_F (Fixture12, augment1)
{
  vector<Contour_handle> top, bottom, all;
  augment(top_orig.begin(), top_orig.end(), bottom_orig.begin(), bottom_orig.end(), back_inserter(top), back_inserter(bottom));

  CHECK_EQUAL(1, bottom_orig.size());
  CHECK_EQUAL(4, top_orig.size());

  CHECK_EQUAL(4, bottom_orig[0]->polygon().size());
  CHECK_EQUAL(6, bottom[0]->polygon().size());

  CHECK_EQUAL(Point_2(0, 0, 0), bottom_orig[0]->polygon()[0]);
  CHECK_EQUAL(Point_2(10, 0, 0), bottom_orig[0]->polygon()[1]);
  CHECK_EQUAL(Point_2(10, 10, 0), bottom_orig[0]->polygon()[2]);
  CHECK_EQUAL(Point_2(0, 10, 0), bottom_orig[0]->polygon()[3]);

  CHECK_EQUAL(Point_2(0, 0, 0), bottom[0]->polygon()[0]);
  CHECK_EQUAL(Point_2(10, 0, 0), bottom[0]->polygon()[1]);
  CHECK_EQUAL(Point_2(10, 1, 0), bottom[0]->polygon()[2]);
  CHECK_EQUAL(Point_2(10, 10, 0), bottom[0]->polygon()[3]);
  CHECK_EQUAL(Point_2(1, 10, 0), bottom[0]->polygon()[4]);
  CHECK_EQUAL(Point_2(0, 10, 0), bottom[0]->polygon()[5]);

  CHECK_EQUAL(Point_2(1, 1, 0), top_orig[0]->polygon()[0]);
  CHECK_EQUAL(Point_2(11, 1, 0), top_orig[0]->polygon()[1]);
  CHECK_EQUAL(Point_2(11, 11, 0), top_orig[0]->polygon()[2]);
  CHECK_EQUAL(Point_2(1, 11, 0), top_orig[0]->polygon()[3]);

  CHECK_EQUAL(Point_2(1, 1, 0), top[0]->polygon()[0]);
  CHECK_EQUAL(Point_2(10, 1, 0), top[0]->polygon()[1]);
  CHECK_EQUAL(Point_2(11, 1, 0), top[0]->polygon()[2]);
  CHECK_EQUAL(Point_2(11, 11, 0), top[0]->polygon()[3]);
  CHECK_EQUAL(Point_2(1, 11, 0), top[0]->polygon()[4]);
  CHECK_EQUAL(Point_2(1, 10, 0), top[0]->polygon()[5]);
}

// tests theorem 6.1 and 6.2
TEST_F (Fixture34, can_tile1)
{
  vector<Contour_handle> top, bottom, all;
  augment(top_orig.begin(), top_orig.end(), bottom_orig.begin(), bottom_orig.end(), back_inserter(top), back_inserter(bottom));

  Hierarchy top_orig_h = Hierarchy(top_orig.begin(), top_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_orig_h = Hierarchy(bottom_orig.begin(), bottom_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy top_h = Hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_h = Hierarchy(bottom.begin(), bottom.end(), Hierarchy_policy::FORCE_CCW);

  // intersection
  CHECK(can_tile(bottom[0], top[0], bottom_h, top_h));
  // intersection
  CHECK(can_tile(bottom[1], top[0], bottom_h, top_h));
  // disjoint, different orientations, both have negative vertices
  // top has no vertex V with NEC(V') == NEC(bottom)
  CHECK(!can_tile(bottom_orig[2], top_orig[0], bottom_orig_h, top_orig_h));
  // after augmenting, top now has V st NEC(V') == NEC(bottom)
  CHECK(can_tile(bottom[2], top[0], bottom_h, top_h));

  // intersection
  CHECK(can_tile(bottom[0], top[1], bottom_h, top_h));
  CHECK(can_tile(bottom[1], top[1], bottom_h, top_h));
  // disjoint, different orientations, both have negative vertices,
  // both are non-insulated
  CHECK(can_tile(bottom[2], top[1], bottom_h, top_h));

  // disjoint, same orientation
  CHECK(!can_tile(bottom[0], top[2], bottom_h, top_h));
  // disjoint, different orientations, top has no negative vertex
  CHECK(!can_tile(bottom[1], top[2], bottom_h, top_h));
  // disjoint, different orientations, top has no negative vertex
  CHECK(!can_tile(bottom[2], top[2], bottom_h, top_h));
}

// tests theorem 6.3
// TEST_F (Fixture56, can_tile2)
// {
//   vector<Contour_handle> top, bottom, all;
//   augment(top_orig.begin(), top_orig.end(), bottom_orig.begin(), bottom_orig.end(), back_inserter(top), back_inserter(bottom));

//   Hierarchy top_orig_h = Hierarchy(top_orig.begin(), top_orig.end(), Hierarchy_policy::FORCE_CCW);
//   Hierarchy bottom_orig_h = Hierarchy(bottom_orig.begin(), bottom_orig.end(), Hierarchy_policy::FORCE_CCW);
//   Hierarchy top_h = Hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
//   Hierarchy bottom_h = Hierarchy(bottom.begin(), bottom.end(), Hierarchy_policy::FORCE_CCW);

//   CHECK(!can_tile(bottom[0], top[0], bottom_h, top_h));
//   CHECK(!can_tile(bottom[1], top[0], bottom_h, top_h));
//   CHECK(can_tile(bottom[2], top[0], bottom_h, top_h));

//   // top[1] is completely insulated so it can't be tiled
//   CHECK(!can_tile(bottom[0], top[1], bottom_h, top_h));
//   CHECK(!can_tile(bottom[1], top[1], bottom_h, top_h));
//   CHECK(!can_tile(bottom[2], top[1], bottom_h, top_h));

//   // top[2] is also completely insulated
//   CHECK(!can_tile(bottom[0], top[2], bottom_h, top_h));
//   CHECK(!can_tile(bottom[1], top[2], bottom_h, top_h));
//   CHECK(!can_tile(bottom[2], top[2], bottom_h, top_h));

//   CHECK(!can_tile(top[0], bottom[0], top_h, bottom_h));
//   CHECK(!can_tile(top[0], bottom[1], top_h, bottom_h));
//   CHECK(can_tile(top[0], bottom[2], top_h, bottom_h));

//   CHECK(!can_tile(top[1], bottom[0], top_h, bottom_h));
//   CHECK(!can_tile(top[1], bottom[1], top_h, bottom_h));
//   CHECK(!can_tile(top[1], bottom[2], top_h, bottom_h));

//   CHECK(!can_tile(top[2], bottom[0], top_h, bottom_h));
//   CHECK(!can_tile(top[2], bottom[1], top_h, bottom_h));
//   CHECK(!can_tile(top[2], bottom[2], top_h, bottom_h));
// }

TEST_F (Fixture12, correspondences1)
{
  CHECK_EQUAL(1, workspace->correspondences.count((workspace->bottom)[0]));
  CHECK_EQUAL(1, workspace->correspondences.count((workspace->top)[0]));
  CHECK_EQUAL(0, workspace->correspondences.count((workspace->top)[1]));
  CHECK_EQUAL(0, workspace->correspondences.count((workspace->top)[2]));
  CHECK_EQUAL(0, workspace->correspondences.count((workspace->top)[3]));
}

TEST_F (Fixture34, correspondences2)
{
  CHECK_EQUAL(2, workspace->correspondences.count((workspace->bottom)[0]));
  CHECK_EQUAL(2, workspace->correspondences.count((workspace->bottom)[1]));
  CHECK_EQUAL(4, workspace->correspondences.count((workspace->top)[0]));
  CHECK_EQUAL(3, workspace->correspondences.count((workspace->top)[1]));
}

TEST_F (Fixture34, tiling_region1)
{
  vector<Contour_handle> top, bottom, all;
  augment(top_orig.begin(), top_orig.end(), bottom_orig.begin(), bottom_orig.end(), back_inserter(top), back_inserter(bottom));

  Hierarchy top_orig_h = Hierarchy(top_orig.begin(), top_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_orig_h = Hierarchy(bottom_orig.begin(), bottom_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy top_h = Hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_h = Hierarchy(bottom.begin(), bottom.end(), Hierarchy_policy::FORCE_CCW);

//   Tiling_region region = tiling_region(top_orig[0]->polygon().vertices_circulator(), bottom_orig_h);
//   CHECK(!region.contains(Point_2(0, 0, 0)));
//   CHECK(!region.contains(Point_2(4, 8, 0)));
//   CHECK(!region.contains(Point_2(6, 2, 0)));
//   CHECK(region.contains(Point_2(5, 5, 0)));
//   CHECK(region.contains(Point_2(6, 6, 0)));
}

TEST_F (Fixture34, tiling_region2)
{
  vector<Contour_handle> top, bottom, all;
  augment(top_orig.begin(), top_orig.end(), bottom_orig.begin(), bottom_orig.end(), back_inserter(top), back_inserter(bottom));

  Hierarchy top_orig_h = Hierarchy(top_orig.begin(), top_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_orig_h = Hierarchy(bottom_orig.begin(), bottom_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy top_h = Hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_h = Hierarchy(bottom.begin(), bottom.end(), Hierarchy_policy::FORCE_CCW);

//   Tiling_region region = tiling_region(top_orig[0]->polygon().vertices_circulator()+1, bottom_orig_h);
//   CHECK(!region.contains(Point_2(0, 0, 0)));
//   CHECK(region.contains(Point_2(0, 10, 0)));
//   CHECK(region.contains(Point_2(4, 8, 0)));
//   CHECK(!region.contains(Point_2(6, 2, 0)));
//   CHECK(region.contains(Point_2(5, 5, 0)));
//   CHECK(region.contains(Point_2(6, 6, 0)));
//   CHECK(!region.contains(Point_2(10, 6, 0)));
}

TEST_F (Fixture34, tiling_region3)
{
  vector<Contour_handle> top, bottom, all;
  augment(top_orig.begin(), top_orig.end(), bottom_orig.begin(), bottom_orig.end(), back_inserter(top), back_inserter(bottom));

  Hierarchy top_orig_h = Hierarchy(top_orig.begin(), top_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_orig_h = Hierarchy(bottom_orig.begin(), bottom_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy top_h = Hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_h = Hierarchy(bottom.begin(), bottom.end(), Hierarchy_policy::FORCE_CCW);

//   Tiling_region region = tiling_region(top_orig[0]->polygon().vertices_circulator()+2, bottom_orig_h);
//   CHECK(!region.contains(Point_2(0, 0, 0)));
//   CHECK(region.contains(Point_2(0, 10, 0)));
//   CHECK(!region.contains(Point_2(4, 8, 0)));
//   CHECK(!region.contains(Point_2(6, 2, 0)));
//   CHECK(!region.contains(Point_2(5, 5, 0)));
//   CHECK(!region.contains(Point_2(6, 6, 0)));
//   CHECK(region.contains(Point_2(10, 6, 0)));
//   CHECK(region.contains(Point_2(10, 0, 0)));
//   CHECK(region.contains(Point_2(5, 10, 0)));
}

TEST_F (Fixture34, tiling_region4)
{
  vector<Contour_handle> top, bottom, all;
  augment(top_orig.begin(), top_orig.end(), bottom_orig.begin(), bottom_orig.end(), back_inserter(top), back_inserter(bottom));

  Hierarchy top_orig_h = Hierarchy(top_orig.begin(), top_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_orig_h = Hierarchy(bottom_orig.begin(), bottom_orig.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy top_h = Hierarchy(top.begin(), top.end(), Hierarchy_policy::FORCE_CCW);
  Hierarchy bottom_h = Hierarchy(bottom.begin(), bottom.end(), Hierarchy_policy::FORCE_CCW);

  Contour_handle contour = top[0];
  const Polygon_2& polygon = contour->polygon();
//   Tiling_region region = tiling_region(polygon.vertices_circulator()+2, bottom_h);
//   CHECK_EQUAL(region, workspace->tiling_regions[workspace->vertices.get(contour, 2)]);
}

TEST (Point_sorting1)
{
  Point_2 point(3, 3, 0);
  vector<Point_2> points;
  points.push_back(Point_2(6, 3, 0));
  points.push_back(Point_2(2, 6, 0));
  points.push_back(Point_2(4, 1, 0));
  points.push_back(Point_2(2, 2, 0));

  std::sort(points.begin(), points.end());
  CHECK_EQUAL(Point_2(2, 2, 0), points[0]);
  CHECK_EQUAL(Point_2(2, 6, 0), points[1]);
  CHECK_EQUAL(Point_2(4, 1, 0), points[2]);
  CHECK_EQUAL(Point_2(6, 3, 0), points[3]);

  std::sort(points.begin(), points.end(), Dist_cmp(point));
  CHECK_EQUAL(Point_2(2, 2, 0), points[0]);
  CHECK_EQUAL(Point_2(4, 1, 0), points[1]);
  CHECK_EQUAL(Point_2(6, 3, 0), points[2]);
  CHECK_EQUAL(Point_2(2, 6, 0), points[3]);
}

TEST_F (Fixture34, Optimality1)
{
//   Tiler_workspace w(all, correspondences, vertices, tiling_regions, otv_table, hierarchies);
  Tiler_workspace& w = *workspace;
  Vertices& vertices = w.vertices;
  const std::vector<Contour_handle>& bottom = (workspace->bottom);
  const std::vector<Contour_handle>& top = (workspace->top);

  Point_3 u2 = vertices.get((workspace->bottom)[0], 2), u3 = vertices.get((workspace->bottom)[0], 3);
  Point_3 v2 = vertices.get((workspace->top)[0], 2), v3 = vertices.get((workspace->top)[0], 3);
  CHECK_EQUAL(6, optimality(u2, u3, v2, v3, w));

//   u2 = vertices.get(bottom[0], 6);
//   u3 = vertices.get(bottom[0], 5);
//   v2 = vertices.get(top[0], 0);
//   v3 = vertices.get(top[0], 9);
//   CHECK_EQUAL(2, optimality(u2, u3, v2, v3, w));

//   u2 = vertices.get(bottom[0], 6);
//   u3 = vertices.get(bottom[0], 7);
//   v2 = vertices.get(top[0], 0);
//   v3 = vertices.get(top[0], 9);
//   CHECK_EQUAL(2, optimality(u2, u3, v2, v3, w));

  u2 = vertices.get(bottom[0], 7);
  u3 = vertices.get(bottom[0], 6);
  v2 = vertices.get(top[0], 0);
  v3 = vertices.get(top[0], 9);
  CHECK_EQUAL(5, optimality(u2, u3, v2, v3, w));

  u2 = vertices.get(bottom[2], 0);
  u3 = vertices.get(bottom[2], 1);
  v2 = vertices.get(top[0], 0);
  v3 = vertices.get(top[0], 9);
  CHECK_EQUAL(6, optimality(u2, u3, v2, v3, w));
}

bool intermediate(const Tiler_workspace& workspace)
{
  static int count = 0;
  std::stringstream ss;
  ss << out_dir << "/out_tiles_" << setfill('0') << setw(2) << count << ".g";
  ofstream out(ss.str().c_str());
  gnuplot_print_tiles(out, workspace.tiles_begin(), workspace.tiles_end());
  out.close();
  LOG4CPLUS_TRACE(logger_test_tile, "intermediate: " << count);
  ++count;
  return true;
}
