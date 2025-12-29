#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <string>
// #include <CGAL/partition_2.h>

#include <ContourTiler/cut.h>
#include <ContourTiler/intersection.h>
#include <ContourTiler/polygon_utils.h>
#include <ContourTiler/print_utils.h>
#include <ContourTiler/reader_gnuplot.h>
#include <ContourTiler/set_utils.h>
#include <ContourTiler/test_common.h>
#include <ContourTiler/test_intersection_output.h>
#include <ContourTiler/test_utils.h>
#include <ContourTiler/triangle_utils.h>

typedef boost::shared_ptr<Triangle> Triangle_handle;

Triangle_handle tile(const Point_3 &a, const Point_3 &b, const Point_3 &c) {
  return Triangle_handle(new Triangle(a, b, c));
}

// void test_it(list<Triangle_handle>& yellow, list<Triangle_handle>& green,
// Number_type yz, Number_type gz, const std::string& test_num)
// {
//   list<Triangle_handle>& ytriangles = yellow;
//   list<Triangle_handle>& gtriangles = green;
//   vector<Triangle_handle> new_yellow, new_green;
//   remove_intersections(ytriangles.begin(), ytriangles.end(),
// 		       gtriangles.begin(), gtriangles.end(),
// 		       yz, gz,
// 		       back_inserter(new_yellow), back_inserter(new_green),
// 		       cl_delta);

//   list<Colored_point_3> all;
//   for (vector<Triangle_handle>::iterator it = new_yellow.begin(); it !=
//   new_yellow.end(); ++it)
//     for (int i = 0; i < 3; ++i)
//       all.push_back(Colored_point_3(vertex(i, **it), 1, 1, 0));
//   for (vector<Triangle_handle>::iterator it = new_green.begin(); it !=
//   new_green.end(); ++it)
//     for (int i = 0; i < 3; ++i)
//       all.push_back(Colored_point_3(vertex(i, **it), 0, 1, 0));

//   ofstream out(("output/test" + test_num + "aa.rawc").c_str());
//   raw_print_tiles_impl(out, all.begin(), all.end(), 1, true);
//   out.close();

//   ofstream yr(("output/test" + test_num + "aayellow.rawc").c_str());
//   raw_print_tiles(yr, new_yellow.begin(), new_yellow.end(), 1, 1, 0);
//   yr.close();

//   ofstream gr(("output/test" + test_num + "aagreen.rawc").c_str());
//   raw_print_tiles(gr, new_green.begin(), new_green.end(), 0, 1, 0);
//   gr.close();

//   //
//   // Output original
//   //
//   list<Colored_point_3> orig;
//   for (list<Triangle_handle>::iterator it = yellow.begin(); it !=
//   yellow.end(); ++it)
//     for (int i = 0; i < 3; ++i)
//       orig.push_back(Colored_point_3(vertex(i, **it), 1, 1, 0));
//   for (list<Triangle_handle>::iterator it = green.begin(); it !=
//   green.end(); ++it)
//     for (int i = 0; i < 3; ++i)
//       orig.push_back(Colored_point_3(vertex(i, **it), 0, 1, 0));

//   ofstream orig_out(("output/test" + test_num + "zz.rawc").c_str());
//   raw_print_tiles_impl(orig_out, orig.begin(), orig.end(), 1, true);
//   orig_out.close();

//   ofstream orig_yg(("output/test" + test_num + "zzyellow.rawc").c_str());
//   raw_print_tiles(orig_yg, yellow.begin(), yellow.end(), 1, 1, 0);
//   orig_yg.close();

//   ofstream orig_gg(("output/test" + test_num + "zzgreen.rawc").c_str());
//   raw_print_tiles(orig_gg, green.begin(), green.end(), 0, 1, 0);
//   orig_gg.close();
// //   mtest_output(new_yellow, new_green, ytriangles, gtriangles);
// }

void test_it(list<Triangle_handle> &yellow, list<Triangle_handle> &green,
             Number_type yz, Number_type gz, const string &expected,
             vector<Triangle> &a, vector<Triangle> &b,
             const string &test_num) {
  //   list<Triangle_handle>& ytriangles = yellow;
  //   list<Triangle_handle>& gtriangles = green;
  //   vector<Triangle_handle> new_yellow, new_green;
  //   remove_intersections(ytriangles.begin(), ytriangles.end(),
  // 		       gtriangles.begin(), gtriangles.end(),
  // 		       yz, gz,
  // 		       back_inserter(new_yellow), back_inserter(new_green),
  // 		       cl_delta);
  // // 		       fabs(yz-gz)/DIV);

  //   list<Colored_point_3> orig_all_c, all_c;
  //   for (vector<Triangle_handle>::iterator it = new_yellow.begin(); it !=
  //   new_yellow.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       all_c.push_back(Colored_point_3(vertex(i, **it), 1, 1, 0));
  //   for (vector<Triangle_handle>::iterator it = new_green.begin(); it !=
  //   new_green.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       all_c.push_back(Colored_point_3(vertex(i, **it), 0, 1, 0));

  //   for (list<Triangle_handle>::iterator it = yellow.begin(); it !=
  //   yellow.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       orig_all_c.push_back(Colored_point_3(vertex(i, **it), 1, 1, 0));
  //   for (list<Triangle_handle>::iterator it = green.begin(); it !=
  //   green.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       orig_all_c.push_back(Colored_point_3(vertex(i, **it), 0, 1, 0));

  //   stringstream ssa(expected), ssb;
  //   raw_print_tiles_impl(ssb, all_c.begin(), all_c.end(), 1, true);

  //   // output to file
  //   ofstream out(("output/test" + test_num + ".rawc").c_str());
  //   raw_print_tiles_impl(out, all_c.begin(), all_c.end(), 1, true);
  //   out.close();

  //   // output to file
  //   ofstream oout(("output/testorig" + test_num + ".rawc").c_str());
  //   raw_print_tiles_impl(oout, orig_all_c.begin(), orig_all_c.end(), 1,
  //   true); oout.close();

  //   a.clear();
  //   b.clear();
  //   if (!expected.empty())
  //     normalized_raw(ssa, true, back_inserter(a));
  //   normalized_raw(ssb, true, back_inserter(b));

  //   //
  //   // Output original
  //   //
  //   list<Colored_point_3> orig;
  //   for (list<Triangle_handle>::iterator it = yellow.begin(); it !=
  //   yellow.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       orig.push_back(Colored_point_3(vertex(i, **it), 1, 1, 0));
  //   for (list<Triangle_handle>::iterator it = green.begin(); it !=
  //   green.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       orig.push_back(Colored_point_3(vertex(i, **it), 0, 1, 0));

  //   ofstream orig_out(("output/test" + test_num + "zz.rawc").c_str());
  //   raw_print_tiles_impl(orig_out, orig.begin(), orig.end(), 1, true);
  //   orig_out.close();

  //   ofstream orig_yg(("output/test" + test_num + "zzyellow.rawc").c_str());
  //   raw_print_tiles(orig_yg, yellow.begin(), yellow.end(), 1, 1, 0);
  //   orig_yg.close();

  //   ofstream orig_gg(("output/test" + test_num + "zzgreen.rawc").c_str());
  //   raw_print_tiles(orig_gg, green.begin(), green.end(), 0, 1, 0);
  //   orig_gg.close();
  // //   mtest_output(new_yellow, new_green, ytriangles, gtriangles);
}

pair<string, string> test_it(list<Triangle_handle> &yellow,
                             list<Triangle_handle> &green, Number_type yz,
                             Number_type gz, const string &expected,
                             const string &test_num) {
  //   list<Triangle_handle>& ytriangles = yellow;
  //   list<Triangle_handle>& gtriangles = green;
  //   vector<Triangle_handle> new_yellow, new_green;
  //   remove_intersections(ytriangles.begin(), ytriangles.end(),
  // 		       gtriangles.begin(), gtriangles.end(),
  // 		       yz, gz,
  // 		       back_inserter(new_yellow), back_inserter(new_green),
  // 		       cl_delta);
  // // 		       fabs(yz-gz)/DIV);

  //   list<Colored_point_3> all_c;
  //   for (vector<Triangle_handle>::iterator it = new_yellow.begin(); it !=
  //   new_yellow.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       all_c.push_back(Colored_point_3(vertex(i, **it), 1, 1, 0));
  //   for (vector<Triangle_handle>::iterator it = new_green.begin(); it !=
  //   new_green.end(); ++it)
  //     for (int i = 0; i < 3; ++i)
  //       all_c.push_back(Colored_point_3(vertex(i, **it), 0, 1, 0));

  //   stringstream ssa(expected), ssb;
  //   raw_print_tiles_impl(ssb, all_c.begin(), all_c.end(), 1, true);

  //   // output to file
  //   ofstream out(("output/test" + test_num + ".rawc").c_str());
  //   raw_print_tiles_impl(out, all_c.begin(), all_c.end(), 1, true);
  //   out.close();

  // //   a.clear();
  // //   b.clear();
  //   vector<Triangle> a, b;
  //   normalized_raw(ssa, true, back_inserter(a));
  //   normalized_raw(ssb, true, back_inserter(b));

  //   stringstream outa, outb;
  //   raw_print_tiles(outa, a.begin(), a.end(), 0, 0, 0);
  //   raw_print_tiles(outb, b.begin(), b.end(), 0, 0, 0);
  //   return make_pair(outa.str(), outb.str());
}

#define DO_CHECK()                                                           \
//   good = (a.size() == b.size());		\
  for (int i = 0; i < a.size(); ++i)		\
    for (int j = 0; j < 3; ++j)			\
      if (a[i][j] != b[i][j])			\
      {						\
// 	good = false;				\
        CHECK_EQUAL(a[i][j], b[i][j]);		\
      }						\
//   if (!good)					\
//   {						\
//     CHECK(false);				\
//     cout << "Expected: " << endl;		\
//     raw_print_tiles(cout, a.begin(), a.end(),0,0,0);	\
//     cout << endl << "Actual: " << endl;		\
//     raw_print_tiles(cout, b.begin(), b.end(),0,0,0);	\
//   }						\
                                       \
// lab book page 53
TEST(intersection1) {
  list<Triangle_handle> yellow, green;
  yellow.push_back(
      tile(Point_3(3, 5, 1), Point_3(3, 1, 1), Point_3(5, 3, 2)));

  green.push_back(tile(Point_3(2, 2, 1), Point_3(8, 2, 2), Point_3(2, 2, 2)));
  green.push_back(tile(Point_3(2, 1, 1), Point_3(8, 2, 2), Point_3(2, 2, 1)));
  green.push_back(tile(Point_3(2, 1, 1), Point_3(8, 0, 2), Point_3(8, 2, 2)));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection1_expected, a, b, "1a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection1_expected, a, b, "1b");
  DO_CHECK();
}

TEST(intersection2) {
  list<Triangle_handle> yellow, green;
  yellow.push_back(
      tile(Point_3(3, 5, 1), Point_3(3, -1, 1), Point_3(5, 3, 2)));

  green.push_back(tile(Point_3(2, 2, 1), Point_3(8, 2, 2), Point_3(2, 2, 2)));
  green.push_back(tile(Point_3(2, 1, 1), Point_3(8, 2, 2), Point_3(2, 2, 1)));
  green.push_back(tile(Point_3(2, 1, 1), Point_3(8, 0, 2), Point_3(8, 2, 2)));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection2_expected, a, b, "2a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection2_expected, a, b, "2b");
  DO_CHECK();
}

TEST(intersection3) {
  Point_3 y[] = {
      Point_3(7, 6, 2),   // 0
      Point_3(6, 2.5, 2), // 1
      Point_3(5, 3, 2),   // 2
      Point_3(3, 5, 1),   // 3
      Point_3(3, 1, 1),   // 4
      Point_3(4, -1, 1),  // 5
      Point_3(6, 1.5, 1), // 6
      Point_3(7, -2, 1),  // 7
  };
  list<Triangle_handle> yellow, green;
  yellow.push_back(tile(y[5], y[1], y[6]));

  green.push_back(tile(Point_3(2, 2, 1), Point_3(8, 2, 2), Point_3(2, 2, 2)));
  green.push_back(tile(Point_3(2, 1, 1), Point_3(8, 2, 2), Point_3(2, 2, 1)));
  green.push_back(tile(Point_3(2, 1, 1), Point_3(8, 0, 2), Point_3(8, 2, 2)));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection3_expected, a, b, "3a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection3_expected, a, b, "3b");
  DO_CHECK();
}

TEST(intersection4) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 4, 2), // 0
      Point_3(1, 0, 1), // 1
      Point_3(3, 2, 1), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection4_expected, a, b, "4a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection4_expected, a, b, "4b");
  DO_CHECK();
}

TEST(intersection5) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 4, 2),     // 0
      Point_3(1, 2, 1),     // 1
      Point_3(2.5, 3.5, 2), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection5_expected, a, b, "5a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection5_expected, a, b, "5b");
  DO_CHECK();
}

TEST(intersection6) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 4, 2), // 0
      Point_3(1, 2, 1), // 1
      Point_3(3, 2, 1), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection6_expected, a, b, "6a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection6_expected, a, b, "6b");
  DO_CHECK();
}

TEST(intersection7) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 4, 2), // 0
      Point_3(2, 1, 1), // 1
      Point_3(3, 2, 1), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
      Point_3(4, 0, 2), // 3
  };
  green.push_back(tile(g[0], g[1], g[2]));
  green.push_back(tile(g[0], g[3], g[1]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection7_expected, a, b, "7a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection7_expected, a, b, "7b");
  DO_CHECK();
}

TEST(intersection8) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 4, 2), // 0
      Point_3(2, 1, 1), // 1
      Point_3(2, 3, 1), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
      Point_3(4, 0, 2), // 3
  };
  green.push_back(tile(g[0], g[1], g[2]));
  green.push_back(tile(g[0], g[3], g[1]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection8_expected, a, b, "8a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection8_expected, a, b, "8b");
  DO_CHECK();
}

TEST(intersection9) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 4, 2),   // 0
      Point_3(2, 1, 1),   // 1
      Point_3(3, 1.5, 1), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
      Point_3(4, 0, 2), // 3
  };
  green.push_back(tile(g[0], g[1], g[2]));
  green.push_back(tile(g[0], g[3], g[1]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection9_expected, a, b, "9a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection9_expected, a, b, "9b");
  DO_CHECK();
}

TEST(intersection10) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 4, 2), // 0
      Point_3(2, 1, 1), // 1
      Point_3(3, 1, 1), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
      Point_3(4, 0, 2), // 3
  };
  green.push_back(tile(g[0], g[1], g[2]));
  green.push_back(tile(g[0], g[3], g[1]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection10_expected, a, b, "10a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection10_expected, a, b, "10b");
  DO_CHECK();
}

TEST(intersection11) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),   // 0
      Point_3(1, 0, 1),   // 1
      Point_3(5, 0, 1),   // 2
      Point_3(3, 2, 1.5), // 3
  };
  yellow.push_back(tile(y[0], y[1], y[3]));
  yellow.push_back(tile(y[0], y[3], y[2]));
  yellow.push_back(tile(y[1], y[2], y[3]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1),   // 2
      Point_3(3, 2, 1.5), // 3
  };
  green.push_back(tile(g[0], g[1], g[3]));
  green.push_back(tile(g[0], g[3], g[2]));
  green.push_back(tile(g[1], g[2], g[3]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection11_expected, a, b, "11a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection11_expected, a, b, "11b");
  DO_CHECK();
}

TEST(intersection12) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),   // 0
      Point_3(1, 0, 1),   // 1
      Point_3(5, 0, 1),   // 2
      Point_3(3, 2, 1.5), // 3
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1),   // 2
      Point_3(3, 2, 1.5), // 3
  };
  green.push_back(tile(g[0], g[1], g[3]));
  green.push_back(tile(g[0], g[3], g[2]));
  green.push_back(tile(g[1], g[2], g[3]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection12_expected, a, b, "12a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection12_expected, a, b, "12b");
  DO_CHECK();
}

TEST(intersection13) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 3.5, 2), // 0
      Point_3(2, -2, 1),  // 1
      Point_3(4, -2, 1),  // 2
      Point_3(3, 2, 1.5), // 3
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1),   // 2
      Point_3(3, 2, 1.5), // 3
  };
  green.push_back(tile(g[0], g[1], g[3]));
  green.push_back(tile(g[0], g[3], g[2]));
  green.push_back(tile(g[1], g[2], g[3]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection13_expected, a, b, "13a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection13_expected, a, b, "13b");
  DO_CHECK();
}

TEST(intersection14) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),   // 0
      Point_3(0, 4, 1.5), // 1
      Point_3(3, 3, 1),   // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection14_expected, a, b, "14a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection14_expected, a, b, "14b");
  DO_CHECK();
}

TEST(intersection15) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),    // 0
      Point_3(0, 4, 1.5),  // 1
      Point_3(3, 3, 1.75), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection15_expected, a, b, "15a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection15_expected, a, b, "15b");
  DO_CHECK();
}

TEST(intersection16) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),   // 0
      Point_3(0, 4, 1.5), // 1
      Point_3(3, 2, 1.5), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection16_expected, a, b, "16a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection16_expected, a, b, "16b");
  DO_CHECK();
}

TEST(intersection17) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),   // 0
      Point_3(0, 4, 1.5), // 1
      Point_3(0, 0, 1.5), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection17_expected, a, b, "17a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection17_expected, a, b, "17b");
  DO_CHECK();
}

TEST(intersection18) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),    // 0
      Point_3(-1, 4, 1.5), // 1
      Point_3(1, 2, 1.5),  // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection18_expected, a, b, "18a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection18_expected, a, b, "18b");
  DO_CHECK();
}

TEST(intersection19) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(-1, 5, 2),    // 0
      Point_3(-1, -1, 1.5), // 1
      Point_3(2, 2, 1.5),   // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection19_expected, a, b, "19a");
  DO_CHECK();
  //   test_it(green, yellow, 2, 1, intersection19_expected, a, b, "19b");
  //   DO_CHECK();
}

TEST(intersection20) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),    // 0
      Point_3(0, 4, 1.5),  // 1
      Point_3(3, 2, 1.85), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection20_expected, a, b, "20a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection20_expected, a, b, "20b");
  DO_CHECK();
}

TEST(intersection21) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 5, 2),   // 0
      Point_3(0, 5, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection21_expected, a, b, "21a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection21_expected, a, b, "21b");
  DO_CHECK();
}

TEST(intersection22) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(4, 4, 2),   // 0
      Point_3(2, 4, 2),   // 1
      Point_3(3, 3, 1.5), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(6, 2, 2), // 1
      Point_3(0, 4, 1), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection22_expected, a, b, "22a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection22_expected, a, b, "22b");
  DO_CHECK();
}

TEST(intersection23) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(3, 4, 2),   // 0
      Point_3(1, 0, 1),   // 1
      Point_3(5, 0, 1),   // 2
      Point_3(3, 2, 1.5), // 3
  };
  yellow.push_back(tile(y[1], y[2], y[3]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1),   // 2
      Point_3(3, 2, 1.5), // 3
  };
  green.push_back(tile(g[0], g[1], g[3]));
  green.push_back(tile(g[0], g[3], g[2]));
  green.push_back(tile(g[1], g[2], g[3]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection23_expected, a, b, "23a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection23_expected, a, b, "23b");
  DO_CHECK();
}

TEST(intersection24) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(7, 3, 2),   // 0
      Point_3(0, 4, 1.5), // 1
      Point_3(1, 1, 1.5), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1),   // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection24_expected, a, b, "24a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection24_expected, a, b, "24b");
  DO_CHECK();
}

TEST(intersection25) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(2, 4, 2),   // 0
      Point_3(0, 0, 1.5), // 1
      Point_3(4, 4, 2),   // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection25_expected, a, b, "25a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection25_expected, a, b, "25b");
  DO_CHECK();
}

TEST(intersection26) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(8, 4, 2),   // 0
      Point_3(0, 0, 1.5), // 1
      Point_3(8, 3, 2),   // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection26_expected, a, b, "26a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection26_expected, a, b, "26b");
  DO_CHECK();
}

TEST(intersection27) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(7, 4, 2),   // 0
      Point_3(2, 2, 1.5), // 1
      Point_3(7, 3, 2),   // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(6, 2, 2), // 1
      Point_3(0, 4, 1), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection27_expected, a, b, "27a");
  DO_CHECK();
  test_it(green, yellow, 2, 1, intersection27_expected, a, b, "27b");
  DO_CHECK();
}

TEST(intersection28) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(8, 3, 2),   // 0
      Point_3(1, 2, 1.5), // 1
      Point_3(8, 2.5, 2), // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1), // 0
      Point_3(4, 2, 2), // 1
      Point_3(0, 4, 1), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  string a, b;
  boost::tie(a, b) =
      test_it(yellow, green, 1, 2, intersection28_expected, "28a");
  CHECK_EQUAL(a, b);
  boost::tie(a, b) =
      test_it(green, yellow, 2, 1, intersection28_expected, "28b");
  CHECK_EQUAL(a, b);
}

TEST(intersection29) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(2, 2, 1.6), // 0
      Point_3(1, 0, 1),   // 1
      Point_3(3, 0, 1),   // 2
      Point_3(3, 2, 1),   // 3
      Point_3(3, 4, 1),   // 4
      Point_3(1, 3, 1),   // 5
      Point_3(-1, 2, 1),  // 6
      Point_3(-1, 1, 1),  // 7
  };
  yellow.push_back(tile(y[0], y[1], y[2]));
  yellow.push_back(tile(y[0], y[2], y[3]));
  yellow.push_back(tile(y[0], y[3], y[4]));
  yellow.push_back(tile(y[0], y[4], y[5]));
  yellow.push_back(tile(y[0], y[5], y[6]));
  yellow.push_back(tile(y[0], y[6], y[7]));
  yellow.push_back(tile(y[0], y[7], y[1]));

  Point_3 g[] = {
      Point_3(0, 0, 2), // 0
      Point_3(4, 2, 1), // 1
      Point_3(0, 4, 2), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection29_expected, a, b, "29a");
  //   DO_CHECK();
  test_it(green, yellow, 2, 1, intersection29_expected, a, b, "29b");
  //   DO_CHECK();
}

TEST(intersection30) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(2, 2, 1.6), // 0
      Point_3(1, 0, 1),   // 1
      Point_3(3, 0, 1),   // 2
      Point_3(3, 2, 1),   // 3
      Point_3(3, 4, 1),   // 4
      Point_3(1, 3, 1),   // 5
      Point_3(-1, 2, 1),  // 6
      Point_3(-1, 1, 1),  // 7
  };
  //   yellow.push_back(tile(y[0], y[1], y[2]));
  yellow.push_back(tile(y[0], y[2], y[3]));
  yellow.push_back(tile(y[0], y[3], y[4]));
  //   yellow.push_back(tile(y[0], y[4], y[5]));
  //   yellow.push_back(tile(y[0], y[5], y[6]));
  //   yellow.push_back(tile(y[0], y[6], y[7]));
  //   yellow.push_back(tile(y[0], y[7], y[1]));

  Point_3 g[] = {
      Point_3(0, 0, 2), // 0
      Point_3(4, 2, 1), // 1
      Point_3(0, 4, 2), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection30_expected, a, b, "30a");
  //   DO_CHECK();
  //   test_it(green, yellow, 2, 1, intersection30_expected, a, b, "30b");
  //   DO_CHECK();
}

TEST(intersection31) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(2, 2, 1.5), // 0
      Point_3(1, 0, 1),   // 1
      Point_3(3, 0, 1),   // 2
      Point_3(3, 2, 1),   // 3
      Point_3(3, 4, 1),   // 4
      Point_3(1, 3, 1),   // 5
      Point_3(-1, 2, 1),  // 6
      Point_3(-1, 1, 1),  // 7
  };
  //   yellow.push_back(tile(y[0], y[1], y[2]));
  //   yellow.push_back(tile(y[0], y[2], y[3]));
  //   yellow.push_back(tile(y[0], y[3], y[4]));
  yellow.push_back(tile(y[0], y[4], y[5]));
  yellow.push_back(tile(y[0], y[5], y[6]));
  yellow.push_back(tile(y[0], y[6], y[7]));
  yellow.push_back(tile(y[0], y[7], y[1]));

  Point_3 g[] = {
      Point_3(0, 0, 2), // 0
      Point_3(4, 2, 1), // 1
      Point_3(0, 4, 2), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, intersection30_expected, a, b, "31a");
  //   DO_CHECK();
  test_it(green, yellow, 2, 1, intersection30_expected, a, b, "31b");
  //   DO_CHECK();
}

TEST(intersection32) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 3.5, 2),   // 0
      Point_3(7, 0, 2),     // 1
      Point_3(4.5, 2.5, 1), // 2
      Point_3(3, 5, 1),     // 3
  };
  yellow.push_back(tile(y[0], y[1], y[2]));
  yellow.push_back(tile(y[0], y[2], y[3]));

  Point_3 g[] = {
      Point_3(3, 3, 1.9),     // 0
      Point_3(3, 2, 1),       // 1
      Point_3(4, 3, 1),       // 2
      Point_3(3, 4, 1),       // 3
      Point_3(1.5, 3.5, 1.5), // 4
      Point_3(0.5, 4, 2),     // 5
  };
  green.push_back(tile(g[0], g[1], g[2]));
  green.push_back(tile(g[0], g[4], g[1]));
  green.push_back(tile(g[0], g[2], g[3]));
  green.push_back(tile(g[1], g[4], g[5]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 2, 1, intersection30_expected, a, b, "32a");
  //   DO_CHECK();
  //   test_it(green, yellow, 2, 1, intersection30_expected, a, b, "32b");
  //   DO_CHECK();
}

TEST(intersection33) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(1, 3.5, 2),   // 0
      Point_3(7, 0, 2),     // 1
      Point_3(4.5, 2.5, 1), // 2
      Point_3(3, 5, 1),     // 3
  };
  yellow.push_back(tile(y[0], y[1], y[2]));
  yellow.push_back(tile(y[0], y[2], y[3]));

  Point_3 g[] = {
      Point_3(1.2, 4, 1),         // 0
      Point_3(0.5, 4.5, 2),       // 1
      Point_3(3, 2, 1),           // 2
      Point_3(4, 3, 1),           // 3
      Point_3(1.56667, 3.5, 1.5), // 4  left center
      Point_3(2.73333, 3, 1.5),   // 5  right center
      //     Point_3(2.73333,2.9,1.9),  // 5  right center
      Point_3(2.1, 3, 1.5), // 6  cut midpoint
  };
  green.push_back(tile(g[5], g[3], g[0]));
  green.push_back(tile(g[5], g[2], g[3]));
  green.push_back(tile(g[5], g[6], g[2]));
  green.push_back(tile(g[5], g[0], g[6]));
  green.push_back(tile(g[4], g[0], g[1]));
  green.push_back(tile(g[4], g[1], g[2]));
  green.push_back(tile(g[4], g[2], g[6]));
  green.push_back(tile(g[4], g[6], g[0]));
  //   green.push_back(tile(g[], g[], g[]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 2, 1, intersection30_expected, a, b, "33a");
  //   DO_CHECK();
  //   test_it(green, yellow, 2, 1, intersection30_expected, a, b, "33b");
  //   DO_CHECK();
}

TEST(intersection34) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(2, 2, 1.6), // 0
      Point_3(1, 0, 1),   // 1
      Point_3(3, 0, 1),   // 2
      Point_3(3, 2, 1),   // 3
      Point_3(3, 4, 1),   // 4
      Point_3(1, 3, 1),   // 5
      Point_3(-1, 2, 1),  // 6
      Point_3(-1, 1, 1),  // 7
  };
  //   yellow.push_back(tile(y[0], y[1], y[2]));
  //   yellow.push_back(tile(y[0], y[2], y[3]));
  //   yellow.push_back(tile(y[0], y[3], y[4]));
  //   yellow.push_back(tile(y[0], y[4], y[5]));
  yellow.push_back(tile(y[0], y[5], y[6]));
  yellow.push_back(tile(y[0], y[6], y[7]));
  //   yellow.push_back(tile(y[0], y[7], y[1]));

  Point_3 g[] = {
      Point_3(0, 0, 2), // 0
      Point_3(4, 2, 1), // 1
      Point_3(0, 4, 2), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, "" /*intersection34_expected*/, a, b, "34a");
  //   DO_CHECK();
  test_it(green, yellow, 2, 1, "" /*intersection34_expected*/, a, b, "34b");
  //   DO_CHECK();
}

TEST(intersection35) {
  list<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(2, 2, 1.6), // 0
      Point_3(1, 0, 1),   // 1
      Point_3(3, 0, 1),   // 2
      Point_3(3, 2, 1),   // 3
      Point_3(3, 4, 1),   // 4
      Point_3(1, 3, 1),   // 5
      Point_3(-1, 2, 1),  // 6
      Point_3(-1, 1, 1),  // 7
  };
  //   yellow.push_back(tile(y[0], y[1], y[2]));
  //   yellow.push_back(tile(y[0], y[2], y[3]));
  //   yellow.push_back(tile(y[0], y[3], y[4]));
  yellow.push_back(tile(y[0], y[4], y[5]));
  yellow.push_back(tile(y[0], y[5], y[6]));
  yellow.push_back(tile(y[0], y[6], y[7]));
  //   yellow.push_back(tile(y[0], y[7], y[1]));

  Point_3 g[] = {
      Point_3(0, 0, 2), // 0
      Point_3(4, 2, 1), // 1
      Point_3(0, 4, 2), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, "" /*intersection35_expected*/, a, b, "35a");
  //   DO_CHECK();
  test_it(green, yellow, 2, 1, "" /*intersection35_expected*/, a, b, "35b");
  //   DO_CHECK();
}

TEST(intersection36) {
  list<Triangle_handle> yellow, green;

  {
    vector<Polygon_2> temp;
    read_polygons_gnuplot3(data_dir + "/int36_1.dat", back_inserter(temp));
    Polygon_2 P = temp[0];
    green.push_back(tile(P[0], P[1], P[2]));
  }
  {
    vector<Polygon_2> temp;
    read_polygons_gnuplot3(data_dir + "/int36_2.dat", back_inserter(temp));
    Polygon_2 P = temp[0];
    yellow.push_back(tile(P[0], P[1], P[2]));
  }

  //   Point_3 y[] = {
  //     Point_3(1,4,2),   // 0
  //     Point_3(1,2,1),   // 1
  //     Point_3(2.5,3.5,2),// 2
  //   };
  //   yellow.push_back(tile(y[0], y[1], y[2]));

  //   Point_3 g[] = {
  //     Point_3(0,0,1),   // 0
  //     Point_3(4,2,2),   // 1
  //     Point_3(0,4,1),   // 2
  //   };
  //   green.push_back(tile(g[0], g[1], g[2]));

  vector<Triangle> a, b;
  bool good;
  test_it(yellow, green, 1, 2, "" /*intersection35_expected*/, a, b, "36a");
  //   DO_CHECK();
  test_it(green, yellow, 2, 1, "" /*intersection35_expected*/, a, b, "36b");
  //   DO_CHECK();
}

template <typename Segment> Point_3 point(const Segment &seg, Number_type u) {
  if (u < 0 || u > 1)
    throw logic_error("Illegal u");
  const Point_3 &s = seg.source();
  const Point_3 &t = seg.target();
  return Point_3(s.x() + (t.x() - s.x()) * u, s.y() + (t.y() - s.y()) * u,
                 s.z() + (t.z() - s.z()) * u);
}

//           (1,10)  (5,10)
//        _____(2,10)______(8,10)
// (0,10) |                    / (10,10)
//  (0,8) |                  / (9,9)
//        |                / (8,8)
//        |              /
//        |            /
//  (0,5) |  (2,5)   / (5,5)
//        |        /
//        |      /
//  (0,2) |    / (2,2)
//  (0,1) |  /
//        |/ (0,0)
//
struct Poly_fixture {
  Poly_fixture() {
    Segment_3 seg[] = {Segment_3(Point_3(0, 0, 0), Point_3(10, 10, 0)),
                       Segment_3(Point_3(10, 10, 0), Point_3(0, 10, 0)),
                       Segment_3(Point_3(0, 10, 0), Point_3(0, 0, 0))};

    boost::unordered_map<Segment_3_, boost::unordered_set<Point_3>> points;
    for (int i = 0; i < 3; ++i) {
      points[Segment_3_(seg[i])].insert(point(seg[i], .5));
      points[Segment_3_(seg[i])].insert(point(seg[i], .2));
      points[Segment_3_(seg[i])].insert(point(seg[i], .8));
      points[Segment_3_(seg[i])].insert(point(seg[i], .9));
    }

    Triangle tri(seg[0].source(), seg[1].source(), seg[2].source());
    poly = make_poly(tri, points);
  }
  Polygon_2 poly;
};

// TEST_F (Poly_fixture, intersection_cut1)
// {
//   Polygon_2 p0, p1;
//   boost::tie(p0, p1) = cut_into_polygons(poly, Segment_3(Point_3(8,10,0),
//   Point_3(0,5,0)));

// //   cout << pp(poly) << endl;
// //   cout << pp(p0) << endl;
// //   cout << pp(p1) << endl;

// }

TEST_F(Poly_fixture, intersection_make_poly1) {
  //   cout << pp(poly) << endl;
}

TEST(intersection_make_poly2) {
  // Check sorting
  Segment_3 seg[] = {Segment_3(Point_3(0, 0, 0), Point_3(10, 10, 0)),
                     Segment_3(Point_3(10, 10, 0), Point_3(0, 10, 0)),
                     Segment_3(Point_3(0, 10, 0), Point_3(0, 0, 0))};

  boost::unordered_map<Point_3, Segment_3_> points;
  for (int i = 0; i < 3; ++i) {
    points[point(seg[i], .5)] = Segment_3_(seg[i]);
    points[point(seg[i], .2)] = Segment_3_(seg[i]);
    points[point(seg[i], .8)] = Segment_3_(seg[i]);
    points[point(seg[i], .9)] = Segment_3_(seg[i]);
  }

  Triangle tri(seg[0].source(), seg[1].source(), seg[2].source());
  Polygon_2 poly = make_poly(tri, points);

  //   cout << pp(poly) << endl;
}

TEST(intersection_sort) {
  // Check sorting
  Segment_3 seg(Point_3(0, 0, 0), Point_3(10, 10, 10));
  Point_3 pt_arr[] = {point(seg, .5), point(seg, .2), point(seg, .8),
                      point(seg, .9)};

  Distance_functor<Point_3> f(seg.source());
  vector<Point_3> points(&(pt_arr[0]), &(pt_arr[3]));
  points.push_back(seg.source());
  points.push_back(seg.target());

  sort(points.begin(), points.end(), f);

  CHECK_EQUAL(0, points[0].x());
  CHECK_EQUAL(2, points[1].x());
  CHECK_EQUAL(5, points[2].x());
  CHECK_EQUAL(8, points[3].x());
  CHECK_EQUAL(10, points[4].x());
}

// TEST_F (Poly_fixture, intersection_poly1)
// {
//   Polyline_2 c;
//   c.push_back(Point_3(0,5,0));
//   c.push_back(Point_3(2,2,0));

//   Polygon_2 p0, p1;
//   boost::tie(p0, p1) = cut_into_polygons(poly, c);

//   CHECK_EQUAL(Point_2(0,0,0), p0[2]);
//   CHECK_EQUAL(Point_2(2,2,0), p0[3]);
//   CHECK_EQUAL(Point_2(0,5,0), p0[4]);
//   CHECK_EQUAL(Point_2(0,2,0), p0[0]);
//   CHECK_EQUAL(Point_2(0,1,0), p0[1]);

//   CHECK_EQUAL(Point_3(5,5,0), p1[0]);
//   CHECK_EQUAL(Point_3(8,8,0), p1[1]);
//   CHECK_EQUAL(Point_3(9,9,0), p1[2]);
//   CHECK_EQUAL(Point_3(0,8,0), p1[9]);
//   CHECK_EQUAL(Point_3(0,5,0), p1[10]);
//   CHECK_EQUAL(Point_3(2,2,0), p1[11]);
// }

TEST(intersection_poly) {
  Polygon_2 poly;
  poly.push_back(Point_3(0, 0, 0));
  poly.push_back(Point_3(2, 2, 0));
  poly.push_back(Point_3(5, 5, 0));
  poly.push_back(Point_3(10, 10, 0));
  poly.push_back(Point_3(0, 10, 0));
  poly.push_back(Point_3(0, 5, 0));
  poly.push_back(Point_3(0, 2, 0));

  list<Polyline_2> cuts;

  Polyline_2 c;
  c.push_back(Point_3(0, 2, 1));
  c.push_back(Point_3(2, 2, 1));
  cuts.push_back(c);

  c = Polyline_2();
  c.push_back(Point_3(0, 5, 1));
  c.push_back(Point_3(2, 5, 1));
  c.push_back(Point_3(2, 2, 1));
  cuts.push_back(c);

  c = Polyline_2();
  c.push_back(Point_3(2, 2, 1));
  c.push_back(Point_3(2, 5, 1));
  c.push_back(Point_3(5, 5, 1));
  cuts.push_back(c);

  Polygon_2 p0, p1, p00, p01, p10, p11;

  vector<Triangle> triangles;
  cut_into_triangles(poly, cuts.begin(), cuts.end(),
                     back_inserter(triangles));

  //   for (vector<Triangle>::iterator it = triangles.begin(); it !=
  //   triangles.end(); ++it)
  //     cout << pp((*it)[0]) << ", " <<  pp((*it)[1]) << ", " << pp((*it)[2])
  //     << endl;
}

TEST(intersection_trim1) {
  Polygon_2 poly;
  poly.push_back(Point_3(10, 0, 0));
  poly.push_back(Point_3(10, 5, 0));
  poly.push_back(Point_3(5, 5, 0));
  poly.push_back(Point_3(5, 10, 0));
  poly.push_back(Point_3(0, 10, 0));
  poly.push_back(Point_3(0, 0, 0));

  Polyline_2 cut;
  cut.push_back(Point_3(0, 0, 0));
  cut.push_back(Point_3(5, 5, 0));
  cut.push_back(Point_3(5, 10, 0));

  Polyline_2 t1 = trim_forward(poly, cut);
  CHECK_EQUAL(Point_3(0, 0, 0), t1[0]);
  CHECK_EQUAL(Point_3(5, 5, 0), t1[1]);

  cut = Polyline_2(cut.rbegin(), cut.rend());
  Polyline_2 t2 = trim_backward(poly, cut);
  CHECK_EQUAL(Point_3(5, 5, 0), t2[0]);
  CHECK_EQUAL(Point_3(0, 0, 0), t2[1]);
}

TEST(intersection_trim2) {
  Polygon_2 poly;
  poly.push_back(Point_3(5, 5, 0));
  poly.push_back(Point_3(5, 10, 0));
  poly.push_back(Point_3(0, 10, 0));
  poly.push_back(Point_3(0, 0, 0));
  poly.push_back(Point_3(10, 0, 0));
  poly.push_back(Point_3(10, 5, 0));

  Polyline_2 cut;
  cut.push_back(Point_3(0, 0, 0));
  cut.push_back(Point_3(5, 5, 0));
  cut.push_back(Point_3(5, 10, 0));

  Polyline_2 t1 = trim_forward(poly, cut);
  CHECK_EQUAL(Point_3(0, 0, 0), t1[0]);
  CHECK_EQUAL(Point_3(5, 5, 0), t1[1]);

  cut = Polyline_2(cut.rbegin(), cut.rend());
  Polyline_2 t2 = trim_backward(poly, cut);
  CHECK_EQUAL(Point_3(5, 5, 0), t2[0]);
  CHECK_EQUAL(Point_3(0, 0, 0), t2[1]);
}

TEST(intersection_trim3) {
  Polygon_2 poly;
  poly.push_back(Point_3(5, 10, 0));
  poly.push_back(Point_3(0, 10, 0));
  poly.push_back(Point_3(0, 0, 0));
  poly.push_back(Point_3(10, 0, 0));
  poly.push_back(Point_3(10, 5, 0));
  poly.push_back(Point_3(5, 5, 0));

  Polyline_2 cut;
  cut.push_back(Point_3(0, 0, 0));
  cut.push_back(Point_3(5, 5, 0));
  cut.push_back(Point_3(5, 10, 0));

  Polyline_2 t1 = trim_forward(poly, cut);
  CHECK_EQUAL(Point_3(0, 0, 0), t1[0]);
  CHECK_EQUAL(Point_3(5, 5, 0), t1[1]);

  cut = Polyline_2(cut.rbegin(), cut.rend());
  Polyline_2 t2 = trim_backward(poly, cut);
  CHECK_EQUAL(Point_3(5, 5, 0), t2[0]);
  CHECK_EQUAL(Point_3(0, 0, 0), t2[1]);
}

TEST(intersection_trim4) {
  Polygon_2 poly;
  poly.push_back(Point_3(0, 10, 0));
  poly.push_back(Point_3(0, 0, 0));
  poly.push_back(Point_3(10, 0, 0));
  poly.push_back(Point_3(10, 5, 0));
  poly.push_back(Point_3(5, 5, 0));
  poly.push_back(Point_3(5, 10, 0));

  Polyline_2 cut;
  cut.push_back(Point_3(0, 0, 1));
  cut.push_back(Point_3(5, 5, 1));
  cut.push_back(Point_3(5, 10, 1));

  Polyline_2 t1 = trim_forward(poly, cut);
  CHECK_EQUAL(Point_3(0, 0, 1), t1[0]);
  CHECK_EQUAL(Point_3(5, 5, 1), t1[1]);

  cut = Polyline_2(cut.rbegin(), cut.rend());
  Polyline_2 t2 = trim_backward(poly, cut);
  CHECK_EQUAL(Point_3(5, 5, 1), t2[0]);
  CHECK_EQUAL(Point_3(0, 0, 1), t2[1]);
}

// Lab book page 53
TEST(intersection_neighbor1) {
  Triangle_handle ytile(
      new Triangle(Point_3(3, 2, 1), Point_3(1, 2, 0), Point_3(3, 0, 0)));
  Triangle_handle gtile(
      new Triangle(Point_3(0, 0, 0), Point_3(4, 0, 1), Point_3(0, 4, 0)));
  list<Triangle_handle> ytiles, gtiles;
  ytiles.push_back(ytile);
  gtiles.push_back(gtile);

  double yz = 0;
  double gz = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(ytiles.begin(),
  //   ytiles.end(),
  // 							  gtiles.begin(),
  // gtiles.end(), 							  z_adj);

  //   Point_3 gn;
  //   CHECK_EQUAL(true, neighbor(gtile, ytile, Point_3(3,1,0.75), 1, gn,
  //   ints));
}

// Lab book page 53
TEST(intersection_neighbor2) {
  Triangle_handle ytile(
      new Triangle(Point_3(3, 2, 1), Point_3(1, 2, 0), Point_3(3, -2, 0)));
  Triangle_handle gtile(
      new Triangle(Point_3(0, 0, 0), Point_3(4, 0, 1), Point_3(0, 4, 0)));
  list<Triangle_handle> ytiles, gtiles;
  ytiles.push_back(ytile);
  gtiles.push_back(gtile);

  double yz = 0;
  double gz = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(ytiles.begin(),
  //   ytiles.end(),
  // 							  gtiles.begin(),
  // gtiles.end(), 							  z_adj);

  //   Point_3 gn;
  //   bool ret = neighbor(gtile, ytile, Point_3(3,1,0.75), 1, gn, ints);
  //   CHECK(ret);
  //   CHECK_EQUAL(Point_3(3,0,0.75), gn);

  //   ret = neighbor(gtile, ytile, Point_3(2,2,0.5), 1, gn, ints);
  //   CHECK(!ret);
}

// Lab book page 53
TEST(intersection_neighbor3) {
  Triangle_handle ytile(
      new Triangle(Point_3(3, 2, 1), Point_3(1, 2, 0), Point_3(3, 0.5, 0)));
  Triangle_handle gtile(
      new Triangle(Point_3(0, 0, 0), Point_3(4, 0, 1), Point_3(0, 4, 0)));
  list<Triangle_handle> ytiles, gtiles;
  ytiles.push_back(ytile);
  gtiles.push_back(gtile);

  double yz = 0;
  double gz = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(ytiles.begin(),
  //   ytiles.end(),
  // 							  gtiles.begin(),
  // gtiles.end(), 							  z_adj);

  //   Point_3 gn;
  //   CHECK_EQUAL(false, neighbor(gtile, ytile, Point_3(3,1,0.75), 1, gn,
  //   ints));
}

// Lab book page 53
TEST(intersection_exit1) {
  Triangle_handle ytile(
      new Triangle(Point_3(3, 2, 1), Point_3(1, 2, 0), Point_3(3, 0, 0)));
  Triangle_handle gtile(
      new Triangle(Point_3(0, 0, 0), Point_3(4, 0, 1), Point_3(0, 4, 0)));
  list<Triangle_handle> ytiles, gtiles;
  ytiles.push_back(ytile);
  gtiles.push_back(gtile);

  double yz = 0;
  double gz = 1;
  int yi = 0;
  int gi = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(ytiles.begin(),
  //   ytiles.end(),
  // 							  gtiles.begin(),
  // gtiles.end(), 							  z_adj);

  //   Polyline_2 cut = find_exit(gtile, ytile, Point_3(3,1,0.75), gi, ints,
  //   z_adj); CHECK_EQUAL(Point_3(3,1,0.75), (Point_3)cut[0]);
  //   CHECK_EQUAL(Point_3(3,0,0.75), (Point_3)cut[1]);
}

// Lab book page 53
TEST(intersection_exit2) {
  Triangle_handle ytile(
      new Triangle(Point_3(3, 2, 1), Point_3(1, 2, 0), Point_3(3, -2, 0)));
  Triangle_handle gtile(
      new Triangle(Point_3(0, 0, 0), Point_3(4, 0, 1), Point_3(0, 4, 0)));
  list<Triangle_handle> ytiles, gtiles;
  ytiles.push_back(ytile);
  gtiles.push_back(gtile);

  double yz = 0;
  double gz = 1;
  int yi = 0;
  int gi = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(ytiles.begin(),
  //   ytiles.end(),
  // 							  gtiles.begin(),
  // gtiles.end(), 							  z_adj);

  //   Polyline_2 cut = find_exit(gtile, ytile, Point_3(3,1,0.75), gi, ints,
  //   z_adj); CHECK_EQUAL(Point_3(3,1,0.75), (Point_3) cut[0]);
  //   CHECK_EQUAL(Point_3(3,0,0.75), (Point_3) cut[1]);

  //   cut = find_exit(gtile, ytile, Point_3(3,0,0.75), gi, ints, z_adj);
  //   CHECK_EQUAL(Point_3(3,0,0.75), (Point_3) cut[0]);
  //   CHECK_EQUAL(Point_3(3,1,0.75), (Point_3) cut[1]);

  //   cut = find_exit(gtile, ytile, Point_3(2,2,0.5), gi, ints, z_adj);
  //   CHECK_EQUAL(Point_3(2,2,0.5), (Point_3) cut[0]);
  //   CHECK_EQUAL(Point_3(1,2,0.25), (Point_3) cut[1]);
  //   CHECK_EQUAL(Point_3(2,0,0.5), (Point_3) cut[2]);

  //   cut = find_exit(gtile, ytile, Point_3(2,0,0.5), gi, ints, z_adj);
  //   CHECK_EQUAL(Point_3(2,0,0.5), (Point_3) cut[0]);
  //   CHECK_EQUAL(Point_3(1,2,0.25), (Point_3) cut[1]);
  //   CHECK_EQUAL(Point_3(2,2,0.5), (Point_3) cut[2]);
}

// Lab book page 53
TEST(intersection_exit3) {
  Triangle_handle ytile(
      new Triangle(Point_3(3, 2, 1), Point_3(1, 2, 0), Point_3(3, 0.5, 0)));
  Triangle_handle gtile(
      new Triangle(Point_3(0, 0, 0), Point_3(4, 0, 1), Point_3(0, 4, 0)));
  list<Triangle_handle> ytiles, gtiles;
  ytiles.push_back(ytile);
  gtiles.push_back(gtile);

  double yz = 0;
  double gz = 1;
  int yi = 0;
  int gi = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(ytiles.begin(),
  //   ytiles.end(),
  // 							  gtiles.begin(),
  // gtiles.end(), 							  z_adj);

  //   Polyline_2 cut = find_exit(gtile, ytile, Point_3(3,1,0.75), gi, ints,
  //   z_adj); CHECK_EQUAL(Point_3(3,1,0.75), (Point_3) cut[0]);
  //   CHECK_EQUAL(Point_3(3,0.5,0.75), (Point_3) cut[1]);
  //   CHECK_EQUAL(Point_3(1,2,0.25), (Point_3) cut[2]);
  //   CHECK_EQUAL(Point_3(2,2,0.5), (Point_3) cut[3]);

  //   cut = find_exit(gtile, ytile, Point_3(2,2,0.5), gi, ints, z_adj);
  //   CHECK_EQUAL(Point_3(2,2,0.5), (Point_3) cut[0]);
  //   CHECK_EQUAL(Point_3(1,2,0.25), (Point_3) cut[1]);
  //   CHECK_EQUAL(Point_3(3,0.5,0.75), (Point_3) cut[2]);
  //   CHECK_EQUAL(Point_3(3,1,0.75), (Point_3) cut[3]);
}

// Lab book page 53
TEST(intersection_exit4) {
  Triangle_handle ytile(
      new Triangle(Point_3(1, 1, 1), Point_3(1, -1, 0), Point_3(3, 1, 0)));
  Triangle_handle gtile(
      new Triangle(Point_3(0, 0, 0), Point_3(4, 0, 1), Point_3(0, 0, 1)));
  list<Triangle_handle> ytiles, gtiles;
  ytiles.push_back(ytile);
  gtiles.push_back(gtile);

  double yz = 0;
  double gz = 1;
  int yi = 0;
  int gi = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(ytiles.begin(),
  //   ytiles.end(),
  // 							  gtiles.begin(),
  // gtiles.end(), 							  z_adj);

  //   Polyline_2 cut = find_exit(gtile, ytile, Point_3(1,0,0.25), gi, ints,
  //   z_adj); CHECK_EQUAL(Point_3(1,0,0.25), (Point_3) cut[0]);
  //   CHECK_EQUAL(1, cut.size());
}

// Lab book page 53
TEST(intersection_exit26) {
  vector<Triangle_handle> yellow, green;

  Point_3 y[] = {
      Point_3(8, 4, 2),   // 0
      Point_3(0, 0, 1.5), // 1
      Point_3(8, 3, 2),   // 2
  };
  yellow.push_back(tile(y[0], y[1], y[2]));

  Point_3 g[] = {
      Point_3(0, 0, 1.5), // 0
      Point_3(6, 2, 2),   // 1
      Point_3(0, 4, 1.5), // 2
  };
  green.push_back(tile(g[0], g[1], g[2]));

  double yz = 0;
  double gz = 1;
  int yi = 0;
  int gi = 1;
  Z_adjustments<Triangle_handle> z_adj(yz, gz, 0.04);
  //   Intersections<Triangle_handle> ints = get_intersections(yellow.begin(),
  //   yellow.end(),
  // 							  green.begin(),
  // green.end(), 							  z_adj);

  //   Polyline_2 cut = find_exit(green[0], yellow[0], Point_3(0,0,1.5), gi,
  //   ints, z_adj);
  // //   cout << pp(cut) << endl;
  // //   CHECK_EQUAL(Point_3(1,0,0.25), (Point_3) cut[0]);
  // //   CHECK_EQUAL(1, cut.size());
}

TEST(intersection_get_z1) {
  Triangle_handle t(
      new Triangle(Point_3(0, 0, 0), Point_3(1, 0, 1), Point_3(0, 1, 0)));

  CHECK_EQUAL(0, get_z(*t, Point_3(0, 0, 0)));
  CHECK_EQUAL(0.5, get_z(*t, Point_3(0.5, 0, 0)));
  CHECK_EQUAL(0, get_z(*t, Point_3(0, 0.5, 0)));
  CHECK_EQUAL(0.5, get_z(*t, Point_3(0.5, 0.5, 0)));
  CHECK_EQUAL(1, get_z(*t, Point_3(1, 0, 0)));
  CHECK_EQUAL(0, get_z(*t, Point_3(0, 1, 0)));
  CHECK_EQUAL(0.75, get_z(*t, Point_3(0.75, 0.25, 0)));
}

TEST(raw_equal1) {
  string sa("3 3        \n"
            "0 0 0 0 1 0 \n"
            "0 1 0 0 1 1 \n"
            "0 0 1 1 1 0 \n"
            "0 1 2       \n"
            "0 2 1       \n"
            "2 1 0       \n");
  string sb("3 3        \n"
            "0 0 1 1 1 0 \n"
            "0 1 0 0 2 1 \n"
            "0 0 0 0 1 3 \n"
            "0 1 2       \n"
            "0 2 1       \n"
            "2 1 0       \n");

  stringstream ssa(sa);
  stringstream ssb(sb);

  vector<Triangle> a, b;
  normalized_raw(ssa, true, back_inserter(a));
  normalized_raw(ssb, true, back_inserter(b));

  CHECK_EQUAL(a.size(), b.size());
  if (a.size() != b.size())
    return;

  for (int i = 0; i < a.size(); ++i)
    for (int j = 0; j < 3; ++j)
      CHECK_EQUAL(a[i][j], b[i][j]);
}

TEST(triangulate1) {
  Polygon_2 p;
  p.push_back(Point_2(0, 2.5));
  p.push_back(Point_2(1, 3));
  p.push_back(Point_2(1.5, 3.25));
  p.push_back(Point_2(0, 4));

  //   if (!p.is_simple())
  //     cout << "Not simple" << endl;
  list<Triangle> nt;
  boost::unordered_map<Point_3, boost::unordered_set<Segment_3_undirected>>
      point2edges;
  triangulate_safe(p, back_inserter(nt), point2edges);
}

TEST(triangulate2) {
  Polygon_2 p;
  p.push_back(Point_2(0, 0));
  p.push_back(Point_2(0.5, 0));
  p.push_back(Point_2(1, 0));
  p.push_back(Point_2(1, 1));
  p.push_back(Point_2(0, 1));

  boost::unordered_map<Point_3, boost::unordered_set<Segment_3_undirected>>
      point2edges;
  point2edges[Point_3(0, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 1, 0), Point_3(0, 0, 0)));
  point2edges[Point_3(0, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 0, 0), Point_3(1, 0, 0)));
  point2edges[Point_3(0.5, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 0, 0), Point_3(1, 0, 0)));
  point2edges[Point_3(1, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 0, 0), Point_3(1, 0, 0)));
  point2edges[Point_3(1, 0, 0)].insert(
      Segment_3_undirected(Point_3(1, 0, 0), Point_3(1, 1, 0)));
  point2edges[Point_3(1, 1, 0)].insert(
      Segment_3_undirected(Point_3(1, 0, 0), Point_3(1, 1, 0)));
  point2edges[Point_3(1, 1, 0)].insert(
      Segment_3_undirected(Point_3(1, 1, 0), Point_3(0, 1, 0)));
  point2edges[Point_3(0, 1, 0)].insert(
      Segment_3_undirected(Point_3(1, 1, 0), Point_3(0, 1, 0)));
  point2edges[Point_3(0, 1, 0)].insert(
      Segment_3_undirected(Point_3(0, 1, 0), Point_3(0, 0, 0)));

  //   Polygon_2 p1;
  //   p1.push_back(Point_2(0, 0));
  //   p1.push_back(Point_2(1, 0));
  //   p1.push_back(Point_2(1, 1));
  //   p1.push_back(Point_2(0, 1));

  //   cout << is_strictly_convex(p) << endl;
  //   cout << is_strictly_convex(p1) << endl;
  //   cout << CGAL::left_turn(Point_2(0,0), Point_2(1,0), Point_2(2,0)) <<
  //   endl; cout << CGAL::right_turn(Point_2(0,0), Point_2(1,0),
  //   Point_2(2,0)) << endl;

  //   foo(p);

  list<Triangle> nt;
  triangulate_safe(p, back_inserter(nt), point2edges);
  //   for (list<Triangle>::iterator it = nt.begin(); it != nt.end(); ++it)
  //     cout << pp_tri(*it) << endl;
}

TEST(getz1) {
  Triangle t(Point_3(4.5, 2.5, 0), Point_3(3, 5, 0), Point_3(2.5, 2.5, 1));
  Point_2 p(2.73333, 3.21429);
  //   cout << get_z(t, p) << endl;

  CGAL::Triangle_2<Kernel> tri(Point_2(5, 2.5), Point_2(7, 0),
                               Point_2(2.5, 2.5));
  //   if (tri.has_on_positive_side(Point_2(2.73333,3.21429)))
  //     cout << "Positive" << endl;
  //   else
  //     cout << "Negative" << endl;
}

// // Makes sure that no three points all on an original tile edge
// // are triangulated.
// bool is_legal(const Point_3& a, const Point_3& b, const Point_3& c,
// 	      const boost::unordered_map<Point_3,
// boost::unordered_set<Segment_3_undirected> >& point2edges)
// {
//   static log4cplus::Logger logger =
//   log4cplus::Logger::getInstance("polygon_utils");

//   LOG4CPLUS_TRACE(logger, "is_legal: " << pp(a) << " " << pp(b) << " " <<
//   pp(c));

//   if (point2edges.find(a) == point2edges.end())
//     return true;
//   if (point2edges.find(b) == point2edges.end())
//     return true;
//   if (point2edges.find(c) == point2edges.end())
//     return true;

//   const boost::unordered_set<Segment_3_undirected>& edgesa =
//   point2edges.find(a)->second; const
//   boost::unordered_set<Segment_3_undirected>& edgesb =
//   point2edges.find(b)->second; const
//   boost::unordered_set<Segment_3_undirected>& edgesc =
//   point2edges.find(c)->second;

//   boost::unordered_set<Segment_3_undirected> edges(set_intersection(edgesa,
//   set_intersection(edgesb, edgesc))); return edges.empty();
// }

// Point_3 yz_swap_pos(const Point_3& p)
// {
//   return Point_3(p.x(), -p.z(), p.y());
// }

// Point_3 yz_swap_neg(const Point_3& p)
// {
//   return Point_3(p.x(), p.z(), -p.y());
// }

// Polygon_2 yz_swap_pos(const Polygon_2& p)
// {
//   Polygon_2 ret;
//   for (Polygon_2::Vertex_iterator it = p.vertices_begin(); it !=
//   p.vertices_end(); ++it)
//     ret.push_back(yz_swap_pos(*it));
//   return ret;
// }

// Polygon_2 yz_swap_neg(const Polygon_2& p)
// {
//   Polygon_2 ret;
//   for (Polygon_2::Vertex_iterator it = p.vertices_begin(); it !=
//   p.vertices_end(); ++it)
//     ret.push_back(yz_swap_neg(*it));
//   return ret;
// }

// Polyline_2 yz_swap_neg(const Polyline_2& p)
// {
//   Polyline_2 ret;
//   for (Polyline_2::const_iterator it = p.begin(); it != p.end(); ++it)
//     ret.push_back(yz_swap_neg(*it));
//   return ret;
// }

// bool is_vertical(const Polygon_2& p)
// {
//   for (int i = 0; i < p.size() - 2; ++i)
//     if (!collinear(p[i], p[i+1], p[i+2]))
//       return false;
//   return true;
// }

// template <typename Cut_iter, typename Out_iter>
// void triangulate(const Polygon_2& polygon,
// 		 Cut_iter cuts_begin, Cut_iter cuts_end,
// 		 const boost::unordered_map<Point_3,
// boost::unordered_set<Segment_3_undirected> >& point2edges, 		 Out_iter
// triangles)
// {
//   typedef CGAL::Triangulation_vertex_base_2<Kernel>                     Vb;
//   typedef CGAL::Triangulation_vertex_base_with_info_2<bool, Kernel, Vb>
//   Info; typedef CGAL::Constrained_triangulation_face_base_2<Kernel> Fb;
//   typedef CGAL::Triangulation_data_structure_2<Info,Fb>              TDS;
//   typedef CGAL::Exact_predicates_tag                               Itag;
//   typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, TDS, Itag>
//   CDT;

//   static log4cplus::Logger logger =
//   log4cplus::Logger::getInstance("polygon_utils");

//   Polygon_2 p(polygon);
//   LOG4CPLUS_TRACE(logger, "Triangulating " << pp(p));
//   if (p.size() < 3) return;

//   bool vertical = is_vertical(p);
//   if (vertical)
//   {
//     LOG4CPLUS_TRACE(logger, "Polygon is vertical.  Rotating.");
//     p = yz_swap_neg(p);
//   }

//   bool reverse = !p.is_counterclockwise_oriented();
//   if (reverse)
//     p.reverse_orientation();

//   CDT cdt;
//   Polygon_2::Vertex_circulator start = p.vertices_circulator();
//   Polygon_2::Vertex_circulator c = start;
//   Polygon_2::Vertex_circulator n = c;
//   ++n;
//   do
//   {
//     cdt.insert_constraint(*c, *n);
//     ++c;
//     ++n;
//   } while (c != start);

//   for (Cut_iter c_it = cuts_begin; c_it != cuts_end; ++c_it)
//   {
//     Polyline_2 cut = *c_it;
//     if (vertical)
//       cut = yz_swap_neg(cut);
//     for (Polyline_2::const_iterator c = cut.begin(); c != cut.end(); ++c)
//     {
//       Polyline_2::const_iterator n = c;
//       ++n;
//       if (n != cut.end())
// 	cdt.insert_constraint(*c, *n);
//     }
//   }

//   // Loop through the triangulation and store the vertices of each triangle
//   for (CDT::Finite_faces_iterator ffi = cdt.finite_faces_begin();
//        ffi != cdt.finite_faces_end();
//        ++ffi)
//   {
//     Triangle t;
//     Point_3 center = centroid(ffi->vertex(0)->point(),
//     ffi->vertex(1)->point(), ffi->vertex(2)->point()); if
//     (p.has_on_bounded_side(center) &&
// 	is_legal(ffi->vertex(0)->point(), ffi->vertex(1)->point(),
// ffi->vertex(2)->point(), point2edges))
//     {
//       for (int i = 0; i < 3; ++i)
//       {
// 	int idx = reverse ? 2-i : i;
// 	if (!vertical)
// 	  t[idx] = ffi->vertex(i)->point();
// 	else
// 	  t[idx] = yz_swap_pos(ffi->vertex(i)->point());
//       }
//       LOG4CPLUS_TRACE(logger, "Adding tile: " << pp_tri(t));
//       *triangles++ = t;
//     }
//   }
// }

TEST(new_triangulate1) {
  Polygon_2 p;
  p.push_back(Point_2(0, 0));
  p.push_back(Point_2(0.5, 0));
  p.push_back(Point_2(1, 0));
  p.push_back(Point_2(1, 1));
  p.push_back(Point_2(0, 1));

  typedef boost::unordered_map<Point_3,
                               boost::unordered_set<Segment_3_undirected>>
      Point2edges;
  Point2edges point2edges;
  point2edges[Point_3(0, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 1, 0), Point_3(0, 0, 0)));
  point2edges[Point_3(0, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 0, 0), Point_3(1, 0, 0)));
  point2edges[Point_3(0.5, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 0, 0), Point_3(1, 0, 0)));
  point2edges[Point_3(1, 0, 0)].insert(
      Segment_3_undirected(Point_3(0, 0, 0), Point_3(1, 0, 0)));
  point2edges[Point_3(1, 0, 0)].insert(
      Segment_3_undirected(Point_3(1, 0, 0), Point_3(1, 1, 0)));
  point2edges[Point_3(1, 1, 0)].insert(
      Segment_3_undirected(Point_3(1, 0, 0), Point_3(1, 1, 0)));
  point2edges[Point_3(1, 1, 0)].insert(
      Segment_3_undirected(Point_3(1, 1, 0), Point_3(0, 1, 0)));
  point2edges[Point_3(0, 1, 0)].insert(
      Segment_3_undirected(Point_3(1, 1, 0), Point_3(0, 1, 0)));
  point2edges[Point_3(0, 1, 0)].insert(
      Segment_3_undirected(Point_3(0, 1, 0), Point_3(0, 0, 0)));

  Polyline_2 cut;
  cut.push_back(Point_3(0, 1, 0));
  cut.push_back(Point_3(0.5, 0.5, 0));
  cut.push_back(Point_3(0.5, 0, 0));
  cut.push_back(Point_3(1, 0, 0));
  cut.push_back(Point_3(1, 1, 0));

  list<Polyline_2> cuts;
  cuts.push_back(cut);

  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("polygon_utils");

  list<Triangle> triangles;
  triangulate(p, cuts.begin(), cuts.end(), point2edges,
              back_inserter(triangles));

  //   for (list<Triangle>::iterator it = triangles.begin(); it !=
  //   triangles.end(); ++it)
  //   {
  //     LOG4CPLUS_INFO(logger, pp_tri(*it));
  //   }
}

TEST(abc1) {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("test_intersection");

  Polygon_2 p;
  p.push_back(Point_3(2, 2, 1.5));
  p.push_back(Point_3(1, 3, 1.75));
  p.push_back(Point_3(1.5, 3.25, 1.625));
  p.push_back(Point_3(0, 4, 2));
  p.push_back(Point_3(0, 2.5, 2));
  p.push_back(Point_3(0, 2, 2));

  Polyline_2 cut;
  cut.push_back(Point_3(0, 2.5, 2));
  cut.push_back(Point_3(0.5, 2.5, 2));
  cut.push_back(Point_3(1, 3, 1.75));
  cut.push_back(Point_3(2, 2, 1.5));
  cut.push_back(Point_3(0, 2, 2));

  Polygon_2 a, b;
  //   boost::tie(a, b) = cut_into_polygons(p, cut);
  //   LOG4CPLUS_INFO(logger, "a: " << pp(a));
  //   LOG4CPLUS_INFO(logger, "b: " << pp(b));
}

TEST(quad1) {

  Number_type r1, r2, c1, c2;

  CHECK(!solve_quad(0, 0, 0, r1, r2, c1, c2));

  CHECK(solve_quad(4, 0, 0, r1, r2, c1, c2));
  CHECK_EQUAL(0, r1);
  CHECK_EQUAL(0, r2);

  CHECK(!solve_quad(0, 0, 4, r1, r2, c1, c2));

  CHECK(solve_quad(0, 4, 4, r1, r2, c1, c2));
  CHECK_EQUAL(-1, r1);
  CHECK_EQUAL(-1, r2);

  CHECK(solve_quad(0, 4, -4, r1, r2, c1, c2));
  CHECK_EQUAL(1, r1);
  CHECK_EQUAL(1, r2);

  CHECK(!solve_quad(1, 0, 4, r1, r2, c1, c2));

  CHECK(solve_quad(1, 0, -4, r1, r2, c1, c2));
  CHECK_EQUAL(2, r1);
  CHECK_EQUAL(-2, r2);

  CHECK(solve_quad(1, 1, -12, r1, r2, c1, c2));
  CHECK_EQUAL(-4, r1);
  CHECK_EQUAL(3, r2);

  CHECK(solve_quad(1, 1, -12, r1, r2, c1, c2));
  CHECK_EQUAL(-4, r1);
  CHECK_EQUAL(3, r2);

  CHECK(solve_quad(3.14, -22.31561999, 16.4101, r1, r2, c1, c2));
  CHECK(abs(r1 - 6.2738853503184711) < 0.001);
  CHECK(abs(r2 - .833) < 0.001);

  CHECK(solve_quad(0.276254000000000, -0.104644000000000, 0.009909640000000,
                   r1, r2, c1, c2));
}

TEST(cut1) {
  Triangle triangle(Point_2(0, 0), Point_2(1, 0), Point_2(0, 1));
  Segment_3 edge(Point_2(0, 0), Point_2(1, 0));
  Point_3 point(0.5, 0, 0);
  Triangle t1, t2;

  boost::tie(t1, t2) = decompose_triangle(triangle, edge, point);

  CHECK(index(Point_2(0, 0), t1) > -1);
  CHECK(index(Point_2(0.5, 0), t1) > -1);
  CHECK(index(Point_2(0, 1), t1) > -1);

  CHECK(index(Point_2(1, 0), t2) > -1);
  CHECK(index(Point_2(0.5, 0), t2) > -1);
  CHECK(index(Point_2(0, 1), t2) > -1);

  cout << pp_tri(t1) << endl;
}

TEST(cut2) {
  Triangle triangle(Point_2(0, 0), Point_2(1, 0), Point_2(0, 1));
  Segment_3 edge(Point_2(0, 0), Point_2(1, 0));
  list<Point_3> points;
  points.push_back(Point_3(0.5, 0, 0));
  points.push_back(Point_3(0.75, 0, 0));
  boost::unordered_map<Segment_3_undirected, list<Point_3>> edge2points;

  edge2points[edge] = points;

  list<Triangle> triangles;

  decompose_triangle(triangle, edge2points, back_inserter(triangles));

  //   CHECK(index(Point_2(0,0), t1) > -1);
  //   CHECK(index(Point_2(0.5,0), t1) > -1);
  //   CHECK(index(Point_2(0,1), t1) > -1);

  //   CHECK(index(Point_2(1,0), t2) > -1);
  //   CHECK(index(Point_2(0.5,0), t2) > -1);
  //   CHECK(index(Point_2(0,1), t2) > -1);

  for (list<Triangle>::const_iterator it = triangles.begin();
       it != triangles.end(); ++it) {
    cout << pp_tri(*it) << endl;
  }
}

TEST(cut3) {
  Triangle triangle(Point_2(1, 0), Point_2(0, 1), Point_2(0, 0));
  Segment_3 edge(Point_2(0, 0), Point_2(1, 0));
  list<Point_3> points;
  points.push_back(Point_3(0.5, 0, 0));
  points.push_back(Point_3(0.75, 0, 0));
  boost::unordered_map<Segment_3_undirected, list<Point_3>> edge2points;

  edge2points[edge] = points;

  list<Triangle> triangles;

  decompose_triangle(triangle, edge2points, back_inserter(triangles));

  //   CHECK(index(Point_2(0,0), t1) > -1);
  //   CHECK(index(Point_2(0.5,0), t1) > -1);
  //   CHECK(index(Point_2(0,1), t1) > -1);

  //   CHECK(index(Point_2(1,0), t2) > -1);
  //   CHECK(index(Point_2(0.5,0), t2) > -1);
  //   CHECK(index(Point_2(0,1), t2) > -1);

  for (list<Triangle>::const_iterator it = triangles.begin();
       it != triangles.end(); ++it) {
    cout << pp_tri(*it) << endl;
  }
}

TEST(cut4) {
  Triangle triangle(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
  Segment_3 edge(Point_2(0, 0), Point_2(1, 0));
  list<Point_3> points;
  points.push_back(Point_3(0.5, 0, 0));
  points.push_back(Point_3(0.75, 0, 0));
  boost::unordered_map<Segment_3_undirected, list<Point_3>> edge2points;

  edge2points[edge] = points;

  list<Triangle> triangles;

  decompose_triangle(triangle, edge2points, back_inserter(triangles));

  //   CHECK(index(Point_2(0,0), t1) > -1);
  //   CHECK(index(Point_2(0.5,0), t1) > -1);
  //   CHECK(index(Point_2(0,1), t1) > -1);

  //   CHECK(index(Point_2(1,0), t2) > -1);
  //   CHECK(index(Point_2(0.5,0), t2) > -1);
  //   CHECK(index(Point_2(0,1), t2) > -1);

  for (list<Triangle>::const_iterator it = triangles.begin();
       it != triangles.end(); ++it) {
    cout << pp_tri(*it) << endl;
  }
}

TEST(cut5) {
  Triangle triangle(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
  Segment_3 edge1(Point_2(0, 0), Point_2(1, 0));
  Segment_3 edge2(Point_2(1, 0), Point_2(0, 1));
  list<Point_3> points;
  points.push_back(Point_3(0.5, 0, 0));
  points.push_back(Point_3(0.75, 0, 0));
  boost::unordered_map<Segment_3_undirected, list<Point_3>> edge2points;

  edge2points[edge1] = points;
  edge2points[edge2] = points;

  list<Triangle> triangles;

  bool caught = false;
  try {
    decompose_triangle(triangle, edge2points, back_inserter(triangles));
  } catch (logic_error &e) {
    caught = true;
  }
  CHECK(caught);

  //   CHECK(index(Point_2(0,0), t1) > -1);
  //   CHECK(index(Point_2(0.5,0), t1) > -1);
  //   CHECK(index(Point_2(0,1), t1) > -1);

  //   CHECK(index(Point_2(1,0), t2) > -1);
  //   CHECK(index(Point_2(0.5,0), t2) > -1);
  //   CHECK(index(Point_2(0,1), t2) > -1);

  for (list<Triangle>::const_iterator it = triangles.begin();
       it != triangles.end(); ++it) {
    cout << pp_tri(*it) << endl;
  }
}
