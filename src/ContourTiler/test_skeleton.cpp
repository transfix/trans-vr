#include <ContourTiler/test_common.h>
#include <ContourTiler/skeleton.h>
#include <ContourTiler/triangle_utils.h>
#include <ContourTiler/print_utils.h>
#include <ContourTiler/reader_gnuplot.h>

TEST (to_simple1)
{
  vector<Polygon_2> temp;
  read_polygons_gnuplot2(data_dir+"/simple1.dat", back_inserter(temp), 1);
  Polygon_2 P = temp[0];
  Untiled_region region(P.vertices_begin(), P.vertices_end());
  to_simple(region);
}

void skeleton_test(const Untiled_region& P)
{
  list<Triangle> tiles;
//   medial_axis_stable(P, 0.5, back_inserter(tiles));

//   for (list<Triangle>::iterator it = tiles.begin(); it != tiles.end(); ++it)
//     cout << pp_tri(*it) << endl;
}

TEST (skeleton1)
{
  Untiled_region P;
  P.push_back(Point_2(0,0));
  P.push_back(Point_2(2,0));
  P.push_back(Point_2(0,2));
  skeleton_test(P);
}

TEST (skeleton2)
{
  Untiled_region P;
  P.push_back(Point_2(0,0));
  P.push_back(Point_2(1,0));
  P.push_back(Point_2(2,0));
  P.push_back(Point_2(0,2));
  skeleton_test(P);
}

TEST (skeleton3)
{
  Untiled_region P;
  P.push_back(Point_2(0,0));
  P.push_back(Point_2(1,-0.00000001));
  P.push_back(Point_2(2,0));
  P.push_back(Point_2(0,2));
  skeleton_test(P);
}

TEST (skeleton4)
{
  Untiled_region P;
  P.push_back(Point_2(0,0));
  P.push_back(Point_2(1,0.00000001));
  P.push_back(Point_2(2,0));
  P.push_back(Point_2(0,2));
  skeleton_test(P);
}

TEST (skeleton5)
{
  Untiled_region P;
  P.push_back(Point_3(0,0,0));
  P.push_back(Point_3(2,0,0));
  P.push_back(Point_3(2,0,1));
  P.push_back(Point_3(2,2,1));
  P.push_back(Point_3(2,2,0));
  P.push_back(Point_3(0,2,0));

  vector<Triangle> tiles;
//   medial_axis_stable(P, 0.5, back_inserter(tiles));

//   CHECK_EQUAL(Point_3(0,0,0), tiles[0][0]);
//   CHECK_EQUAL(Point_3(2,0,0), tiles[0][1]);
//   CHECK_EQUAL(Point_3(1,1,0.5), tiles[0][2]);
//   CHECK_EQUAL(Point_3(2,0,0), tiles[1][0]);
//   CHECK_EQUAL(Point_3(2,0,1), tiles[1][1]);
//   CHECK_EQUAL(Point_3(1,1,0.5), tiles[1][2]);
//   CHECK_EQUAL(Point_3(2,0,1), tiles[2][0]);
//   CHECK_EQUAL(Point_3(2,2,1), tiles[2][1]);
//   CHECK_EQUAL(Point_3(1,1,0.5), tiles[2][2]);
//   CHECK_EQUAL(Point_3(2,2,1), tiles[3][0]);
//   CHECK_EQUAL(Point_3(2,2,0), tiles[3][1]);
//   CHECK_EQUAL(Point_3(1,1,0.5), tiles[3][2]);
//   CHECK_EQUAL(Point_3(2,2,0), tiles[4][0]);
//   CHECK_EQUAL(Point_3(0,2,0), tiles[4][1]);
//   CHECK_EQUAL(Point_3(1,1,0.5), tiles[4][2]);
//   CHECK_EQUAL(Point_3(0,2,0), tiles[5][0]);
//   CHECK_EQUAL(Point_3(0,0,0), tiles[5][1]);
//   CHECK_EQUAL(Point_3(1,1,0.5), tiles[5][2]);
}

TEST (skeleton6)
{
  Untiled_region P;
  P.push_back(Point_3(0,1,0));
  P.push_back(Point_3(1,1,0));
  P.push_back(Point_3(1,1,1));
  P.push_back(Point_3(2,0,1));
  P.push_back(Point_3(2,3,1));
  P.push_back(Point_3(1,2,1));
  P.push_back(Point_3(1,2,0));
  P.push_back(Point_3(1,3,0));
  P.push_back(Point_3(0,2,0));

  vector<Triangle> tiles;
//   medial_axis_stable(P, 0.5, back_inserter(tiles));

//   ofstream r("output/test.raw");
//   raw_print_tiles(r, tiles.begin(), tiles.end(), 1, 1, 1);
//   r.close();
}

TEST (skeleton7)
{
  Untiled_region P;
  P.push_back(Point_3(0,1,0));
  P.push_back(Point_3(1,1,0));
  P.push_back(Point_3(1,1,1));
  P.push_back(Point_3(2,0,1));
  P.push_back(Point_3(2,3,1));
  P.push_back(Point_3(2,3,0));
  P.push_back(Point_3(1,2,0));
  P.push_back(Point_3(1,3,0));
  P.push_back(Point_3(0,2,0));

  vector<Triangle> tiles;
//   medial_axis_stable(P, 0.5, back_inserter(tiles));

//   ofstream r("output/test.raw");
//   raw_print_tiles(r, tiles.begin(), tiles.end(), 1, 1, 1);
//   r.close();
}

TEST (skeleton8)
{
  Untiled_region P;
  P.push_back(Point_3(0,2,0));
  P.push_back(Point_3(1.5,1.5,0));
  P.push_back(Point_3(2,0,0));
  P.push_back(Point_3(2.5,1.5,0));
  P.push_back(Point_3(4,2,0));
  P.push_back(Point_3(2.5,2.5,0));
  P.push_back(Point_3(2,4,0));
  P.push_back(Point_3(1.5,2.5,0));

  vector<Triangle> tiles;
//   medial_axis_stable(P, 0.5, back_inserter(tiles));

//   ofstream r("output/test.raw");
//   raw_print_tiles(r, tiles.begin(), tiles.end(), 1, 1, 1);
//   r.close();
}

