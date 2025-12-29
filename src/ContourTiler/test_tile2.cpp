#include <ContourTiler/mtiler_operations.h>
#include <ContourTiler/reader_gnuplot.h>
#include <ContourTiler/test_common.h>
#include <ContourTiler/test_tile2_expected.h>
#include <ContourTiler/test_utils.h>
#include <ContourTiler/tiler.h>
#include <string>

template <typename File_iter>
string tile_test(File_iter files_begin, File_iter files_end,
                 double z_scale = 1) {
  vector<vector<Contour_handle>> slices;

  int i = 0;
  for (File_iter it = files_begin; it != files_end; ++it) {
    vector<Contour_handle> slice;
    Number_type z = i + 1;
    //     z = (i % 2) + 1; // for multi
    read_contours_gnuplot2(*it, back_inserter(slice), z);
    slices.push_back(slice);
    ++i;
  }

  Tiler_options options;
  options.output_dir() = "output";
  options.base_name() = "test";
  options.output_raw() = true;
  options.output_gnuplot() = false;
  //   TW_handle tw = tile(slices[0].begin(), slices[0].end(),
  //   slices[1].begin(), slices[1].end(), options);

  vector<Slice> slices_(2);
  slices_[0].insert("a002", slices[0].begin(), slices[0].end());
  slices_[1].insert("a002", slices[1].begin(), slices[1].end());
  boost::unordered_map<string, Color> comp2color;
  tile(slices_.begin(), slices_.end(), comp2color, options);

  ifstream in("output/test_tiles.raw");
  string nraw = normalized_raw(in, true);
  in.close();

  return nraw;
}

TEST(tile1) {
  vector<string> files;
  files.push_back("../test_data/test1_1.dat");
  files.push_back("../test_data/test1_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile1_exp, raw);
}

TEST(tile2) {
  vector<string> files;
  files.push_back("../test_data/test2_1.dat");
  files.push_back("../test_data/test2_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile2_exp, raw);
}

TEST(tile3) {
  vector<string> files;
  files.push_back("../test_data/test3_1.dat");
  files.push_back("../test_data/test3_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile3_exp, raw);
}

TEST(tile4) {
  vector<string> files;
  files.push_back("../test_data/test4_1.dat");
  files.push_back("../test_data/test4_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile4_exp, raw);
}

TEST(tile5) {
  vector<string> files;
  files.push_back("../test_data/test5_1.dat");
  files.push_back("../test_data/test5_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile5_exp, raw);
}

TEST(tile6) {
  vector<string> files;
  files.push_back("../test_data/test6_1.dat");
  files.push_back("../test_data/test6_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile6_exp, raw);
}

TEST(tile7) {
  vector<string> files;
  files.push_back("../test_data/test7_1.dat");
  files.push_back("../test_data/test7_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile7_exp, raw);
}

TEST(tile8) {
  vector<string> files;
  files.push_back("../test_data/test8_1.dat");
  files.push_back("../test_data/test8_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile8_exp, raw);
}

// TEST (tile9)
// {
//   vector<string> files;
//   files.push_back("../test_data/test9_1.dat");
//   files.push_back("../test_data/test9_2.dat");

//   string raw = tile_test(files.begin(), files.end());
//   CHECK_EQUAL(tile9_12_exp, raw);

//   files.clear();
//   files.push_back("../test_data/test9_3.dat");
//   files.push_back("../test_data/test9_4.dat");

//   raw = tile_test(files.begin(), files.end());
//   CHECK_EQUAL(tile9_34_exp, raw);
// }

// TEST (tile10)
// {
//   vector<string> files;
//   files.push_back("../test_data/test10_1.dat");
//   files.push_back("../test_data/test10_2.dat");

//   string raw = tile_test(files.begin(), files.end(), 0.04);
//   CHECK_EQUAL(tile10_exp, raw);
// }

TEST(tile11) {
  vector<string> files;
  files.push_back("../test_data/test11_1.dat");
  files.push_back("../test_data/test11_2.dat");

  string raw = tile_test(files.begin(), files.end(), 0.04);
  //   CHECK_EQUAL(tile11_exp, raw);
}

TEST(tile12) {
  vector<string> files;
  files.push_back("../test_data/test12_1.dat");
  files.push_back("../test_data/test12_2.dat");

  string raw = tile_test(files.begin(), files.end());
  //   CHECK_EQUAL(tile12_exp, raw);
}

TEST(tile13) {
  vector<string> files;
  files.push_back("../test_data/test13_1.dat");
  files.push_back("../test_data/test13_2.dat");

  string raw = tile_test(files.begin(), files.end());
  //   CHECK_EQUAL(tile13_exp, raw);
}

TEST(tile14) {
  vector<string> files;
  files.push_back("../test_data/test14_1.dat");
  files.push_back("../test_data/test14_2.dat");

  string raw = tile_test(files.begin(), files.end(), 0.04);
  //   CHECK_EQUAL(tile14_exp, raw);
}

TEST(tile15) {
  vector<string> files;
  files.push_back("../test_data/test15_1.dat");
  files.push_back("../test_data/test15_2.dat");

  string raw = tile_test(files.begin(), files.end(), 0.04);
  //   CHECK_EQUAL(tile15_exp, raw);
}

TEST(tile16) {
  vector<string> files;
  files.push_back("../test_data/test16_1.dat");
  files.push_back("../test_data/test16_2.dat");

  string raw = tile_test(files.begin(), files.end(), 0.04);
  //   CHECK_EQUAL(tile16_exp, raw);
}

TEST(tile17) {
  vector<string> files;
  files.push_back("../test_data/test17_1.dat");
  files.push_back("../test_data/test17_2.dat");

  string raw = tile_test(files.begin(), files.end(), 0.04);
  //   CHECK_EQUAL(tile17_exp, raw);
}

TEST(tile29) {
  vector<string> files;
  files.push_back("../test_data/test29_1.dat");
  files.push_back("../test_data/test29_2.dat");

  string raw = tile_test(files.begin(), files.end());
  CHECK_EQUAL(tile29_exp, raw);
}
