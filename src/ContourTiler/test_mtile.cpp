#include <string>

#include <ContourTiler/test_common.h>
#include <ContourTiler/mtiler_operations.h>
#include <ContourTiler/tiler.h>
#include <ContourTiler/reader_gnuplot.h>

// template <typename File_iter>
// void mtile_test(File_iter files_begin, File_iter files_end)
// {
// //   vector<vector<Contour_handle> > slices;
//   vector<Slice> slices;

//   int i = 0;
//   for (File_iter it = files_begin; it != files_end; ++it)
//   {
//     vector<Contour_handle> slice;
//     Number_type z = i + 1;
//     z = (i % 2) + 1; // for multi
//     read_contours_gnuplot2(*it, back_inserter(slice), z);
//     slices.push_back(slice);
//     Slice s;
//     s.
//     ++i;
//   }

//   Tiler_options options;
//   options.output_dir() = "output";
//   options.output_raw() = true;
//   options.output_gnuplot() = true;
//   options.multi() = true;
//   tile(slices, options);
// }

// // ./test_dat -m test21.dat test22.dat test23.dat test24.dat
// TEST (mtile1)
// {
//   vector<string> files;
//   files.push_back("../test_data/test21.dat");
//   files.push_back("../test_data/test22.dat");
//   files.push_back("../test_data/test23.dat");
//   files.push_back("../test_data/test24.dat");

//   mtile_test(files.begin(), files.end());
// }

// ./test_dat -m test9_1.dat test9_2.dat test9_3.dat test9_4.dat
// TEST (mtile2)
// {
//   vector<string> files;
//   files.push_back("../test_data/test9_1.dat");
//   files.push_back("../test_data/test9_2.dat");
//   files.push_back("../test_data/test9_3.dat");
//   files.push_back("../test_data/test9_4.dat");

//   mtile_test(files.begin(), files.end());
// }

