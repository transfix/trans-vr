#ifndef __READER_RAW_H__
#define __READER_RAW_H__

#include <ContourTiler/common.h>
#include <ContourTiler/triangle_utils.h>
#include <fstream>
#include <iostream>
#include <list>

CONTOURTILER_BEGIN_NAMESPACE

template <typename OutputIterator>
void read_triangles_raw(std::istream &in, OutputIterator triangles,
                        bool color) {

  int num_points, num_triangles;
  in >> num_points >> num_triangles;

  vector<Point_3> points;
  for (int i = 0; i < num_points; ++i) {
    Point_3 p;
    in >> p;
    points.push_back(p);
    if (color) {
      double r, g, b;
      in >> r >> g >> b;
    }
  }

  for (int i = 0; i < num_triangles; ++i) {
    int a, b, c;
    in >> a >> b >> c;
    *triangles = Triangle(points[a], points[b], points[c]);
    ++triangles;
  }
}

template <typename OutputIterator>
void read_triangles_raw(const std::string &filename, OutputIterator triangles,
                        bool color) {
  std::ifstream in(filename.c_str());
  read_triangles_raw(in, triangles, color);
  in.close();
}
CONTOURTILER_END_NAMESPACE

#endif
