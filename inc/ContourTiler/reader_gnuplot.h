#ifndef __READER_GNUPLOT_H__
#define __READER_GNUPLOT_H__

#include <ContourTiler/Contour.h>
#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include <list>

CONTOURTILER_BEGIN_NAMESPACE

template <typename OutputIterator>
void read_polygons_gnuplot2(std::istream &in, OutputIterator polygons,
                            Number_type z) {
  typedef typename Polygon_2::Point_2 Point;
  using namespace std;
  using namespace boost;

  static const boost::regex whitespace("^[ \t]*$");

  string line;
  Point point;
  getline(in, line);
  while (!in.eof()) {
    // find the first point
    while (!in.eof() && regex_match(line, whitespace))
      std::getline(in, line);

    // read in the polygon
    Polygon_2 polygon;
    while (!regex_match(line, whitespace)) {
      std::stringstream ss(line);
      double x, y;
      ss >> x >> y;
      point = Point(x, y, z);
      //       point.z() = z;
      polygon.push_back(point);
      line = "";
      std::getline(in, line);
    }

    if (!polygon.is_empty()) {
      // gnuplot repeats the last point, so erase it
      polygon.erase(polygon.vertices_end() - 1);
      *polygons = polygon;
      ++polygons;
    }
  }
}

template <typename OutputIterator>
void read_polygons_gnuplot2(const std::string &filename,
                            OutputIterator polygons, Number_type z) {
  std::ifstream in(filename.c_str());
  read_polygons_gnuplot2(in, polygons, z);
  in.close();
}

template <typename OutputIterator>
void read_contours_gnuplot2(std::istream &in, OutputIterator contours,
                            Number_type z) {
  //   typedef typename Contour::Handle Contour_handle;
  //   typedef typename Contour::Polygon_2 Polygon_2;
  typedef std::list<Polygon_2> Polygon_list;

  Polygon_list polygons;
  read_polygons_gnuplot2(in, std::back_inserter(polygons), z);
  typename Polygon_list::const_iterator it;
  for (it = polygons.begin(); it != polygons.end(); ++it) {
    Contour_handle contour = Contour::create(*it);
    contour->info().slice() = z;
    *contours = contour;
    ++contours;
  }
}

template <typename OutputIterator>
void read_contours_gnuplot2(const std::string &filename,
                            OutputIterator contours, Number_type z) {
  std::ifstream in(filename.c_str());
  read_contours_gnuplot2(in, contours, z);
  in.close();
}

template <typename OutputIterator>
void read_polygons_gnuplot3(std::istream &in, OutputIterator polygons) {
  typedef typename Polygon_2::Point_2 Point;
  using namespace std;
  using namespace boost;

  static const boost::regex whitespace("^[ \t]*$");

  string line;
  Point point;
  getline(in, line);
  while (!in.eof()) {
    // find the first point
    while (!in.eof() && regex_match(line, whitespace))
      std::getline(in, line);

    // read in the polygon
    Polygon_2 polygon;
    while (!regex_match(line, whitespace)) {
      std::stringstream ss(line);
      double x, y, z;
      ss >> x >> y >> z;
      point = Point(x, y, z);
      //       point.z() = z;
      polygon.push_back(point);
      line = "";
      std::getline(in, line);
    }

    if (!polygon.is_empty()) {
      // gnuplot repeats the last point, so erase it
      polygon.erase(polygon.vertices_end() - 1);
      *polygons = polygon;
      ++polygons;
    }
  }
}

template <typename OutputIterator>
void read_polygons_gnuplot3(const std::string &filename,
                            OutputIterator polygons) {
  std::ifstream in(filename.c_str());
  read_polygons_gnuplot3(in, polygons);
  in.close();
}

CONTOURTILER_END_NAMESPACE

#endif
