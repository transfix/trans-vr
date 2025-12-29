#ifndef __READER_PTS_H__
#define __READER_PTS_H__

#include <ContourTiler/Contour.h>
#include <fstream>
#include <iostream>
#include <list>

CONTOURTILER_BEGIN_NAMESPACE

template <typename OutputIterator>
void read_polygons_pts(std::istream &in, OutputIterator polygons) {
  while (!in.eof()) {
    Polygon polygon;
    in >> polygon;
    if (!polygon.is_empty()) {
      *polygons = polygon;
      ++polygons;
    }
  }
}

template <typename OutputIterator>
void read_polygons_pts(const std::string &filename, OutputIterator polygons) {
  std::ifstream in(filename.c_str());
  read_polygons_pts<Polygon>(in, polygons);
  in.close();
}

template <typename OutputIterator>
void read_contours_pts(std::istream &in, OutputIterator contours, int z = 0) {
  typedef typename Contour::Handle Contour_handle;
  typedef typename Contour::Polygon Polygon;
  typedef std::list<Polygon> Polygon_list;

  Polygon_list polygons;
  read_polygons_pts(in, std::back_inserter(polygons));
  typename Polygon_list::const_iterator it;
  for (it = polygons.begin(); it != polygons.end(); ++it) {
    Contour_handle contour = Contour::create(*it);
    contour->info().slice() = z;
    *contours = contour;
    ++contours;
  }
}

template <typename OutputIterator>
void read_contours_pts(const std::string &filename, OutputIterator contours,
                       int z = 0) {
  std::ifstream in(filename.c_str());
  read_contours_pts<Contour>(in, contours, z);
  in.close();
}

CONTOURTILER_END_NAMESPACE

#endif
