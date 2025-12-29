#ifndef __SER_READER_H__
#define __SER_READER_H__

// Make sure TIXML_USE_STL is defined

#include <boost/foreach.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

//------------------------------------------------------------
// class Ser_exception
//------------------------------------------------------------
class Ser_exception : public std::runtime_error {
public:
  Ser_exception(const std::string &error) : std::runtime_error(error) {}
  ~Ser_exception() throw() {}
};

//------------------------------------------------------------
// class Ser_point
//------------------------------------------------------------
class Ser_point {
public:
  Ser_point(double x, double y) : _x(x), _y(y) {}
  double x() const { return _x; }
  double y() const { return _y; }

private:
  double _x, _y;
};

//------------------------------------------------------------
// class Ser_contour
//------------------------------------------------------------
class Ser_contour {
public:
  typedef std::vector<Ser_point>::iterator iterator;
  typedef std::vector<Ser_point>::const_iterator const_iterator;

public:
  Ser_contour() {}
  template <typename Iter>
  Ser_contour(const std::string &name, Iter points_begin, Iter points_end)
      : _name(name), _points(points_begin, points_end) {}

  const std::string &name() const { return _name; }
  size_t size() const { return _points.size(); }

  iterator begin() { return _points.begin(); }
  const_iterator begin() const { return _points.begin(); }
  iterator end() { return _points.end(); }
  const_iterator end() const { return _points.end(); }

  friend std::ostream &operator<<(std::ostream &out,
                                  const Ser_contour &contour) {
    out << contour.name() << " size=" << contour.size();
    return out;
  }

private:
  std::string _name;
  std::vector<Ser_point> _points;
};

//------------------------------------------------------------
// class Ser_section
//------------------------------------------------------------
class Ser_section {
public:
  typedef std::vector<Ser_contour>::iterator iterator;
  typedef std::vector<Ser_contour>::const_iterator const_iterator;

public:
  Ser_section() {}
  template <typename Iter>
  Ser_section(int index, double thickness, Iter contour_begin,
              Iter contour_end)
      : _index(index), _thickness(thickness),
        _contours(contour_begin, contour_end) {}

  int index() const { return _index; }
  double thickness() const { return _thickness; }
  size_t size() const { return _contours.size(); }

  iterator begin() { return _contours.begin(); }
  const_iterator begin() const { return _contours.begin(); }
  iterator end() { return _contours.end(); }
  const_iterator end() const { return _contours.end(); }

  friend std::ostream &operator<<(std::ostream &out,
                                  const Ser_section &section) {
    out << "section " << section.index() << " size=" << section.size();
    return out;
  }

private:
  int _index;
  double _thickness;
  std::vector<Ser_contour> _contours;
};

//------------------------------------------------------------
// helper function
//------------------------------------------------------------
Ser_section process_section(const std::string &filebase, int index,
                            std::vector<std::string> &warnings,
                            bool apply_transforms);

//------------------------------------------------------------
// The main function to call:
//     vector<Ser_section> sections;
//     vector<string> warnings;
//     try {
//       read_ser(filebase, first, last, back_inserter(sections),
//       back_inserter(warnings), false);
//     }
//     catch (Ser_exception& e) {
//       cout << "Error: " << e.what() << endl;
//     }
//------------------------------------------------------------
template <typename Section_iter, typename Warning_iter>
void read_ser(const std::string &filebase, int first_section,
              int last_section, Section_iter sections, Warning_iter warnings,
              bool apply_transforms) {
  std::vector<std::string> warns;
  for (int i = first_section; i <= last_section; ++i) {
    *sections++ = process_section(filebase, i, warns, apply_transforms);
  }
  BOOST_FOREACH (const std::string &w, warns) {
    *warnings++ = w;
  }
}

#endif
