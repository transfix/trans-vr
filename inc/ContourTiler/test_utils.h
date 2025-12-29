#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <ContourTiler/print_utils.h>
#include <ContourTiler/reader_raw.h>
#include <ContourTiler/test_common.h>
#include <ContourTiler/triangle_utils.h>

CONTOURTILER_BEGIN_NAMESPACE

template <typename Tri> bool tris_equal(const Tri &a, const Tri &b) {
  for (int i = 0; i < 3; ++i)
    if (index(a[i], b) == -1)
      return false;
  return true;
}

class x_comparator {
public:
  template <typename Tri> bool operator()(const Tri &a, const Tri &b) {
    if (a[0] == b[0]) {
      if (a[1] == b[1])
        return CGAL::lexicographically_xyz_smaller(a[2], b[2]);
      return CGAL::lexicographically_xyz_smaller(a[1], b[1]);
    }
    return CGAL::lexicographically_xyz_smaller(a[0], b[0]);
  }
};

class l_order {
public:
  template <typename Tri> void operator()(Tri &t) {
    int idx = 0;
    for (int i = 1; i < 3; ++i) {
      if (CGAL::lexicographically_xyz_smaller(t[i], t[idx]))
        idx = i;
    }
    int a = idx;
    int b = next_idx(a);
    int c = next_idx(b);
    if (CGAL::lexicographically_xyz_smaller(t[c], t[b]))
      std::swap(b, c);
    t = Tri(t[a], t[b], t[c]);
  }
};

template <typename Out_iter>
void normalized_raw(std::istream &in, bool color, Out_iter triangles) {
  vector<Triangle> t;
  read_triangles_raw(in, back_inserter(t), color);

  std::for_each(t.begin(), t.end(), l_order());
  std::sort(t.begin(), t.end(), x_comparator());

  for (int i = 0; i < t.size(); ++i) {
    *triangles = t[i];
    ++triangles;
  }
}

inline std::string normalized_raw(std::istream &in, bool color) {
  vector<Triangle> a;
  normalized_raw(in, color, back_inserter(a));

  stringstream out;
  raw_print_tiles(out, a.begin(), a.end(), 1, 1, 1);
  return out.str();
}

CONTOURTILER_END_NAMESPACE

#endif
