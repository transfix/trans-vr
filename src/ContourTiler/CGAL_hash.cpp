#include <ContourTiler/CGAL_hash.h>

#include <vector>

namespace CGAL {

//   std::size_t hash_value(const CGAL::Point_2<CGAL::Cartesian<double> >& point)
//   {
//     std::size_t seed = 0;
//     boost::hash_combine(seed, point.x());
//     boost::hash_combine(seed, point.y());
//     return seed;
//   }

  template <typename Kernel>
  std::size_t hash_value(const CGAL::Point_2<Kernel>& point)
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, to_double(point.x()));
    boost::hash_combine(seed, to_double(point.y()));
    return seed;
  }

  template <typename Kernel>
  std::size_t hash_value(const CGAL::Point_3<Kernel>& point)
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, to_double(point.x()));
    boost::hash_combine(seed, to_double(point.y()));
    boost::hash_combine(seed, to_double(point.z()));
    return seed;
  }
//   std::size_t hash_value(const CGAL::Point_3<CGAL::Cartesian<double> >& point)
//   {
//     std::size_t seed = 0;
//     boost::hash_combine(seed, point.x());
//     boost::hash_combine(seed, point.y());
//     boost::hash_combine(seed, point.z());
//     return seed;
//   }
  template <typename Kernel>
  std::size_t hash_value(const CGAL::Segment_3<Kernel>& segment)
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, segment.source());
    boost::hash_combine(seed, segment.target());
    return seed;
  }

  template <typename Kernel>
  std::size_t hash_value(const CGAL::Segment_2<Kernel>& segment)
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, segment.source());
    boost::hash_combine(seed, segment.target());
    return seed;
  }

//   std::size_t hash_value(const CGAL::Segment_3<Cartesian_25<double> >& segment)
//   {
//     std::size_t seed = 0;
//     boost::hash_combine(seed, segment.source());
//     boost::hash_combine(seed, segment.target());
//     return seed;
//   }


//   template std::size_t hash_value<CGAL::Cartesian<double> >(const CGAL::Point_3<CGAL::Cartesian<double> >& point);
//   template std::size_t hash_value<Cartesian_25<double> >(const CGAL::Point_3<Cartesian_25<double> >& point);
//   template std::size_t hash_value<CGAL::Cartesian<double> >(const CGAL::Segment_3<CGAL::Cartesian<double> >& point);
//   template std::size_t hash_value<Cartesian_25<double> >(const CGAL::Segment_3<Cartesian_25<double> >& point);

  template std::size_t hash_value<CGAL::Cartesian<Number_type> >(const CGAL::Point_2<CGAL::Cartesian<Number_type> >& point);
  template std::size_t hash_value<Cartesian_25<Number_type> >(const CGAL::Point_2<Cartesian_25<Number_type> >& point);
  template std::size_t hash_value<CGAL::Cartesian<Number_type> >(const CGAL::Point_3<CGAL::Cartesian<Number_type> >& point);
  template std::size_t hash_value<Cartesian_25<Number_type> >(const CGAL::Point_3<Cartesian_25<Number_type> >& point);
  template std::size_t hash_value<CGAL::Cartesian<Number_type> >(const CGAL::Segment_3<CGAL::Cartesian<Number_type> >& point);
  template std::size_t hash_value<Cartesian_25<Number_type> >(const CGAL::Segment_3<Cartesian_25<Number_type> >& point);
  template std::size_t hash_value<CGAL::Cartesian<Number_type> >(const CGAL::Segment_2<CGAL::Cartesian<Number_type> >& point);
  template std::size_t hash_value<Cartesian_25<Number_type> >(const CGAL::Segment_2<Cartesian_25<Number_type> >& point);
}
