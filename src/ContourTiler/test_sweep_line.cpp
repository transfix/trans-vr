#include <ContourTiler/test_common.h>
#include <ContourTiler/sweep_line_visitors.h>
#include <ContourTiler/print_utils.h>

template <typename Segment>
void sl_test(vector<Segment>& segments)
{
  cout << "All" << endl;
  sweep_line(segments.begin(), segments.end(), Debug_visitor());
  cout << "No end-end" << endl;
  sweep_line(segments.begin(), segments.end(), filter_visitor(Debug_visitor(), true, false));
  cout << "No end-int" << endl;
  sweep_line(segments.begin(), segments.end(), filter_visitor(Debug_visitor(), false, true));
}

TEST (sweep_line1)
{
  vector<Segment_3> segments;
  segments.push_back(Segment_3(Point_2(0,0), Point_2(0,2)));
  segments.push_back(Segment_3(Point_2(1,0), Point_2(2,2)));
  segments.push_back(Segment_3(Point_2(2,0), Point_2(4,2)));
  segments.push_back(Segment_3(Point_2(0.5,1), Point_2(2.5,1)));
  sl_test(segments);
}

TEST (sweep_line2)
{
  vector<Segment_3> segments;
  segments.push_back(Segment_3(Point_2(0,0), Point_2(2,0)));
  segments.push_back(Segment_3(Point_2(1,0), Point_2(3,0)));
  sl_test(segments);
}

TEST (sweep_line3)
{
  vector<Segment_3> segments;
  segments.push_back(Segment_3(Point_2(0,0), Point_2(0,2)));
  segments.push_back(Segment_3(Point_2(0,1), Point_2(0,3)));
  sl_test(segments);
}

TEST (sweep_line4)
{
  vector<Segment_3> segments;
  segments.push_back(Segment_3(Point_3(0,0,0), Point_3(3,0,0)));
  segments.push_back(Segment_3(Point_3(0,0,1), Point_3(3,0,1)));
  sl_test(segments);
}

TEST (sweep_line5)
{
  vector<Segment_3> segments;
  segments.push_back(Segment_3(Point_2(0,0), Point_2(2,0)));
  segments.push_back(Segment_3(Point_2(1,0), Point_2(2,0)));
  sl_test(segments);
}

TEST (sweep_line6)
{
  vector<Segment_3> segments;
  segments.push_back(Segment_3(Point_2(0,0), Point_2(1,0)));
  segments.push_back(Segment_3(Point_2(1,0), Point_2(2,0)));
  sl_test(segments);
}

TEST (sweep_line7)
{
  Polygon_2 p;
  p.push_back(Point_2(5.83641, 2.91));
  p.push_back(Point_2(5.83641, 2.80082));
  p.push_back(Point_2(5.84909, 2.79131));
  p.push_back(Point_2(5.86178, 2.79131));
  p.push_back(Point_2(5.88714, 2.78655));
  p.push_back(Point_2(5.89983, 2.78497));
  p.push_back(Point_2(5.90617, 2.7818));
  p.push_back(Point_2(5.91251, 2.7818));
  p.push_back(Point_2(5.96166, 2.76594));
  p.push_back(Point_2(5.97434, 2.76753));
  p.push_back(Point_2(5.97434, 2.91));
  vector<Segment_2> segments(p.edges_begin(), p.edges_end());
//   segments.push_back(Segment_2(Point_2(5.7, 2.7818), Point_2(5.92, 2.7818)));

//   sl_test(segments);
  list<SL_intersection> intersections;
  get_intersections(segments.begin(), segments.end(), back_inserter(intersections), false, false);
//   cout << "Intersections:" << endl;
//   for (list<SL_intersection>::iterator it = intersections.begin(); it != intersections.end(); ++it)
//     cout << pp(it->point()) << " " << pp(it->a()) << " " << pp(it->b()) << endl;

//   cout << "Intersections? " << has_intersection(segments.begin(), segments.end(), false, false) << endl;
}

TEST (sweep_line8)
{
  Polygon_2 p;
  p.push_back(Point_2(0,0));
  p.push_back(Point_2(0,2));
  p.push_back(Point_2(2,0));
  vector<Segment_2> segments(p.edges_begin(), p.edges_end());
  sl_test(segments);
}

TEST (sweep_line9)
{
  vector<Segment_2> segments;
  segments.push_back(Segment_2(Point_2(0,0), Point_2(3,0)));
  segments.push_back(Segment_2(Point_2(1,0), Point_2(2,0)));
  sl_test(segments);
}

TEST (sweep_line10)
{
  Polygon_2 P, Q, Pe, Qe;
  P.push_back(Point_2(0,0,0));
  P.push_back(Point_2(4,0,0));
  P.push_back(Point_2(4,4,0));
  P.push_back(Point_2(0,4,0));
  
  Q.push_back(Point_2(1,1,1));
  Q.push_back(Point_2(5,1,1));
  Q.push_back(Point_2(4,2,1));
  Q.push_back(Point_2(4,3,1));
  Q.push_back(Point_2(3,4,1));
  Q.push_back(Point_2(1,4,1));

  std::list<Segment_2> segments;
  segments.insert(segments.end(), P.edges_begin(), P.edges_end());
  segments.insert(segments.end(), Q.edges_begin(), Q.edges_end());

  list<SL_intersection> ints;
  get_intersections(segments.begin(), segments.end(), back_inserter(ints), true, false);

//   cout << "Num intersections: " << ints.size() << endl;

  typedef boost::unordered_map<Segment_2, list<Point_2> > Map;
  Map map;
  get_interior_only(ints.begin(), ints.end(), map);

//   for (Map::iterator it = map.begin(); it != map.end(); ++it)
//   {
//     for (list<Point_2>::iterator pit = it->second.begin(); pit != it->second.end(); ++pit)
//       cout << pp(it->first) << " " << pp(*pit) << endl;
//   }
}

TEST (sweep_line11)
{
  vector<Segment_2> segments0, segments1, all;
  segments0.push_back(Segment_2(Point_2(0,0), Point_2(1,0)));
  segments0.push_back(Segment_2(Point_2(1,0), Point_2(1,1)));
  segments0.push_back(Segment_2(Point_2(1,1), Point_2(0,1)));
  segments0.push_back(Segment_2(Point_2(0,1), Point_2(0,0)));

  segments1.push_back(Segment_2(Point_2(1,1), Point_2(2,1)));
  segments1.push_back(Segment_2(Point_2(2,1), Point_2(2,2)));
  segments1.push_back(Segment_2(Point_2(2,2), Point_2(1,2)));
  segments1.push_back(Segment_2(Point_2(1,2), Point_2(1,1)));

  all.insert(all.end(), segments0.begin(), segments0.end());
  all.insert(all.end(), segments1.begin(), segments1.end());

//   sl_test(all);

//   cout << "All" << endl;
//   sweep_line(segments0.begin(), segments0.end(), segments1.begin(), segments1.end(), Debug_visitor());
//   cout << "No end-end" << endl;
//   sweep_line(segments0.begin(), segments0.end(), segments1.begin(), segments1.end(), filter_visitor(Debug_visitor(), true, false));
//   cout << "No end-int" << endl;
//   sweep_line(segments0.begin(), segments0.end(), segments1.begin(), segments1.end(), filter_visitor(Debug_visitor(), false, true));
}

TEST (sweep_line12)
{
  vector<Segment_2> segments0, segments1;
  segments0.push_back(Segment_2(Point_2(5.16883,6.42268,114), Point_2(5.16883,6.41973,114)));
  segments0.push_back(Segment_2(Point_2(5.16883,6.41973,114), Point_2(5.16883,6.41678,114)));
  segments1.push_back(Segment_2(Point_2(5.16883,6.42268,113), Point_2(5.16883,6.41973,113)));

//   segments.push_back(Segment_2(Point_2(10,10), Point_2(5,5)));
//   segments.push_back(Segment_2(Point_2(10,10), Point_2(5,5)));

  typedef list<SL_intersection>::const_iterator i_iter;
  list<SL_intersection> ints;
  get_intersections(segments0.begin(), segments0.end(), 
		    segments1.begin(), segments1.end(), 
		    back_inserter(ints), true, false);
  cout << ints.size() << endl;

  typedef boost::unordered_map<Segment_2, list<Point_2> > Map;
  Map map;
  get_interior_only(ints.begin(), ints.end(), map);
//   sl_test(segments);
}

