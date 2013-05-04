#include <string>
#include <boost/shared_ptr.hpp>

#include <ContourTiler/test_common.h>
#include <ContourTiler/Tiling_region.h>
#include <ContourTiler/Wedge.h>

// typedef boost::shared_ptr<const Wedge> HWedge;

// TEST (Wedge3)
// {
//   HWedge w1 = Wedge::LS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
//   HWedge w2 = Wedge::LS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
//   HWedge w = w2;
//   CHECK_EQUAL(w, w1->intersection(w2));

//   w2 = Wedge::LS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 1));
//   w = w2;
//   CHECK_EQUAL(w, w1->intersection(w2));

//   w2 = Wedge::LS(Point_2(1, 1), Point_2(0, 0), Point_2(1, 0));
//   w = w2;
//   CHECK_EQUAL(w, w1->intersection(w2));

//   w2 = Wedge::LS(Point_2(1, 2), Point_2(0, 0), Point_2(2, 1));
//   w = w2;
//   CHECK_EQUAL(w, w1->intersection(w2));

//   w2 = Wedge::LS(Point_2(-1, 1), Point_2(0, 0), Point_2(1, 1));
//   w = Wedge::LS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 1));
//   CHECK_EQUAL(w, w1->intersection(w2));

//   w2 = Wedge::LS(Point_2(1, 1), Point_2(0, 0), Point_2(-1, 1));
//   w = Wedge::LS(Point_2(1, 1), Point_2(0, 0), Point_2(1, 0));
//   CHECK_EQUAL(w, w1->intersection(w2));

// }

// TEST (Wedge2)
// {
//   HWedge ls = Wedge::LS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
//   HWedge rs = Wedge::RS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
//   HWedge i = ls->intersection(rs);
//   CHECK(i->is_empty());
// }

// TEST (Wedge1)
// {
//   HWedge ls = Wedge::LS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
//   CHECK(ls->contains(Point_2(1, 1)));
//   CHECK(!ls->contains(Point_2(0, 1)));
//   CHECK(!ls->contains(Point_2(1, 0)));
//   CHECK(!ls->contains(Point_2(-1, -1)));

//   HWedge rs = Wedge::RS(Point_2(0, 1), Point_2(0, 0), Point_2(1, 0));
//   CHECK(!rs->contains(Point_2(1, 1)));
//   CHECK(!rs->contains(Point_2(0, 1)));
//   CHECK(!rs->contains(Point_2(1, 0)));
//   CHECK(rs->contains(Point_2(-1, -1)));
// }

TEST (Tiling_region1)
{
  HTiling_region region = Wedge::LS(Point_2(0, 0), Point_2(1, 0), Point_2(2, 0), 0);
  CHECK(!region->contains(Point_2(0, 0)));
  CHECK(!region->contains(Point_2(1, 0)));
  CHECK(!region->contains(Point_2(1.5, 0)));
  CHECK(region->contains(Point_2(1, 1)));
  CHECK(!region->contains(Point_2(-1000, 0)));
  CHECK(region->contains(Point_2(-1000, .0001)));
  CHECK(!region->contains(Point_2(1000, 0)));
  CHECK(region->contains(Point_2(1000, .0001)));

  CHECK(!region->contains(Point_2(1, -.0001)));
  CHECK(!region->contains(Point_2(1, -1000)));
  CHECK(!region->contains(Point_2(1000, -.0001)));
  CHECK(!region->contains(Point_2(1000, -1000)));
  CHECK(!region->contains(Point_2(-1000, -.0001)));
  CHECK(!region->contains(Point_2(-1000, -1000)));
}

TEST (Tiling_region1_2)
{
  HTiling_region region = Wedge::RS(Point_2(0, 0), Point_2(1, 0), Point_2(2, 0), 0);
  region = region->get_complement();
  CHECK(region->contains(Point_2(0, 0)));
  CHECK(region->contains(Point_2(1, 0)));
  CHECK(region->contains(Point_2(1.5, 0)));
  CHECK(region->contains(Point_2(1, 1)));
  CHECK(region->contains(Point_2(-1000, 0)));
  CHECK(region->contains(Point_2(-1000, .0001)));
  CHECK(region->contains(Point_2(1000, 0)));
  CHECK(region->contains(Point_2(1000, .0001)));

  CHECK(!region->contains(Point_2(1, -.0001)));
  CHECK(!region->contains(Point_2(1, -1000)));
  CHECK(!region->contains(Point_2(1000, -.0001)));
  CHECK(!region->contains(Point_2(1000, -1000)));
  CHECK(!region->contains(Point_2(-1000, -.0001)));
  CHECK(!region->contains(Point_2(-1000, -1000)));
}

TEST (Tiling_region2)
{
  HTiling_region region = Wedge::LS(Point_2(0, 0), Point_2(1, 0), Point_2(1, 1), 0);
  CHECK(!region->contains(Point_2(0, 0)));
  CHECK(!region->contains(Point_2(1, 0)));
  CHECK(!region->contains(Point_2(1.5, 0)));
  CHECK(!region->contains(Point_2(1, 1000)));
  CHECK(region->contains(Point_2(0, 1)));
  CHECK(!region->contains(Point_2(-1000, 0)));
  CHECK(region->contains(Point_2(-1000, .0001)));
  CHECK(!region->contains(Point_2(1000, 0)));
  CHECK(!region->contains(Point_2(1000, .0001)));

  CHECK(!region->contains(Point_2(1, -.0001)));
  CHECK(!region->contains(Point_2(1, -1000)));
  CHECK(!region->contains(Point_2(1000, -.0001)));
  CHECK(!region->contains(Point_2(1000, -1000)));
  CHECK(!region->contains(Point_2(-1000, -.0001)));
  CHECK(!region->contains(Point_2(-1000, -1000)));

  CHECK(region->contains(Segment_3(Point_2(0, 1), Point_2(0, 2))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(0, 2))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(2, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(2, 3))));
  CHECK(!region->contains(Segment_3(Point_2(0.5, -10), Point_2(10, 0.5))));
}

TEST (Tiling_region2_2)
{
  HTiling_region region = Wedge::RS(Point_2(0, 0), Point_2(1, 0), Point_2(1, 1), 0);
  region = region->get_complement();
  CHECK(region->contains(Point_2(0, 0)));
  CHECK(region->contains(Point_2(1, 0)));
  CHECK(!region->contains(Point_2(1.5, 0)));
  CHECK(region->contains(Point_2(1, 1000)));
  CHECK(region->contains(Point_2(0, 1)));
  CHECK(region->contains(Point_2(-1000, 0)));
  CHECK(region->contains(Point_2(-1000, .0001)));
  CHECK(!region->contains(Point_2(1000, 0)));
  CHECK(!region->contains(Point_2(1000, .0001)));

  CHECK(!region->contains(Point_2(1, -.0001)));
  CHECK(!region->contains(Point_2(1, -1000)));
  CHECK(!region->contains(Point_2(1000, -.0001)));
  CHECK(!region->contains(Point_2(1000, -1000)));
  CHECK(!region->contains(Point_2(-1000, -.0001)));
  CHECK(!region->contains(Point_2(-1000, -1000)));

  CHECK(region->contains(Segment_3(Point_2(0, 1), Point_2(0, 2))));
  CHECK(region->contains(Segment_3(Point_2(-1, 0), Point_2(0, 2))));
  CHECK(region->contains(Segment_3(Point_2(-1, 0), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(2, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(2, 3))));
  CHECK(!region->contains(Segment_3(Point_2(0.5, -10), Point_2(10, 0.5))));
}

TEST (Tiling_region3)
{
  HTiling_region region = Wedge::LS(Point_2(1, 1), Point_2(1, 0), Point_2(0, 0), 0);
  CHECK(!region->contains(Point_2(0, 0)));
  CHECK(!region->contains(Point_2(1, 0)));
  CHECK(region->contains(Point_2(1.5, 0)));
  CHECK(!region->contains(Point_2(1, 1000)));
  CHECK(!region->contains(Point_2(0, 1)));
  CHECK(!region->contains(Point_2(-1000, 0)));
  CHECK(!region->contains(Point_2(-1000, .0001)));
  CHECK(region->contains(Point_2(1000, 0)));
  CHECK(region->contains(Point_2(1000, .0001)));

  CHECK(region->contains(Point_2(1, -.0001)));
  CHECK(region->contains(Point_2(1, -1000)));
  CHECK(region->contains(Point_2(1000, -.0001)));
  CHECK(region->contains(Point_2(1000, -1000)));
  CHECK(region->contains(Point_2(-1000, -.0001)));
  CHECK(region->contains(Point_2(-1000, -1000)));

  CHECK(!region->contains(Segment_3(Point_2(0, 1), Point_2(0, 2))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(0, 2))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(2, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(2, 3))));
  CHECK(region->contains(Segment_3(Point_2(0.5, -10), Point_2(10, 0.5))));

//   CHECK(!Tiling_region());
}

TEST (Tiling_region3_2)
{
  HTiling_region region = Wedge::RS(Point_2(1, 1), Point_2(1, 0), Point_2(0, 0), 0);
  region = region->get_complement();
  CHECK(region->contains(Point_2(0, 0)));
  CHECK(region->contains(Point_2(1, 0)));
  CHECK(region->contains(Point_2(1.5, 0)));
  CHECK(region->contains(Point_2(1, 1000)));
  CHECK(!region->contains(Point_2(0, 1)));
  CHECK(region->contains(Point_2(-1000, 0)));
  CHECK(!region->contains(Point_2(-1000, .0001)));
  CHECK(region->contains(Point_2(1000, 0)));
  CHECK(region->contains(Point_2(1000, .0001)));

  CHECK(region->contains(Point_2(1, -.0001)));
  CHECK(region->contains(Point_2(1, -1000)));
  CHECK(region->contains(Point_2(1000, -.0001)));
  CHECK(region->contains(Point_2(1000, -1000)));
  CHECK(region->contains(Point_2(-1000, -.0001)));
  CHECK(region->contains(Point_2(-1000, -1000)));

  CHECK(!region->contains(Segment_3(Point_2(0, 1), Point_2(0, 2))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(0, 2))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(1, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, 0), Point_2(2, 3))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1), Point_2(2, 3))));
  CHECK(region->contains(Segment_3(Point_2(0.5, -10), Point_2(10, 0.5))));

//   CHECK(!Tiling_region());
}

TEST (Tiling_region4)
{
  HTiling_region region0 = Wedge::LS(Point_2(3, -2), Point_2(0, 0), Point_2(-3, -2), 0);
  HTiling_region region1 = Wedge::LS(Point_2(2, 0), Point_2(0, 0), Point_2(-1, -2), 0);
  HTiling_region region = region0 & region1;
  CHECK(!region->contains(Point_2(0, 0)));
  CHECK(region->contains(Point_2(0, -1)));
  CHECK(region->contains(Point_2(1, -1)));
  CHECK(!region->contains(Point_2(3, -2)));
  CHECK(!region->contains(Point_2(-2.5, -2)));
  CHECK(!region->contains(Point_2(-2, -2)));
  CHECK(!region->contains(Point_2(-3, -2)));
  CHECK(!region->contains(Point_2(2, -1)));
  CHECK(!region->contains(Point_2(-1, 0)));
  CHECK(!region->contains(Point_2(1, 1)));
}

TEST (Tiling_region4_2)
{
  HTiling_region region0 = Wedge::LS(Point_2(3, -2), Point_2(0, 0), Point_2(-3, -2), 0);
  HTiling_region region1 = Wedge::LS(Point_2(2, 0), Point_2(0, 0), Point_2(-1, -2), 0);
  HTiling_region region = region0 | region1;
  CHECK(!region->contains(Point_2(0, 0)));
  CHECK(region->contains(Point_2(0, -1)));
  CHECK(region->contains(Point_2(1, -1)));
  CHECK(region->contains(Point_2(3, -2)));
  CHECK(region->contains(Point_2(-2.5, -2)));
  CHECK(region->contains(Point_2(-2, -2)));
  CHECK(!region->contains(Point_2(-3, -2)));
  CHECK(region->contains(Point_2(2, -1)));
  CHECK(!region->contains(Point_2(-1, 0)));
  CHECK(!region->contains(Point_2(1, 1)));
}

TEST (Tiling_region4_3)
{
  HTiling_region region0 = Wedge::LS(Point_2(3, -2), Point_2(0, 0), Point_2(-3, -2), 0);
  HTiling_region region1 = Wedge::LS(Point_2(2, 0), Point_2(0, 0), Point_2(-1, -2), 0);
  HTiling_region region = (region0 | region1);
  region = region->get_complement();
  CHECK(region->contains(Point_2(0, 0)));
  CHECK(!region->contains(Point_2(0, -1)));
  CHECK(!region->contains(Point_2(1, -1)));
  CHECK(!region->contains(Point_2(3, -2)));
  CHECK(!region->contains(Point_2(-2.5, -2)));
  CHECK(!region->contains(Point_2(-2, -2)));
  CHECK(region->contains(Point_2(-3, -2)));
  CHECK(!region->contains(Point_2(2, -1)));
  CHECK(region->contains(Point_2(-1, 0)));
  CHECK(region->contains(Point_2(1, 1)));
}

TEST (Tiling_region5)
{
//   Tiling_region region0 = Wedge::LS(Point_2(3, -2), Point_2(0, 0), Point_2(-3, -2));
//   Tiling_region region1 = Wedge::LS(Point_2(2, 0), Point_2(0, 0), Point_2(-1, -2));
  HTiling_region region = Tiling_region::overlapping_vertex(Point_2(3, -2), Point_2(0, 0), Point_2(-3, -2),
						    Point_2(2, 0), Point_2(0, 0), Point_2(-1, -2));
  CHECK(region->contains(Point_2(-2.5, -2)));
  CHECK(region->contains(Point_2(2, -1)));
  CHECK(!region->contains(Point_2(0, -1)));
  CHECK(!region->contains(Point_2(1, -1)));
  CHECK(region->contains(Point_2(2, -1)));
  CHECK(!region->contains(Point_2(-1, 0)));
  CHECK(!region->contains(Point_2(1, 1)));

  // On boundary
  CHECK(region->contains(Point_2(0, 0)));
  CHECK(region->contains(Point_2(3, -2)));
  CHECK(region->contains(Point_2(-3, -2)));
  CHECK(region->contains(Point_2(-2, -2)));
}

TEST (Tiling_region6)
{
  HTiling_region region = Wedge::LS(Point_2(0, 8), Point_2(8, 8), Point_2(8, 0), 0);
  CHECK(!region->contains(Segment_3(Point_2(8, 8), Point_2(9, 5))));
  region = Wedge::RS(Point_2(0, 8), Point_2(8, 8), Point_2(8, 0), 0);
  region = region->get_complement();
  CHECK(region->contains(Segment_3(Point_2(8, 8), Point_2(9, 5))));
}

TEST (Tiling_region7)
{
  // regions are going in opposite directions
  HTiling_region region = Tiling_region::overlapping_vertex(Point_2(-1, 0), Point_2(0, 0), Point_2(1, 0),
						    Point_2(1, 0), Point_2(0, 0), Point_2(-1, 0));
//   CHECK(!region->contains(Point_2(0, 0)));
//   CHECK(!region->contains(Point_2(-1, 0)));
//   CHECK(!region->contains(Point_2(1, 0)));
//   CHECK(!region->contains(Point_2(0, -1)));
//   CHECK(!region->contains(Point_2(0, 1)));
}

TEST (Tiling_region8)
{
  HTiling_region region = Tiling_region::overlapping_vertex(Point_2(-1, 0), Point_2(0, 0), Point_2(1, 0),
						    Point_2(-1, 0), Point_2(0, 0), Point_2(1, 0));
  CHECK(region->contains(Point_2(0, 0)));
  CHECK(region->contains(Point_2(-1, 0)));
  CHECK(region->contains(Point_2(1, 0)));
  CHECK(!region->contains(Point_2(0, -1)));
  CHECK(!region->contains(Point_2(0, 1)));
}

TEST (Tiling_region9)
{
  HTiling_region region = Tiling_region::overlapping_vertex(Point_2(0, 1, 0), Point_2(0, 0, 0), Point_2(1, 0, 0),
							   Point_2(0, -1,1), Point_2(0, 0, 1), Point_2(-1,0, 1));
  CHECK(region->contains(Point_2(0, 0, 0)));
  CHECK(!region->contains(Point_2(0, 0, 1)));
  CHECK(region->contains(Segment_3(Point_2(-1, -1, 0), Point_2(0, 0, 0))));
  CHECK(!region->contains(Segment_3(Point_2(-1, -1, 1), Point_2(0, 0, 1))));
}

