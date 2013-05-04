#include <ContourTiler/test_common.h>
#include <ContourTiler/reader_gnuplot.h>
#include <ContourTiler/augment.h>
#include <ContourTiler/print_utils.h>
#include <ContourTiler/offset_polygon.h>
#include <ContourTiler/polygon_difference.h>
#include <ContourTiler/minkowski.h>
#include <ContourTiler/polygon_utils.h>

TEST (offset2)
{
  vector<Contour_handle> temp;
  read_contours_gnuplot2(data_dir+"/origp.dat", back_inserter(temp), 1);
  Polygon_2 p = temp[0]->polygon();

  temp.clear();
  read_contours_gnuplot2(data_dir+"/origq.dat", back_inserter(temp), 1);
  Polygon_2 q = temp[0]->polygon();

  list<Polygon_2> newp, newq;
  polygon_difference(p, q, back_inserter(newp), back_inserter(newq));
//   cout << pp(p) << " " << pp(q);
}

TEST (offset1)
{
  vector<Contour_handle> temp;
  read_contours_gnuplot2(data_dir+"/off1.dat", back_inserter(temp), 1);
  Polygon_2 p = temp[0]->polygon();

  Number_type offset = -0.01;
  p = remove_collinear(p, 0);
  cout << pp(offset_polygon(p, offset));
}

TEST (augment2)
{
  Polygon_2 P, Q, Pe, Qe;
  P.push_back(Point_2(0,0,0));
  P.push_back(Point_2(8,0,0));
  P.push_back(Point_2(8,4,0));
  P.push_back(Point_2(0,4,0));
  
  Q.push_back(Point_2(2,1,1));
  Q.push_back(Point_2(7,1,1));
  Q.push_back(Point_2(7,5,1));
  Q.push_back(Point_2(6,5,1));
  Q.push_back(Point_2(6,2,1));
  Q.push_back(Point_2(3,2,1));
  Q.push_back(Point_2(3,5,1));
  Q.push_back(Point_2(4,5,1));
  Q.push_back(Point_2(4,3,1));
  Q.push_back(Point_2(5,3,1));
  Q.push_back(Point_2(5,6,1));
  Q.push_back(Point_2(2,6,1));
  
  Pe.push_back(Point_3(0,0,0));
  Pe.push_back(Point_3(8,0,0));
  Pe.push_back(Point_3(8,4,0));
  Pe.push_back(Point_3(7,4,0));
  Pe.push_back(Point_3(6,4,0));
  Pe.push_back(Point_3(5,4,0));
  Pe.push_back(Point_3(4,4,0));
  Pe.push_back(Point_3(3,4,0));
  Pe.push_back(Point_3(2,4,0));
  Pe.push_back(Point_3(0,4,0));

  Qe.push_back(Point_3(2,1,1));
  Qe.push_back(Point_3(7,1,1));
  Qe.push_back(Point_3(7,4,1));
  Qe.push_back(Point_3(7,5,1));
  Qe.push_back(Point_3(6,5,1));
  Qe.push_back(Point_3(6,4,1));
  Qe.push_back(Point_3(6,2,1));
  Qe.push_back(Point_3(3,2,1));
  Qe.push_back(Point_3(3,4,1));
  Qe.push_back(Point_3(3,5,1));
  Qe.push_back(Point_3(4,5,1));
  Qe.push_back(Point_3(4,4,1));
  Qe.push_back(Point_3(4,3,1));
  Qe.push_back(Point_3(5,3,1));
  Qe.push_back(Point_3(5,4,1));
  Qe.push_back(Point_3(5,6,1));
  Qe.push_back(Point_3(2,6,1));
  Qe.push_back(Point_3(2,4,1));

  boost::tie(P, Q) = augment1(P, Q);
  CHECK_EQUAL(Pe, P);
  CHECK_EQUAL(Qe, Q);
}

TEST (augment3)
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
  
  Pe.push_back(Point_3(0,0,0));
  Pe.push_back(Point_3(4,0,0));
  Pe.push_back(Point_3(4,1,0));
  Pe.push_back(Point_3(4,2,0));
  Pe.push_back(Point_3(4,3,0));
  Pe.push_back(Point_3(4,4,0));
  Pe.push_back(Point_3(3,4,0));
  Pe.push_back(Point_3(1,4,0));
  Pe.push_back(Point_3(0,4,0));

  Qe.push_back(Point_3(1,1,1));
  Qe.push_back(Point_3(4,1,1));
  Qe.push_back(Point_3(5,1,1));
  Qe.push_back(Point_3(4,2,1));
  Qe.push_back(Point_3(4,3,1));
  Qe.push_back(Point_3(3,4,1));
  Qe.push_back(Point_3(1,4,1));

  boost::tie(P, Q) = augment1(P, Q);
  CHECK_EQUAL(Pe, P);
  CHECK_EQUAL(Qe, Q);
}
