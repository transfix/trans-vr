#ifndef DATASTRUCT_H
#define DATASTRUCT_H

#include <CGAL/Cartesian.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Point_3.h>
#include <CGAL/Quotient.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <vector>

typedef CGAL::Lazy_exact_nt<CGAL::Quotient<CGAL::MP_Float>> NT;
typedef CGAL::Cartesian<NT> K;
typedef K::Point_3 Point;
typedef K::Vector_3 Vector;
typedef std::pair<std::vector<Point>, Vector> Cylinder;

/*
// define your Point, Vector and other primitive classes here.
// below were my definitions using CGAL.

struct K : CGAL::Exact_predicates_inexact_constructions_kernel {};

typedef VC_vertex< K >                                          Vertex;
typedef VC_cell< K >                                            Cell;
typedef CGAL::Triangulation_data_structure_3<Vertex, Cell> 	TDS;
typedef CGAL::Delaunay_triangulation_3<K,TDS>  	                Triangulation;

typedef Triangulation::Point                                    Point;
typedef Point::R                          			Rep;
typedef CGAL::Vector_3<Rep>			     		Vector;
typedef CGAL::Ray_3<Rep>                                        Ray;
typedef CGAL::Segment_3<Rep>                                    Segment;
typedef CGAL::Triangle_3<Rep>                                   Triangle_3;
typedef CGAL::Tetrahedron_3<Rep>                     		Tetrahedron;
typedef CGAL::Aff_transformation_3<Rep>                     	Aff_tr_3;
*/

#endif // DATASTRUCT_H
