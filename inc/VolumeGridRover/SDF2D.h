/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <cmath>
#include <set>
#include <list>
#include <iostream>
#include <cfloat>
#include <fstream>
#include <string>
#include <exception>

#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>

#include <CGAL/Timer.h>
#include <CGAL/basic.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/Partition_is_valid_traits_2.h>
#include <CGAL/polygon_function_objects.h>
#include <CGAL/partition_2.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/random_polygon_2.h>
#include <CGAL/Search_traits.h>
#include <CGAL/Kernel_traits.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Cartesian.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Bbox_2.h>
#include <CGAL/Segment_2.h>

namespace SDF2D {

  namespace Math {
    static bool betweenAngle(double theta, double low, double high) {
      // make values positive
      while(theta < 0)
	theta += (2.0*M_PI);
      assert(low >= 0);
      assert(high >= 0);
    
      return (low <= theta && theta <= high);
    }
    // is x strictly between x1 and x2 ?
    static bool betweenStrict(double x, double x1, double x2) {
      // IMPORTANT-- we need to make one side take precendence in a case where both are equal
      // this is because sometimes we shoot rays at endpoints and we need to act as though
      // only 1 segment was hit
      if((x1 < x) && (x <= x2))
	return true;
      else
	return false;
    }
    static int min(int lhs, int rhs) {
      if(lhs<rhs)
	return lhs;
      else
	return rhs;
    }
    static int max(int lhs, int rhs) {
      if(lhs>rhs)
	return lhs;
      else
	return rhs;
    }
    static int roundAwayFromZero(float input) {
      if(input > 0) {
	return (int)ceil(input);
      }
      else {
	return (int)floor(input);
      }
    }
  };

  class Point2D {
  public:
    // ctors
    Point2D() : m_X(0), m_Y(0) {}
    Point2D(float x, float y) : m_X(x), m_Y(y) {}

    // helpers
    float distance(const Point2D& rhs) const {
      float dx = (rhs.getX() - getX());
      float dy = (rhs.getY() - getY());
      return sqrt(dx*dx + dy*dy);
    }
#if 0
    void draw() {
      glBegin(GL_POINTS);
      glVertex2f();
      glEnd();
    }
    void glVertex2f() const {
      ::glVertex2f(getX(),getY());
    }
#endif
    float magnitude() {
      return distance(Point2D(0,0));
    }
    float crossMagnitude(const Point2D& other) {
      return fabs((getX()*other.getY()) - (getY()*other.getX()));
    }
    float dot(const Point2D& other) {
      return (getX()*other.getX() + getY()*other.getY());
    }
        
    // accessors
    float getX() const { return m_X; }
    float getY() const { return m_Y; }
    float x() const { return m_X; }
    float y() const { return m_Y; }
        
    // operators
    void move(float x, float y) { m_X = x; m_Y = y; }
    void move(const Point2D& delta) { m_X += delta.x(); m_Y += delta.y(); }
        
    const Point2D operator+(const Point2D &other) const {
      Point2D result(getX() + other.getX(),
		     getY() + other.getY());
      return result;
    }
    const Point2D operator-(const Point2D &other) const {
      Point2D result(getX() - other.getX(),
		     getY() - other.getY());
      return result;
    }
    const Point2D operator*(const double &other) const {
      Point2D result(getX() * other,
		     getY() * other);
      return result;
    }
    const Point2D integer() const {
      return Point2D(Math::roundAwayFromZero(x()),
		     Math::roundAwayFromZero(y()));
    }
  private:
    float m_X;
    float m_Y;
  };

  typedef boost::shared_ptr<Point2D> Point2DPtr;

  class Rect2D {
  public:
    Rect2D(Point2D min, Point2D max) : _min(min), _max(max) {}
    // from normalized coordinates back to planar coordinates (assuming this was the bounding of
    // the planar coordinates)
    Point2D denormalize(Point2D input) const {
      double dx = _max.getX() - _min.getX();
      double dy = _max.getY() - _min.getY();
      double max = (dx>dy)?dx:dy;
      double x = _min.getX() + (max * input.getX());
      double y = _min.getY() + (max * input.getY());
      return Point2D(x,y);
    }
    Point2D getMin() const { return _min; }
    Point2D getMax() const { return _max; }
  private:
    Point2D _min, _max;
  };

  class Edge2D {
  public:
    Edge2D(Point2DPtr first, Point2DPtr second) {
      m_First = first;
      m_Second = second;
    }

    // accessors
    Point2DPtr getFirst() const { return m_First; }
    Point2DPtr getSecond() const { return m_Second; }
        
    // helper functions
    bool contains(const Point2DPtr& point) {
      return ((point == m_First) || (point == m_Second));
    }
        
    bool operator==(const Edge2D& rhs) {
      return (contains(rhs.getFirst()) && contains(rhs.getSecond()));
    }
        
#if 0
    void draw() {
      glBegin(GL_LINES);
      m_First->glVertex2f();
      m_Second->glVertex2f();
      glEnd();
    }
#endif
        
    // does this segment intersect the X-ALIGNED line or the Y-ALIGNED line intersecting this
    // point ?
    bool intersectsXAxisAlignedLine(double x, double yBelow, double yAbove) {
      // interpolated test
      Point2DPtr minXPt = getMinXPt();
      Point2DPtr maxXPt = getMaxXPt();
      double deltaX = (maxXPt->getX() - minXPt->getX());
      double deltaY = (maxXPt->getY() - minXPt->getY());
      double alpha = (x - minXPt->getX())/deltaX;

      // axis bound test
      bool xIntersects = Math::betweenStrict(x, minXPt->getX(), maxXPt->getX());
                
      double segmentYIntersection = minXPt->getY() + (alpha * deltaY);
                
      bool yIntersects = Math::betweenStrict(segmentYIntersection, yBelow, yAbove);
                
      return (xIntersects && yIntersects);
    }
        
    bool intersectsYAxisAlignedLine(double y, double xLeft, double xRight) {
      // interpolated test
      Point2DPtr minYPt = getMinYPt();
      Point2DPtr maxYPt = getMaxYPt();
      double deltaX = (maxYPt->getX() - minYPt->getX());
      double deltaY = (maxYPt->getY() - minYPt->getY());
      double alpha = (y - minYPt->getY())/deltaY;

      // axis bound test
      bool yIntersects = Math::betweenStrict(y, minYPt->getY(), maxYPt->getY());
                
      double segmentXIntersection = minYPt->getX() + (alpha * deltaX);
                
      bool xIntersects = Math::betweenStrict(segmentXIntersection, xLeft, xRight);
                
      return (xIntersects && yIntersects);
    }

    Point2DPtr getMinXPt() {
      if(firstIsXMin())
	return m_First;
      else
	return m_Second;
    }
    Point2DPtr getMaxXPt() {
      if(!firstIsXMin())
	return m_First;
      else
	return m_Second;
    }
    Point2DPtr getMinYPt() {
      if(firstIsYMin())
	return m_First;
      else
	return m_Second;
    }
    Point2DPtr getMaxYPt() {
      if(!firstIsYMin())
	return m_First;
      else
	return m_Second;
    }

  private:
    bool firstIsXMin() { return (m_First->getX() < m_Second->getX()); }
    bool firstIsYMin() { return (m_First->getY() < m_Second->getY()); }
        
    Point2DPtr m_First, m_Second;
  };

  typedef boost::shared_ptr<Edge2D> Edge2DPtr;

  typedef CGAL::Simple_cartesian<double> K;
  typedef K::Point_2 Point_2;
  typedef CGAL::Polygon_2<K> Polygon_2;
  typedef CGAL::Segment_2<K> Segment_2;

  struct Point2DWrapper {
    double vec[3];
    std::set<Edge2DPtr> m_EdgeSet;
        
    Point2DWrapper() { vec[0]= vec[1] = vec[2] = 0; }
    Point2DWrapper (double x, double y, double z) { vec[0]=x; vec[1]=y; vec[2]=z;  }
    Point2DWrapper (Point2DPtr ptr, std::set<Edge2DPtr> edgeSet) {
      vec[0]=ptr->getX();
      vec[1]=ptr->getY();
      vec[2]=0;
      m_EdgeSet = edgeSet;
    }

    double x() const { return vec[ 0 ]; }
    double y() const { return vec[ 1 ]; }
    double z() const { return vec[ 2 ]; }

    double& x() { return vec[ 0 ]; }
    double& y() { return vec[ 1 ]; }
    double& z() { return vec[ 2 ]; }

    bool operator==(const Point2DWrapper& p) const
    {
      return (x() == p.x()) && (y() == p.y()) && (z() == p.z())  ;
    }

    bool  operator!=(const Point2DWrapper& p) const { return ! (*this == p); }
  }; //end of class

}

//this needs to be outside the SDF2D namespace
namespace CGAL {
  
  template <>
    struct Kernel_traits<SDF2D::Point2DWrapper> {
      struct Kernel {
	typedef double FT;
	typedef double RT;
      };
    };
}

namespace SDF2D {

  struct Construct_coord_iterator {
	typedef const double* result_type;
    const double* operator()(const Point2DWrapper& p) const
    { return static_cast<const double*>(p.vec); }

    const double* operator()(const Point2DWrapper& p, int)  const
    { return static_cast<const double*>(p.vec+3); }
  };

  struct Distance2DWrapper {
    typedef Point2DWrapper Query_item;

    double transformed_distance(const Point2DWrapper& p1, const Point2DWrapper& p2) const {
      double distx= p1.x()-p2.x();
      double disty= p1.y()-p2.y();
      double distz= p1.z()-p2.z();
      return distx*distx+disty*disty+distz*distz;
    }

    template <class TreeTraits>
    double min_distance_to_rectangle(const Point2DWrapper& p,
				     const CGAL::Kd_tree_rectangle<TreeTraits>& b) const {
      double distance(0.0), h = p.x();
      if (h < b.min_coord(0)) distance += (b.min_coord(0)-h)*(b.min_coord(0)-h);
      if (h > b.max_coord(0)) distance += (h-b.max_coord(0))*(h-b.max_coord(0));
      h=p.y();
      if (h < b.min_coord(1)) distance += (b.min_coord(1)-h)*(b.min_coord(1)-h);
      if (h > b.max_coord(1)) distance += (h-b.max_coord(1))*(h-b.min_coord(1));
      h=p.z();
      if (h < b.min_coord(2)) distance += (b.min_coord(2)-h)*(b.min_coord(2)-h);
      if (h > b.max_coord(2)) distance += (h-b.max_coord(2))*(h-b.max_coord(2));
      return distance;
    }

    template <class TreeTraits>
    double max_distance_to_rectangle(const Point2DWrapper& p,
				     const CGAL::Kd_tree_rectangle<TreeTraits>& b) const {
      double h = p.x();

      double d0 = (h >= (b.min_coord(0)+b.max_coord(0))/2.0) ?
	(h-b.min_coord(0))*(h-b.min_coord(0)) : (b.max_coord(0)-h)*(b.max_coord(0)-h);

      h=p.y();
      double d1 = (h >= (b.min_coord(1)+b.max_coord(1))/2.0) ?
	(h-b.min_coord(1))*(h-b.min_coord(1)) : (b.max_coord(1)-h)*(b.max_coord(1)-h);
      h=p.z();
      double d2 = (h >= (b.min_coord(2)+b.max_coord(2))/2.0) ?
	(h-b.min_coord(2))*(h-b.min_coord(2)) : (b.max_coord(2)-h)*(b.max_coord(2)-h);
      return d0 + d1 + d2;
    }

    double new_distance(double& dist, double old_off, double new_off,
			int /* cutting_dimension */)  const {
      return dist + new_off*new_off - old_off*old_off;
    }

    double transformed_distance(double d) const { return d*d; }

    double inverse_of_transformed_distance(double d) { return std::sqrt(d); }

  };

  typedef CGAL::Random_points_in_cube_3<Point2DWrapper> Random_points_iterator;
  typedef CGAL::Counting_iterator<Random_points_iterator> N_Random_points_iterator;
  typedef CGAL::Search_traits<double, Point2DWrapper, const double*, Construct_coord_iterator> Traits;
  typedef CGAL::Orthogonal_k_neighbor_search<Traits, Distance2DWrapper> K_neighbor_search;
  typedef K_neighbor_search::Tree Tree;

  class Polygon {
  public:
    std::vector<Point2DPtr> getVertexArray() const { return m_SkeletonPoints; }
    std::vector<Edge2D> getEdgeArray() const { return m_SkeletonEdges; }
        
    Polygon() { clear(); }
    ~Polygon() {}
    void addPoint(Point2DPtr newPt) { m_SkeletonPoints.push_back(newPt); }
    void addPoint(Point2D newPt) {
      Point2DPtr newPtr = Point2DPtr(new Point2D(newPt));
      m_SkeletonPoints.push_back(newPtr);
    }
    void addEdge(Edge2D edge) { m_SkeletonEdges.push_back(edge); }
    void addEdge(int first, int second) {
      Edge2D newEdge(m_SkeletonPoints[first], m_SkeletonPoints[second]);
      m_SkeletonEdges.push_back(newEdge);
    }
    bool getCGALPolygonAndPoints(Polygon_2& polygon, std::vector<Point2DWrapper>& treeVertices);
    Rect2D getBoundingRect();
        
    int pointPtrToVertexIndex(Point2DPtr ptr) const;
    void destroyPoint(const Point2DPtr& ptr);
    bool edgeExistsAlready(const Edge2D& edge);
    void clear();

    void load(const std::string& filename);
    void save(const std::string& filename);
  private:
    std::vector<Point2DPtr> m_SkeletonPoints;
    std::vector<Edge2D> m_SkeletonEdges;
  };

  typedef boost::tuple<unsigned int, unsigned int> Dimension;
  typedef CGAL::Bbox_2 BoundingBox;
  typedef boost::multi_array<double, 2> Image;
  typedef boost::shared_ptr<Image> ImagePtr;
  typedef Image::index ImageIndex;

  class NonSimplePolygonException : public std::exception
  {
  public:
    NonSimplePolygonException() throw () {}
    virtual const char * what () const throw () 
    { return "Cannot calculate SDF of non-simple polygon."; }
  };

  enum SignMethod { ANGLE_SUM, COUNT_EDGE_INTERSECTIONS };
  enum DistanceMethod { BRUTE_FORCE, K_NEIGHBOR_SEARCH };

  Image signedDistanceFunction(const std::vector<Polygon_2>&,
			       const Dimension&,
			       const BoundingBox&,
			       SignMethod sign_method = COUNT_EDGE_INTERSECTIONS,
			       DistanceMethod dist_method = K_NEIGHBOR_SEARCH);
}
