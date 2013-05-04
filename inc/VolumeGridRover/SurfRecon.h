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

#ifndef __SURFRECON_H__
#define __SURFRECON_H__

#include <vector>
#include <map>
#include <list>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <CGAL/Cartesian.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Point_3.h>
#include <CGAL/Vector_3.h>

// using GSL library for contour interpolation
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

namespace SurfRecon
{
  typedef CGAL::Cartesian<double> Kernel;
  typedef CGAL::Point_2< Kernel > Point_2;
  typedef boost::tuple<std::vector<double>, /* knots */
		       std::vector<Point_2> > B_Spline; /* control points */

  static inline std::vector<double>& getKnots(B_Spline &b) { return boost::get<0>(b); }
  static inline std::vector<double> getKnots(const B_Spline &b) { return boost::get<0>(b); }
  static inline std::vector<Point_2>& getPoints(B_Spline &b) { return boost::get<1>(b); }
  static inline std::vector<Point_2> getPoints(const B_Spline &b) { return boost::get<1>(b); }

  typedef CGAL::Point_3< Kernel > Point;
  typedef CGAL::Vector_3< Kernel > Vector;
  typedef boost::tuple<float, float, float> Color;
  typedef boost::shared_ptr<Point> PointPtr;
  typedef boost::weak_ptr<Point> WeakPointPtr;
  enum Orientation { XY, XZ, ZY };
  static const char *OrientationStrings[] = { "XY", "XZ", "ZY" };
  typedef std::list<PointPtr> PointPtrList;
  typedef std::vector<PointPtr> PointPtrVector;
  //curve is a list of points and a slice and orientation
  typedef boost::tuple<unsigned int, /* slice */
                       Orientation,  /* orientation */
                       PointPtrList, /* list of points */
                       std::string,  /* object name */
                       std::vector<B_Spline> /* a spline fit to this curve's points  */> Curve;
  static inline unsigned int& getCurveSlice(Curve& c) { return boost::get<0>(c); }
  static inline Orientation& getCurveOrientation(Curve& c) { return boost::get<1>(c); }
  static inline std::string getCurveOrientationString(Curve& c) { return std::string(OrientationStrings[getCurveOrientation(c)]); }
  static inline PointPtrList& getCurvePoints(Curve& c) { return boost::get<2>(c); }
  static inline std::string& getCurveName(Curve& c) { return boost::get<3>(c); }
  static inline std::vector<B_Spline>& getCurveSpline(Curve& c) { return boost::get<4>(c); }
  typedef boost::shared_ptr<Curve> CurvePtr;
  typedef std::vector<CurvePtr> CurvePtrVector;
  typedef boost::shared_ptr<CurvePtrVector> CurvePtrVectorPtr;

  std::vector<B_Spline> fit_spline(const SurfRecon::CurvePtr curve);
  std::vector<B_Spline> fit_spline(const SurfRecon::PointPtrList& curve);

  enum InterpolationType 
    {
      Linear,
      Polynomial,
      CSpline,
      CSplinePeriodic,
      Akima,
      AkimaPeriodic
    };

  static const char *InterpolationTypeStrings[] = 
    {
      "Linear",
      "Polynomial",
      "CSpline",
      "CSplinePeriodic",
      "Akima",
      "AkimaPeriodic"
    };

  static inline InterpolationType getInterpolationTypeFromString(const std::string& inttype)
    {
      for(int i=0; i<6; i++)
	if(inttype == InterpolationTypeStrings[i])
	  return InterpolationType(i);
      return Linear; //default to linear if string is invalid
    }

  class Contour
  {
  public:
    Contour(const Color& c = Color(1.0,1.0,1.0), 
	    const std::string& n = "", InterpolationType it = Linear,
	    unsigned int numSamp = 5,
	    const CurvePtrVectorPtr& l = CurvePtrVectorPtr(new std::vector<CurvePtr>)) 
      : _curves(l), _color(c), _name(n), _interpolationType(it), _numSamples(numSamp), _selected(false) {}
    Contour(const Contour& c) 
      : _curves(c._curves), _color(c._color), _name(c._name), 
      _interpolationType(c._interpolationType), _numSamples(c._numSamples), _selected(c._selected) {}
    Contour(const CurvePtrVectorPtr& l) : _curves(l) {}

    std::vector<CurvePtr>& curves() { return *_curves; }
    const std::vector<CurvePtr>& curves() const { return *_curves; }

    const CurvePtr& operator[](int i) const { return (*_curves)[i]; }
    CurvePtr& operator[](int i) { return (*_curves)[i]; }

    void add(const CurvePtr& l) { _curves->push_back(l); _currentCurve = l; }
    void add(const PointPtr& p, unsigned int slice = 0, Orientation o = XY) 
    { 
      if(_currentCurve == NULL || _currentCurve->get<0>() != slice || _currentCurve->get<1>() != o)
	{
	  _currentCurve.reset(new Curve(slice,o,SurfRecon::PointPtrList(),std::string()));
	  _curves->push_back(_currentCurve);
	}
      boost::get<2>(*_currentCurve).push_back(p);
    }
    void prepend(const PointPtr& p, unsigned int slice = 0, Orientation o = XY)
    {
      if(_currentCurve == NULL || _currentCurve->get<0>() != slice || _currentCurve->get<1>() != o)
	{
	  _currentCurve.reset(new Curve(slice,o,SurfRecon::PointPtrList(),std::string()));
	  _curves->push_back(_currentCurve);
	}
      boost::get<2>(*_currentCurve).push_front(p);
    }
    void clear() { _curves->clear(); }

    void currentCurve(int c) const { _currentCurve = (*_curves)[c]; }
    void currentCurve(const CurvePtr& c) const
    {
      //make sure that the curve pointer is in this contour's list of curves
      for(CurvePtrVector::iterator i = _curves->begin();
	  i != _curves->end();
	  i++)
	if(c == *i) _currentCurve = c;
    }
    const CurvePtr& currentCurve() const { return _currentCurve; }
    CurvePtr& currentCurve() { return _currentCurve; }
    
    const PointPtrList& currentCurvePoints() const { return _currentCurve->get<2>(); }
    PointPtrList& currentCurvePoints() { return _currentCurve->get<2>(); }
    
    int numCurves() const { return _curves->size(); }
    
    void name(const std::string& n) { _name = n; }
    std::string name() const { return _name; }

    void color(const Color& c) { _color = c; }
    const Color& color() const { return _color; }

    void interpolationType(InterpolationType it) { _interpolationType = it; }
    InterpolationType interpolationType() const { return _interpolationType; }

    void numberOfSamples(unsigned int numSamp) { _numSamples = numSamp; }
    unsigned int numberOfSamples() const { return _numSamples; }

    void selected(bool s) { _selected = s; }
    bool selected() const { return _selected; }
    
    void computeSplines() /* compute splines for this contour */
    {
      for(CurvePtrVector::iterator i = _curves->begin();
	  i != _curves->end();
	  i++)
	{
	  getCurveSpline(**i) = fit_spline(*i);
	}
    }

    void clearSplines() /* clear the splines if we dont need them */
    {
      for(CurvePtrVector::iterator i = _curves->begin();
	  i != _curves->end();
	  i++)
	{
	  getCurveSpline(**i).clear();
	}
    }

  private:
    CurvePtrVectorPtr _curves; //all the curves that link the points together

    Color _color;
    std::string _name;

    mutable CurvePtr _currentCurve;

    InterpolationType _interpolationType;

    unsigned int _numSamples;

    bool _selected;
  };
  
  typedef boost::shared_ptr<Contour> ContourPtr;
  //3D array of contours
  //var,time,contour_list
  typedef std::map<std::string,SurfRecon::ContourPtr> ContourPtrMap;
  typedef std::vector<std::vector<ContourPtrMap> > ContourPtrArray;
  typedef boost::shared_ptr<ContourPtrArray> ContourPtrArrayPtr;
};

#endif
