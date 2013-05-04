/*
  Copyright 2007-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of libCVC.

  libCVC is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  libCVC is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: BoundingBox.h 5355 2012-04-06 22:16:56Z transfix $ */

#ifndef __CVC_BOUNDINGBOX_H__
#define __CVC_BOUNDINGBOX_H__

#include <CVC/Types.h>
#include <CVC/Exception.h>
#include <CVC/Dimension.h>
#include <CVC/Namespace.h>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

// If your compiler complains the "The class "CVC::GenericBoundingBox<T>" has no member "minx"."
// Add your architecture Q_OS_XXXX flag (see qglobal.h) in this list.
//#if defined (Q_OS_IRIX) || defined (Q_OS_AIX) || defined (Q_OS_HPUX)
# define UNION_NOT_SUPPORTED
//#endif

namespace CVC_NAMESPACE
{
  CVC_DEF_EXCEPTION(InvalidBoundingBox);
  CVC_DEF_EXCEPTION(InvalidBoundingBoxString);

  /*
    GenericBoundingBox min/max values are inclusive.
  */
  template <typename T>
  class GenericBoundingBox
  {
  public:
    /* The internal data representation is public. */

#if defined (DOXYGEN) || defined (UNION_NOT_SUPPORTED)
    T minx, miny, minz;
#else
    union
    {
      struct { T minx, miny, minz; };
      T min_[3];
    };
#endif
    
#if defined (DOXYGEN) || defined (UNION_NOT_SUPPORTED)
    T maxx, maxy, maxz;
#else
    union
    {
      struct { T maxx, maxy, maxz; };
      T max_[3];
    };
#endif    
    
    /* default constructor */
    GenericBoundingBox()
      {
	setMin(T(0),T(0),T(0));
	setMax(T(0),T(0),T(0));
      }
      
      /* standard constructor */
    GenericBoundingBox(T minx_, T miny_, T minz_, T maxx_, T maxy_, T maxz_)
      { 
	setMin(minx_,miny_,minz_);
	setMax(maxx_,maxy_,maxz_);
	checkBounds();
      }

    GenericBoundingBox(const Dimension& dimension)
      { 
	setMin(T(0),T(0),T(0));
	setMax(T(dimension.xdim-1),T(dimension.ydim-1),T(dimension.zdim-1));
	checkBounds();
      }

    //initialize from string
    GenericBoundingBox(const std::string& s)
      {
        str(s);
      }
	
    /*
      Universal explicit converter from any class to GenericBoundingBox (as long as that class implements
      operator[]).
    */
    template <class C> explicit GenericBoundingBox(const C& m)
      { 
	setMin(T(m[0]),T(m[1]),T(m[2]));
	setMax(T(m[3]),T(m[4]),T(m[5]));
	checkBounds(); 
      }

    GenericBoundingBox<T>& operator=(const GenericBoundingBox<T>& b)
    {
      setMin(b.minx,b.miny,b.minz);
      setMax(b.maxx,b.maxy,b.maxz);
      return *this;
    }

    void setMin(T minx_, T miny_, T minz_)
    { 
      minx=minx_; miny=miny_; minz=minz_;
    }

    void setMax(T maxx_, T maxy_, T maxz_)
    {
      maxx=maxx_; maxy=maxy_; maxz=maxz_;
    }

    
#ifdef UNION_NOT_SUPPORTED
# define REALMAX (&maxx)
# define REALMIN (&minx)
#else
# define REALMAX max_
# define REALMIN min_
#endif

    /* Bracket operator with a constant return value. */
    T operator[](int i) const { return i>=3 ? REALMAX[i-3] : REALMIN[i]; }

    /* Bracket operator returning an l-value. */
    T& operator[](int i) { return i>=3 ? REALMAX[i-3] : REALMIN[i]; }

#undef REALMAX
#undef REALMIN

    /*
      Union operator
    */

    GenericBoundingBox<T> operator+(const GenericBoundingBox<T>& rhs) const
      {
	if(rhs.isNull() && !isNull()) return *this; /* if one of the boxes are null, 
						       the result of this operation should be the non-null box */
	if(isNull() && !rhs.isNull()) return rhs;
	if(rhs.isNull() && isNull()) return *this;

	GenericBoundingBox<T> ret;
	ret.minx = std::min(minx,rhs.minx);
	ret.miny = std::min(miny,rhs.miny);
	ret.minz = std::min(minz,rhs.minz);
	ret.maxx = std::max(maxx,rhs.maxx);
	ret.maxy = std::max(maxy,rhs.maxy);
	ret.maxz = std::max(maxz,rhs.maxz);

	return ret;
      }

    GenericBoundingBox<T>& operator+=(const GenericBoundingBox<T>& rhs)
      {
	if(rhs.isNull() && !isNull()) return *this;
	if(isNull() && !rhs.isNull())
	  {
	    (*this) = rhs;
	    return *this;
	  }
	if(rhs.isNull() && isNull()) return *this;

	minx = std::min(minx,rhs.minx);
	miny = std::min(miny,rhs.miny);
	minz = std::min(minz,rhs.minz);
	maxx = std::max(maxx,rhs.maxx);
	maxy = std::max(maxy,rhs.maxy);
	maxz = std::max(maxz,rhs.maxz);
	
	return *this;
      }

    /*
      Intersection operator
    */
    GenericBoundingBox<T> operator-(const GenericBoundingBox<T>& rhs) const
      {
	if(rhs.isNull()) return rhs; /* if one of the boxes are null, 
					the result of this operation should be null */
	if(isNull()) return *this;
	GenericBoundingBox<T> ret;
	ret.minx = std::max(minx,rhs.minx);
	ret.miny = std::max(miny,rhs.miny);
	ret.minz = std::max(minz,rhs.minz);
	ret.maxx = std::min(maxx,rhs.maxx);
	ret.maxy = std::min(maxy,rhs.maxy);
	ret.maxz = std::min(maxz,rhs.maxz);

	/*
	  check to see if there is no intersection.  If there isn't, set ret
	  to null (else it will cause an exception on future operations).
	*/
	if(ret.minx > ret.maxx ||
	   ret.miny > ret.maxy ||
	   ret.minz > ret.maxz)
	  ret.minx = ret.maxx = ret.miny = ret.maxy = ret.minz = ret.maxz = T(0);

	return ret;
      }

    GenericBoundingBox<T>& operator-=(const GenericBoundingBox<T>& rhs)
      {
	if(rhs.isNull())
	  {
	    *this = rhs;
	    return *this;
	  }
	if(isNull()) return *this;
	minx = std::max(minx,rhs.minx);
	miny = std::max(miny,rhs.miny);
	minz = std::max(minz,rhs.minz);
	maxx = std::min(maxx,rhs.maxx);
	maxy = std::min(maxy,rhs.maxy);
	maxz = std::min(maxz,rhs.maxz);

	/*
	  check to see if there is no intersection.  If there isn't, set ret
	  to null (else it will cause an exception on future operations).
	*/
	if(minx > maxx ||
	   miny > maxy ||
	   minz > maxz)
	  minx = maxx = miny = maxy = minz = maxz = T(0);

	return *this;
      }

    bool isWithin(const GenericBoundingBox<T>& b) const
    {
      if(b.isNull()) return false;
      if(minx >= b.minx && miny >= b.miny && minz >= b.minz &&
	 maxx <= b.maxx && maxy <= b.maxy && maxz <= b.maxz)
	return true;
      return false;
    }

    bool contains(T x, T y, T z) const
    {
      if(isNull()) return false;
      if(x >= minx && y >= miny && z >= minz &&
	 x <= maxx && y <= maxy && z <= maxz)
	return true;
      return false;
    }

    bool operator==(const GenericBoundingBox<T>& b) const
    {
      return isWithin(b) && b.isWithin(*this);
    }

    bool operator!=(const GenericBoundingBox<T>& b) const
    {
      return !((*this)==b);
    }

    //lexographic ordering of bounding boxes only
    bool operator<(const GenericBoundingBox<T>& b) const
    {
      return minx < b.minx && miny < b.miny && minz < b.minz;
    }

    bool isNull() const
    {
      return std::fabs(volume()) >= 0 && std::fabs(volume()) <= 0;
    }

    T volume() const
    {
      checkBounds();
      return T(maxx-minx)*T(maxy-miny)*T(maxz-minz);
    }

    T XMax() const { return maxx; }
    T XMin() const { return minx; }
    T YMax() const { return maxy; }
    T YMin() const { return miny; }
    T ZMax() const { return maxz; }
    T ZMin() const { return minz; }

    // 09/10/2011 -- Joe R. -- Adding span calculation via Dimension object
    double XSpan(const Dimension& dim) const { return dim.XDim()-1 == 0 ? 1.0 : (maxx-minx)/(dim.XDim()-1); }
    double YSpan(const Dimension& dim) const { return dim.YDim()-1 == 0 ? 1.0 : (maxy-miny)/(dim.YDim()-1); }
    double ZSpan(const Dimension& dim) const { return dim.ZDim()-1 == 0 ? 1.0 : (maxz-minz)/(dim.ZDim()-1); }

    void normalize()
    {
      *this = GenericBoundingBox<T>(std::min(minx,maxx),
				    std::min(miny,maxy),
				    std::min(minz,maxz),
				    std::max(minx,maxx),
				    std::max(miny,maxy),
				    std::max(minz,maxz));
    }

    //conversion to/from csv - transfix - 04/06/2012
    std::string str() const
    {
      using namespace boost;
      return boost::str(format("%1%,%2%,%3%,%4%,%5%,%6%")
                        % minx % miny % minz
                        % maxx % maxy % maxz);
    }

    void str(const std::string& s) throw(InvalidBoundingBoxString)
    {
      using namespace std;
      using namespace boost;
      using namespace boost::algorithm;
      vector<string> parts;
      split(parts,s,is_any_of(","));
      if(parts.size()!=6) throw InvalidBoundingBoxString(s);
      try
        {
          for(int i = 0; i < 6; i++)
            {
              trim(parts[i]);
              (*this)[i] = lexical_cast<T>(parts[i]);
            }
        }
      catch(std::exception& e)
        {
          throw InvalidBoundingBoxString(e.what());
        }
    }

  private:
    void checkBounds() const throw(InvalidBoundingBox)
    {
      using namespace boost;
      std::string buf;
      if(minx > maxx)
	{
          buf = boost::str(format("minx: %f, maxx: %f")
                           % double(minx)
                           % double(maxx));
	  throw InvalidBoundingBox(buf);
	}
      else if(miny > maxy)
	{
          buf = boost::str(format("miny: %f, maxy: %f")
                           % double(miny)
                           % double(maxy));
	  throw InvalidBoundingBox(buf);
	}
      else if(minz > maxz)
	{
          buf = boost::str(format("minz: %f, maxz: %f")
                           % double(minz)
                           % double(maxz));
	  throw InvalidBoundingBox(buf);
	}
    }
  };

  typedef GenericBoundingBox<double> BoundingBox; // object space
  typedef GenericBoundingBox<uint64> IndexBoundingBox; // image space
};

#endif
