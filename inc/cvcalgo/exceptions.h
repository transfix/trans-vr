/*
  Copyright 2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: exceptions.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef __CVCALGO_EXCEPTIONS_H__
#define __CVCALGO_EXCEPTIONS_H__

#include <boost/format.hpp>
#include <exception>
#include <string>

namespace cvcalgo
{
  /***** exceptions ****/
  class exception : public std::exception
  {
  public:
    exception() {}
    virtual ~exception() throw() {}
    virtual const std::string& what_str() const throw () = 0;
    virtual const char *what () const throw()
    {
      return what_str().c_str();
    }
  };
  
#define CVCALGO_DEF_EXCEPTION(name) \
  class name : public cvcalgo::exception \
  { \
  public: \
    name () : _msg("cvcalgo::"#name) {} \
    name (const std::string& msg) : \
      _msg(boost::str(boost::format("cvcalgo::" #name " exception: %1%") % msg)) {} \
    virtual ~name() throw() {} \
    virtual const std::string& what_str() const throw() { return _msg; } \
  private: \
    std::string _msg; \
  }

  CVCALGO_DEF_EXCEPTION(UnsupportedException);
}

#endif

