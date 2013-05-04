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

/* $Id: Exception.h 5224 2012-03-02 21:36:30Z transfix $ */

#ifndef __CVC_EXCEPTIONS_H__
#define __CVC_EXCEPTIONS_H__

#include <CVC/Namespace.h>

#include <boost/format.hpp>
#include <exception>
#include <string>

namespace CVC_NAMESPACE
{
  /***** Exceptions ****/
  class Exception : public std::exception
  {
  public:
    Exception() {}
    virtual ~Exception() throw() {}
    virtual const std::string& what_str() const throw () = 0;
    virtual const char *what () const throw()
    {
      return what_str().c_str();
    }
  };
  
#define CVC_DEF_EXCEPTION(name) \
  class name : public CVC_NAMESPACE::Exception \
  { \
  public: \
    name () : _msg("CVC::"#name) {} \
    name (const std::string& msg) : \
      _msg(boost::str(boost::format("CVC::" #name " exception: %1%") % msg)) {} \
    virtual ~name() throw() {} \
    virtual const std::string& what_str() const throw() { return _msg; } \
  private: \
    std::string _msg; \
  }

  CVC_DEF_EXCEPTION(ReadError);
  CVC_DEF_EXCEPTION(WriteError);
  CVC_DEF_EXCEPTION(MemoryAllocationError);
  CVC_DEF_EXCEPTION(SubVolumeOutOfBounds);
  CVC_DEF_EXCEPTION(UnsupportedVolumeFileType);
  CVC_DEF_EXCEPTION(UnsupportedGeometryFileType);
  CVC_DEF_EXCEPTION(IndexOutOfBounds);
  CVC_DEF_EXCEPTION(NullDimension);
  CVC_DEF_EXCEPTION(VolumePropertiesMismatch);
  CVC_DEF_EXCEPTION(VolumeCacheDirectoryFileError);
  CVC_DEF_EXCEPTION(NetworkError);
  CVC_DEF_EXCEPTION(XmlRpcServerTerminate);
};

#endif

