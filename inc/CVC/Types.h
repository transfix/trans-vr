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

/* $Id: Types.h 5143 2012-02-19 04:57:36Z transfix $ */

#ifndef __CVC_TYPES_H__
#define __CVC_TYPES_H__

#include <CVC/Namespace.h>

#include <boost/cstdint.hpp>
#include <boost/signals2.hpp>
#include <boost/function.hpp>
#include <boost/any.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

#include <map>
#include <string>
#include <vector>

#ifndef CVC_VERSION_STRING
#define CVC_VERSION_STRING "1.0.0"
#endif

#define CVC_ENABLE_LOCALE_BOOL
#ifdef CVC_ENABLE_LOCALE_BOOL
#include <iostream>
#endif

namespace CVC_NAMESPACE
{
  typedef boost::int64_t  int64;
  typedef boost::uint64_t uint64;

  enum DataType 
    { 
      UChar = 0, 
      UShort, 
      UInt, 
      Float, 
      Double, 
      UInt64,
      Char,
      Int,
      Int64,
      Undefined
    };

  static const unsigned int DataTypeSizes[] = 
    { 
      sizeof(unsigned char), 
      sizeof(unsigned short), 
      sizeof(unsigned int), 
      sizeof(float), 
      sizeof(double), 
      sizeof(uint64),
      sizeof(char),
      sizeof(int),
      sizeof(int64),
      0
    };

  static const char * DataTypeStrings[] = 
    {
      "unsigned char",
      "unsigned short",
      "unsigned int",
      "float",
      "double",
      "uint64",
      "char",
      "int",
      "int64",
      "void"
    };


  //This is to be used with boost::lexical_cast<> like so
  // bool b = boost::lexical_cast< CVC_NAMESPACE::LocaleBool >("true");
  //Found here on stack overflow: http://bit.ly/oR1wnk
#ifdef CVC_ENABLE_LOCALE_BOOL
  struct LocaleBool {
    bool data;
    LocaleBool() {}
    LocaleBool( bool data ) : data(data) {}
    operator bool() const { return data; }
    friend std::ostream & operator << ( std::ostream &out, LocaleBool b ) {
        out << std::boolalpha << b.data;
        return out;
    }
    friend std::istream & operator >> ( std::istream &in, LocaleBool &b ) {
        in >> std::boolalpha >> b.data;
        return in;
    }
  };
#endif

  typedef boost::signals2::signal<void ()>                   Signal;
  typedef boost::signals2::signal<void (const std::string&)> MapChangeSignal;
  typedef std::map<std::string, boost::any>                  DataMap;
  typedef std::map<std::string, std::string>                 DataTypeNameMap;
  typedef std::map<std::string, DataType>                    DataTypeEnumMap;
  typedef std::map<std::string, std::string>                 PropertyMap;
  typedef boost::shared_ptr<boost::thread>                   ThreadPtr;
  typedef std::map<std::string, ThreadPtr>                   ThreadMap;
  typedef std::map<boost::thread::id, double>                ThreadProgressMap;
  typedef std::map<boost::thread::id, std::string>           ThreadKeyMap;
  typedef std::map<boost::thread::id, std::string>           ThreadInfoMap;
  typedef boost::function<bool (const std::string&)>         DataReader;
  typedef std::vector<DataReader>                            DataReaderCollection;
  typedef boost::shared_ptr<boost::mutex>                    MutexPtr;
  typedef boost::tuple<MutexPtr,std::string>                 MutexMapElement;
  typedef std::map<std::string, MutexMapElement>             MutexMap;
}

#endif
