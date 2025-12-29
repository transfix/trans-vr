/*
  Copyright 2007-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolMagick.

  VolMagick is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolMagick is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: Exceptions.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_EXCEPTIONS_H__
#define __VOLMAGICK_EXCEPTIONS_H__

#include <CVC/Exception.h>
#include <boost/format.hpp>
#include <exception>
#include <string>

namespace VolMagick {
typedef CVC::Exception Exception;

#define VOLMAGICK_DEF_EXCEPTION(name)                                        \
  class name : public VolMagick::Exception {                                 \
  public:                                                                    \
    name() : _msg("VolMagick::" #name) {}                                    \
    name(const std::string &msg)                                             \
        : _msg(boost::str(                                                   \
              boost::format("VolMagick::" #name " exception: %1%") % msg)) { \
    }                                                                        \
    virtual ~name() throw() {}                                               \
    virtual const std::string &what_str() const throw() { return _msg; }     \
                                                                             \
  private:                                                                   \
    std::string _msg;                                                        \
  }

VOLMAGICK_DEF_EXCEPTION(ReadError);
VOLMAGICK_DEF_EXCEPTION(WriteError);
VOLMAGICK_DEF_EXCEPTION(MemoryAllocationError);
VOLMAGICK_DEF_EXCEPTION(SubVolumeOutOfBounds);
VOLMAGICK_DEF_EXCEPTION(UnsupportedVolumeFileType);
VOLMAGICK_DEF_EXCEPTION(IndexOutOfBounds);
VOLMAGICK_DEF_EXCEPTION(NullDimension);
VOLMAGICK_DEF_EXCEPTION(VolumePropertiesMismatch);
VOLMAGICK_DEF_EXCEPTION(VolumeCacheDirectoryFileError);
}; // namespace VolMagick

#endif
