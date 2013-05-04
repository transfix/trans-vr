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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: VolumeCache.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLUMECACHE_H__
#define __VOLUMECACHE_H__

#include <set>
#include <string>
#include <boost/multi_array.hpp>

#include <VolMagick/VolMagick.h>

namespace VolMagick
{
  /*
    Compare the volume of each set of voxels
  */
  struct dimcmp
  {
    bool operator()(const VolumeFileInfo& vfi1, const VolumeFileInfo& vfi2)
    {
      return vfi1.dimension().size() < vfi2.dimension().size();
    }
  };

  class VolumeCache
  {
  public:
    VolumeCache(const Dimension& d = Dimension(128,128,128)) 
      : _maxDimension(d) {}
    
    VolumeCache(const VolumeCache& vc) 
      : _volCacheInfo(vc._volCacheInfo),
      _maxDimension(vc.maxDimension()) {}
    
    ~VolumeCache() {}

    VolumeCache& operator=(const VolumeCache& vc)
      {
	_volCacheInfo = vc._volCacheInfo;
	return *this;
      }

    void maxDimension(const Dimension& d) { _maxDimension = d; }
    Dimension& maxDimension() { return _maxDimension; }
    const Dimension& maxDimension() const { return _maxDimension; }

    void add(const VolumeFileInfo& vfi); //add a volume to the cache

    void clear() { _volCacheInfo.clear(); } //clear the cache

    unsigned int size() const { return _volCacheInfo.size(); } //return the number of volumes in the cache

    BoundingBox boundingBox() const
    {
      if(size()>0)
	return (*(_volCacheInfo.begin())).boundingBox();
      else
	return BoundingBox();
    }

    /*
      Return a volume for the request region.
    */
    Volume get(const BoundingBox& requestRegion, unsigned int var = 0, unsigned int time = 0) const;

  private:
    std::set<VolumeFileInfo, dimcmp> _volCacheInfo;
    Dimension _maxDimension;
    boost::multi_array<double,2> _globalMin; //min and max of all volume levels in the cache
    boost::multi_array<double,2> _globalMax;
  };

  /*
    Build a cache of vfi in specified directory.  Returns built cache.
  */
  VolumeCache buildCache(const VolumeFileInfo& vfi, const std::string& dir);

  /*
    Check to see if the cache of vfi in the specified directory is outdated.
  */
  bool cacheUpdateNeeded(const VolumeFileInfo& vfi, const std::string& dir);

  /*
    Load cache of vfi from specified directory.
  */
  VolumeCache loadCache(const VolumeFileInfo& vfi, const std::string& dir);
};

#endif
