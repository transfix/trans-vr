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

/* $Id: VolumeFileInfo.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_VOLUMEFILEINFO_H__
#define __VOLMAGICK_VOLUMEFILEINFO_H__

#include <VolMagick/Types.h>
#include <VolMagick/Dimension.h>
#include <VolMagick/BoundingBox.h>

#include <string>
#include <vector>

namespace VolMagick
{
  class VolumeFileInfo
  {
  public:
    VolumeFileInfo() :
      _dimension(_data._dimension), _boundingBox(_data._boundingBox),
      _minIsSet(_data._minIsSet), _min(_data._min),
      _maxIsSet(_data._maxIsSet), _max(_data._max),
      _numVariables(_data._numVariables), _numTimesteps(_data._numTimesteps), 
      _voxelTypes(_data._voxelTypes), _filename(_data._filename), _names(_data._names),
      _tmin(_data._tmin), _tmax(_data._tmax) {}

    VolumeFileInfo(const std::string& file) :
      _dimension(_data._dimension), _boundingBox(_data._boundingBox),
      _minIsSet(_data._minIsSet), _min(_data._min),
      _maxIsSet(_data._maxIsSet), _max(_data._max),
      _numVariables(_data._numVariables), _numTimesteps(_data._numTimesteps), 
      _voxelTypes(_data._voxelTypes), _filename(_data._filename), _names(_data._names),
      _tmin(_data._tmin), _tmax(_data._tmax)
      { read(file); }

    VolumeFileInfo(const VolumeFileInfo& vfi) :
      _dimension(_data._dimension), _boundingBox(_data._boundingBox),
      _minIsSet(_data._minIsSet), _min(_data._min),
      _maxIsSet(_data._maxIsSet), _max(_data._max),
      _numVariables(_data._numVariables), _numTimesteps(_data._numTimesteps), 
      _voxelTypes(_data._voxelTypes), _filename(_data._filename), _names(_data._names),
      _tmin(_data._tmin), _tmax(_data._tmax),
      _data(vfi._data) {}
    ~VolumeFileInfo() {}

    VolumeFileInfo& operator=(const VolumeFileInfo& vfi)
      {
	if(this == &vfi)
	  return *this;
	_data = vfi._data;
	return *this;
      }

    /*
      call VolumeFileInfo::read() to fill out this object from
      the info in the supplied file header.
    */
    void read(const std::string& filename);

    /***** Volume info accessors *****/
    /*
      Volume Dimensions
    */
    Dimension& dimension() { return _data._dimension; }
    const Dimension& dimension() const { return _data._dimension; }
    void dimension(const Dimension& d) { _data._dimension = d; }
    uint64 XDim() const { return dimension().xdim; }
    uint64 YDim() const { return dimension().ydim; }
    uint64 ZDim() const { return dimension().zdim; }

    /*
      Bounding box in object space
     */
    BoundingBox& boundingBox() { return _data._boundingBox; }
    const BoundingBox& boundingBox() const { return _data._boundingBox; }
    void boundingBox(const BoundingBox& box) { _data._boundingBox = box; }
    
    double XMin() const { return boundingBox().minx; }
    double XMax() const { return boundingBox().maxx; }
    double YMin() const { return boundingBox().miny; }
    double YMax() const { return boundingBox().maxy; }
    double ZMin() const { return boundingBox().minz; }
    double ZMax() const { return boundingBox().maxz; }
    void TMin(double t) { _data._tmin = t; }
    double TMin() const { return _data._tmin; }
    void TMax(double t) { _data._tmax = t; }
    double TMax() const { return _data._tmax; }

    double XSpan() const { return XDim()-1 == 0 ? 1.0 : (boundingBox().maxx-boundingBox().minx)/(XDim()-1); }
    double YSpan() const { return YDim()-1 == 0 ? 1.0 : (boundingBox().maxy-boundingBox().miny)/(YDim()-1); }
    double ZSpan() const { return ZDim()-1 == 0 ? 1.0 : (boundingBox().maxz-boundingBox().minz)/(ZDim()-1); }
    double TSpan() const { return (TMax()-TMin())/numTimesteps(); }

    /* min and max voxel values */
    double min(unsigned int var = 0, unsigned int time = 0) const 
    { 
      if(!isSet()) return std::numeric_limits<double>::max();
      if(!_data._minIsSet[var][time]) calcMinMax(var,time); return _data._min[var][time]; 
    }
    void min(double val, unsigned int var, unsigned int time)
    { if(isSet()) { _data._min[var][time] = val; _data._minIsSet[var][time] = true; } }
    double max(unsigned int var = 0, unsigned int time = 0) const 
    { 
      if(!isSet()) return -std::numeric_limits<double>::max();
      if(!_data._maxIsSet[var][time]) calcMinMax(var,time); return _data._max[var][time];
    }
    void max(double val, unsigned int var, unsigned int time)
    { if(isSet()) { _data._max[var][time] = val; _data._maxIsSet[var][time] = true; } }

    void numVariables(unsigned int vars) { _data._numVariables = vars; _data._voxelTypes.resize(vars); }
    unsigned int numVariables() const { return _data._numVariables; }
    void numTimesteps(unsigned int times) { _data._numTimesteps = times; }
    unsigned int numTimesteps() const { return _data._numTimesteps; }
    std::vector<VoxelType> voxelTypes() const { return _data._voxelTypes; }
    std::vector<VoxelType>& voxelTypes() { return _data._voxelTypes; }
    VoxelType voxelTypes(unsigned int vt) const { return _data._voxelTypes[vt]; }
    VoxelType& voxelTypes(unsigned int vt) { return _data._voxelTypes[vt]; }
    VoxelType voxelType() const { return voxelTypes(0); }
    std::string voxelTypeStr(unsigned vt = 0) const { return std::string(VoxelTypeStrings[voxelTypes(vt)]); }
    uint64 voxelSizes(unsigned int vt = 0) const { return VoxelTypeSizes[voxelTypes(vt)]; }
    uint64 voxelSize() const { return voxelSizes(); }

    std::string filename() const { return _data._filename; }

    std::string name(unsigned int var = 0) const { return _data._names[var]; }

    bool isSet() const { return !filename().empty(); }

    struct Data
    {
      Data() : 
	_numVariables(0), _numTimesteps(0),
	_tmin(0.0), _tmax(0.0) {}

      Dimension _dimension;
      BoundingBox _boundingBox;
      mutable std::vector<std::vector<bool> > _minIsSet;
      mutable std::vector<std::vector<double> > _min;
      mutable std::vector<std::vector<bool> > _maxIsSet;
      mutable std::vector<std::vector<double> > _max;
      unsigned int _numVariables;
      unsigned int _numTimesteps;
      std::vector<VoxelType> _voxelTypes;
      std::string _filename;
      std::vector<std::string> _names;
      double _tmin, _tmax;
    };

  private:
    void calcMinMax(unsigned int var = 0, unsigned int time = 0) const;

    //references to the members of _data
    Dimension& _dimension;
    BoundingBox& _boundingBox;
    std::vector<std::vector<bool> >& _minIsSet;
    std::vector<std::vector<double> >& _min;
    std::vector<std::vector<bool> >& _maxIsSet;
    std::vector<std::vector<double> >& _max;
    unsigned int& _numVariables;
    unsigned int& _numTimesteps;
    std::vector<VoxelType>& _voxelTypes;
    std::string& _filename;
    std::vector<std::string>& _names;
    double& _tmin;
    double& _tmax;

    Data _data;
  };
}

#endif
