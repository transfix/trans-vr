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

/* $Id: Voxels.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef __VOLMAGICK_VOXELS_H__
#define __VOLMAGICK_VOXELS_H__

#include <VolMagick/Dimension.h>
#include <VolMagick/Exceptions.h>
#include <VolMagick/Types.h>
#include <algorithm>
#include <boost/shared_array.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstring>

namespace VolMagick {
class VoxelOperationStatusMessenger;
class CompositeFunction;

class Voxels {
public:
  Voxels(const Dimension &d = Dimension(4, 4, 4), VoxelType vt = CVC::UChar);
  Voxels(const void *v, const Dimension &d, VoxelType vt);
  Voxels(const Voxels &v);
  virtual ~Voxels();

  /*
    Voxels Dimensions
  */
  Dimension &dimension() { return _dimension; }
  const Dimension &dimension() const { return _dimension; }
  virtual void dimension(const Dimension &d,
                         boost::shared_array<unsigned char> voxels =
                             boost::shared_array<unsigned char>());
  uint64 XDim() const { return dimension().xdim; }
  uint64 YDim() const { return dimension().ydim; }
  uint64 ZDim() const { return dimension().zdim; }

  /*
    Voxel I/O
  */
  double operator()(uint64 i) const /* reading a voxel value */
  {
    if (i >= XDim() * YDim() * ZDim())
      throw IndexOutOfBounds("");

    switch (voxelType()) {
    case CVC::UChar:
      return double(*((unsigned char *)(_voxels.get() + i * voxelSize())));
    case CVC::UShort:
      return double(*((unsigned short *)(_voxels.get() + i * voxelSize())));
    case CVC::UInt:
      return double(*((unsigned int *)(_voxels.get() + i * voxelSize())));
    case CVC::Float:
      return double(*((float *)(_voxels.get() + i * voxelSize())));
    case CVC::Double:
      return double(*((double *)(_voxels.get() + i * voxelSize())));
    case CVC::UInt64:
      return double(*((uint64 *)(_voxels.get() + i * voxelSize())));
    }
    return 0;
  }
  double operator()(uint64 i, uint64 j,
                    uint64 k) const /* reading a voxel value */
  {
    return (*this)(i + j * XDim() + k * XDim() * YDim());
  }

  void operator()(uint64 i, double val) /* writing a voxel value */
  {
    if (i >= XDim() * YDim() * ZDim())
      throw IndexOutOfBounds("");

    preWrite();

    switch (voxelType()) {
    case CVC::UChar:
      *((unsigned char *)(_voxels.get() + i * voxelSize())) =
          (unsigned char)(val);
      break;
    case CVC::UShort:
      *((unsigned short *)(_voxels.get() + i * voxelSize())) =
          (unsigned short)(val);
      break;
    case CVC::UInt:
      *((unsigned int *)(_voxels.get() + i * voxelSize())) =
          (unsigned int)(val);
      break;
    case CVC::Float:
      *((float *)(_voxels.get() + i * voxelSize())) = float(val);
      break;
    case CVC::Double:
      *((double *)(_voxels.get() + i * voxelSize())) = double(val);
      break;
    case CVC::UInt64:
      *((uint64 *)(_voxels.get() + i * voxelSize())) = uint64(val);
    }

    // NOTE: we cant modify min/max here because it would mess up a map()
    // operation, and perhaps other things if(_minIsSet && val < min())
    // min(val); if(_maxIsSet && val > max()) max(val);
  }
  void operator()(uint64 i, uint64 j, uint64 k,
                  double val) /* writing a voxel value */
  {
    (*this)(i + j * XDim() + k * XDim() * YDim(), val);
  }

  unsigned char *operator*() {
    preWrite();
    return _voxels.get();
  }
  const unsigned char *operator*() const { return _voxels.get(); }

  VoxelType voxelType() const { return _voxelType; }
  void voxelType(VoxelType);
  uint64 voxelSize() const { return VoxelTypeSizes[voxelType()]; }
  const char *voxelTypeStr() const { return VoxelTypeStrings[voxelType()]; }

  /* min and max values */
  double min() const {
    if (!_minIsSet)
      calcMinMax();
    return _min;
  }
  void min(double m) {
    _min = m;
    _minIsSet = true;
  }
  double max() const {
    if (!_maxIsSet)
      calcMinMax();
    return _max;
  }
  void max(double m) {
    _max = m;
    _maxIsSet = true;
  }
  void unsetMinMax() { _minIsSet = _maxIsSet = false; }
  bool minIsSet() const { return _minIsSet; }
  bool maxIsSet() const { return _maxIsSet; }

  /* calculate min and max values for selected subvolumes */
  double min(uint64 off_x, uint64 off_y, uint64 off_z,
             const Dimension &dim) const;
  double max(uint64 off_x, uint64 off_y, uint64 off_z,
             const Dimension &dim) const;

  Voxels &operator=(const Voxels &vox) {
    copy(vox);
    return *this;
  }

  bool operator==(const Voxels &vox) const {
    return (_voxels == vox._voxels) ||
           (strncmp(reinterpret_cast<const char *>(_voxels.get()),
                    reinterpret_cast<const char *>(vox._voxels.get()),
                    std::min(dimension().size(), vox.dimension().size())) ==
            0);
  }

  bool operator!=(const Voxels &vox) const { return !(*this == vox); }

  void messenger(const VoxelOperationStatusMessenger *vosm) { _vosm = vosm; }
  const VoxelOperationStatusMessenger *messenger() const { return _vosm; }

  boost::tuple<const uint64 *, uint64> histogram(uint64 size = 256) const {
    calcHistogram(size);
    return boost::make_tuple(_histogram.get(), _histogramSize);
  }

  /*
    operations!
  */
  virtual Voxels &
  copy(const Voxels &vox); // turns this object into a copy of vox
  // subvolume extraction: removes voxels outside of the subvolume specified
  virtual Voxels &sub(uint64 off_x, uint64 off_y, uint64 off_z,
                      const Dimension &subvoldim);
  Voxels &fill(double val); // set all voxels to the specified value
  Voxels &fillsub(uint64 off_x, uint64 off_y, uint64 off_z,
                  const Dimension &subvoldim,
                  double val); // set all voxels in specified subvolume to val
  Voxels &map(double min_, double max_); // maps voxels from min to max
  Voxels &
  resize(const Dimension &newdim); // resizes this object to the specified
                                   // dimension using trilinear interpolation
  Voxels &bilateralFilter(double radiometricSigma = 200.0,
                          double spatialSigma = 1.5,
                          unsigned int filterRadius = 2);
  // Voxels& rotate(double deg_x, double deg_y, double deg_z); //rotates the
  // object about the x,y,z axis
  /*
    compose vox into this object using the specified composite function.  Yes,
    the offset may be negative. Only the voxels that overlap will be subject
    to the composition function.
  */
  virtual Voxels &composite(const Voxels &compVox, int64 off_x, int64 off_y,
                            int64 off_z, const CompositeFunction &func);

  /*
   * Zeyun's Contrast enhancement: enhances contrast between voxel values.
   * 'resistor' must be a value between 0.0 and 1.0 Requres memory to hold the
   * original volume + 6x the original volume using float values for voxels...
   */
  virtual Voxels &contrastEnhancement(double resistor = 0.95);

  /*
   * Zeyun's anisotropic diffusion: filters noise but preserves edges more
   * than bilateral filter
   */
  virtual Voxels &anisotropicDiffusion(unsigned int iterations = 20);

  /*
   * Dr. Zhang's gdtv filter.
   */
  virtual Voxels &gdtvFilter(double parameterq, double lambda,
                             unsigned int iteration, unsigned int neigbour);

protected:
  void calcMinMax() const;
  void preWrite() {
    _histogramDirty = true; // invalidate the histogram

    if (_voxels.unique())
      return; // nothing to copy if our voxels are already unique

    try {
      boost::shared_array<unsigned char> tmp(_voxels);
      _voxels.reset(
          new unsigned char[XDim() * YDim() * ZDim() * voxelSize()]);
      memcpy(_voxels.get(), tmp.get(),
             XDim() * YDim() * ZDim() * voxelSize());
    } catch (std::bad_alloc &e) {
      throw MemoryAllocationError(
          "Could not allocate memory for voxels during copy-on-write!");
    }
  }
  void calcHistogram(uint64 size) const;

  // unsigned char *_voxels;
  boost::shared_array<unsigned char> _voxels;

  Dimension _dimension;

  VoxelType _voxelType;

  mutable bool _minIsSet;
  mutable double _min;
  mutable bool _maxIsSet;
  mutable double _max;

  const VoxelOperationStatusMessenger *_vosm;

  // computed on demand even for const reference so declare as mutable
  mutable boost::shared_array<uint64> _histogram;
  mutable uint64 _histogramSize;
  mutable bool _histogramDirty;
};
} // namespace VolMagick

#endif
