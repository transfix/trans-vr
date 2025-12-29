/*
  Copyright 2011 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TexMol.

  TexMol is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TexMol is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/
#ifndef __HISTOGRAM_DATA_H__
#define __HISTOGRAM_DATA_H__

#include <cassert>
#include <iostream>
#include <vector>

class HistogramData {
public:
  HistogramData();
  HistogramData(const std::vector<double> &inputPoints, unsigned int numBins);
  ~HistogramData();
  void debug();
  void save();
  void load();
  int width() {
    if (!_initialized) {
      return 0;
    }
    return (int)_bins.size();
  }
  unsigned int height() {
    if (!_initialized) {
      return 0;
    }
    return _binMax;
  }
  double getBin(int i) {
    if (!_initialized) {
      return 0;
    }
    if (i < 0 || i >= width()) {
      return 0;
    }
    return _bins[i];
  }
  double getBinNormalized(int i) {
    if (!_initialized) {
      return 0;
    }
    if (i < 0 || i >= width()) {
      return 0;
    }
    return _bins[i] / (double)_binMax;
  }
  void rebin(int newBins);
  double binToWidth(int bin);

private:
  void add(double input);
  bool _initialized;
  double _inputMin, _inputMax;
  unsigned int _binMin, _binMax;
  std::vector<double> _inputPoints;
  std::vector<unsigned int> _bins;
  double _binWidth;
};

#endif
