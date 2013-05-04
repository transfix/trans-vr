/*
  Copyright 2000-2003 The University of Texas at Austin

	Authors: Xiaoyu Zhang 2000-2002 <xiaoyu@ices.utexas.edu>
					 John Wiggins 2003 <prok@cs.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of iotree.

  iotree is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  iotree is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef SLICE_FACTORY_H
#define SLICE_FACTORY_H

#include <c2c_codec/slice.h>

/**
*  The interface of getting slices from various sources
*/
template <class T>
class  SliceFactory {
public:
	SliceFactory() {}

	///
	virtual ~SliceFactory() {}

	///
	virtual Slice<T>* getNextSlice() = 0;

	///	rewind to the first slices
	virtual void reset() {}

	///	get the total number of slices
	virtual int  getNumSlices() {return 0;}

	/// get the dimension info of the data
	virtual void getDimension(int dim[3], float orig[3], float span[3]) {}

	/// get the number of current slice
	virtual int current() {return 0;}

};

#endif

