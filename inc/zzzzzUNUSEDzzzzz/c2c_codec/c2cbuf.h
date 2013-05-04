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
#ifndef C2C_BUFFER_H
#define C2C_BUFFER_H

//#define ZP_CODEC

typedef struct {
  int ix, iy;
  unsigned char code;
} Cell;

template <class T> 			   
class C2CBuffer {
public:

	///
	virtual ~C2CBuffer() {}

	/**
	 * get the next slice 
	 * @return The number of relevant vertices in the slcie
	 * @param data The function values on relevant vertices	(out)
	 * @param color The color values on relevant vertices (out)
	 * @param bits The bit map of the slice	(out)
	 */
	virtual int getSlice(T* data, unsigned char *color, unsigned char* bits) = 0;

	/**
	 * get the next layer
	 * @return The number of intersected cells in the layer
	 * @param cells The array of intersected cells (out). The cells array
	 * 				is allocated in the function and freed by caller
	 */
	virtual int getLayer(Cell* &cells) = 0;

	/// rewind to the start
	virtual void reset() {}

	/// get the dimension info
	virtual void getDimension(int dim[3], float orig[3], float span[3]) {}

	/// get the number of current slice
	virtual int currentSlice() { return 0; }

	/// get the number of current layer
	virtual int currentLayer() { return 0; }

	///	get the isovalue of the surface
	virtual float getIsovalue() { return 0; }
};

#endif

