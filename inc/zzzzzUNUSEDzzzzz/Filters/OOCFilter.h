/*
  Copyright 2002-2005 The University of Texas at Austin
  
	Authors: John Wiggins <prok@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// OOCFilter.h: interface for the OutOfCoreFilter class.
//
//////////////////////////////////////////////////////////////////////

#ifndef OOC_FILTER_H
#define OOC_FILTER_H

///\enum OutOfCoreFilter::Type
///\brief These are data types for OutOfCoreFilters

///\class OutOfCoreFilter OOCFilter.h
///\author John Wiggins
///\brief OutOfCoreFilter is an abstract base class very much like Filter, except it is for filters
/// that can be run without having an entire image in memory all at once. The way it should be used is
/// something like this: construct the filter, call initFilter, find out how many slices need to be
/// cached with getNumCacheSlices, add that number of slices to the cache via addSlice, then filter
/// the first slice with filterSlice, then add another slice, then filter a slice, and so on until the
/// last slice has been filtered. See the filtered subvolume saving code in NewVolumeMainWindow and
/// the implementation OOCBilateralFilter for a working example.
class OutOfCoreFilter  
{
public:
	enum Type {U_CHAR=0, U_SHORT, U_INT, FLOAT, DOUBLE};

	OutOfCoreFilter();
	virtual ~OutOfCoreFilter();

///\fn virtual void setDataType(Type t)
///\brief This function sets the data type of the data to be filtered
///\param t A Type
	virtual void setDataType(Type t) = 0;
	
///\fn virtual int getNumCacheSlices() const
///\brief This function returns the number of slices that need to be added to the filter via
/// addSlice() before the the filter can run.
///\return An int giving the number of slices the cache needs to contain for filtering to work
	virtual int getNumCacheSlices() const = 0;

///\fn virtual bool initFilter(unsigned int width, unsigned int height, unsigned int depth, double functionMin, double functionMax)
///\brief This function is called to initialize the filter for a specific function (volume)
///\param width The width of the image to be filtered
///\param height The height of the image to be filtered
///\param depth The depth of the image to be filtered
///\param functionMin The minimum value of the function
///\param functionMax The maximum value of the function
///\return A bool indicating success or failure
	virtual bool initFilter(unsigned int width, unsigned int height, unsigned int depth, double functionMin, double functionMax) = 0;
///\fn virtual void addSlice(void *slice)
///\brief Adds a slice to the filter's cache
///\param slice A slice of image data
	virtual void addSlice(void *slice) = 0;
///\fn virtual bool filterSlice(void *slice)
///\brief Filters a single slice of a 3D image
///\param slice A slice of image data
	virtual bool filterSlice(void *slice) = 0;

};

#endif

