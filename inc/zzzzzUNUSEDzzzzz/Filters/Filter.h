/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
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
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// Filter.h: interface for the Filter class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_FILTER_H__07B57F3C_DFC3_4D99_8C03_5AA083700663__INCLUDED_)
#define AFX_FILTER_H__07B57F3C_DFC3_4D99_8C03_5AA083700663__INCLUDED_

class QWidget;

///\class Filter Filter.h
///\author Anthony Thane
///\brief Filter is an abstract base class for 3D image filters
class Filter  
{
public:
	Filter();
	virtual ~Filter();

///\fn virtual bool applyFilter(QWidget* parent, unsigned char* image, unsigned int width, unsigned int height, unsigned int depth)
///\brief This function does the work of the filter.
///\param parent A QWidget that can be used by the filter to construct any UI needed
///\param image The image to be filtered
///\param width The width of the image
///\param height The height of the image
///\param depth The depth of the image
///\return A bool indicating success or failure
	virtual bool applyFilter(QWidget* parent, unsigned char* image, unsigned int width, unsigned int height,
		unsigned int depth) = 0;

};

#endif // !defined(AFX_FILTER_H__07B57F3C_DFC3_4D99_8C03_5AA083700663__INCLUDED_)
