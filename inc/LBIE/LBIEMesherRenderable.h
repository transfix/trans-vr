/*
  Copyright 2006 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of LBIE.

  LBIE is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  LBIE is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __LBIEMESHERRENDERABLE_H__
#define __LBIEMESHERRENDERABLE_H__

//similar to MultiContour class
class LBIEMesherRenderable : public Renderable
{
 public:
  LBIEMesherRenderable();
  virtual ~LBIEMesherRenderable();

  //single variable data load
  void setData(unsigned char* data, 
	       unsigned int width, unsigned int height, unsigned int depth,
	       double aspectX, double aspectY, double aspectZ,
	       double subMinX, double subMinY, double subMinZ,
	       double subMaxX, double subMaxY, double subMaxZ,
	       double minX, double minY, double minZ,
	       double maxX, double maxY, double maxZ);

  //RGBA data load
  void setData(unsigned char* data, unsigned char* red, unsigned char* green, unsigned char* blue,
		unsigned int width, unsigned int height, unsigned int depth,
		double aspectX, double aspectY, double aspectZ,
		double subMinX, double subMinY, double subMinZ,
		double subMaxX, double subMaxY, double subMaxZ,
		double minX, double minY, double minZ,
		double maxX, double maxY, double maxZ);

  
};

#endif
