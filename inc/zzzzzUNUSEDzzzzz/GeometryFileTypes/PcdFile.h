/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	         Jose Rivera <transfix@ices.utexas.edu>
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

// PcdFile.h: interface for the PcdFile class.
//
//////////////////////////////////////////////////////////////////////

/* $Id: PcdFile.h 1498 2010-03-10 22:50:29Z transfix $ */

#ifndef __GEOMETRYFILETYPES__PCDFILE_H__
#define __GEOMETRYFILETYPES__PCDFILE_H__

#include <GeometryFileTypes/GeometryFileType.h>

//using std::string;

///\class PcdFile PcdFile.h
///\author Jose Rivera
///\brief This is a GeometryFileType instance for reading and writing pcds files (point cloud data)
class PcdFile : public GeometryFileType  
{
public:
	virtual ~PcdFile();

	virtual Geometry* loadFile(const string& fileName);
	virtual bool checkType(const string& fileName);
	virtual bool saveFile(const Geometry* geometry, const string& fileName);

	virtual string extension() { return "pcd"; };
	virtual string filter() { return "Pcd files (*.pcd)"; };

	static PcdFile ms_PcdFileRepresentative;
	static GeometryFileType* getRepresentative();

protected:
	PcdFile();
};

#endif
