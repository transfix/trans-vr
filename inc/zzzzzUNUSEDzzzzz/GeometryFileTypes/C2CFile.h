/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: John Wiggins <prok@ices.utexas.edu>
					 Anthony Thane <thanea@ices.utexas.edu>
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

// C2CFile.h: interface for the C2CFile class.
//
//////////////////////////////////////////////////////////////////////

#ifndef C2C_FILE_H
#define C2C_FILE_H

#include <GeometryFileTypes/GeometryFileType.h>

///\class C2CFile C2CFile.h
///\author John Wiggins
///\brief This is a GeometryFileType that reads c2c files.
///\warning This class does not write c2c files
class C2CFile : public GeometryFileType  
{
public:
	virtual ~C2CFile();

	virtual Geometry* loadFile(const string& fileName);
	virtual bool checkType(const string& fileName);
	virtual bool saveFile(const Geometry* geometry, const string& fileName);

	virtual string extension() { return "c2c"; };
	virtual string filter() { return "C2C files (*.c2c)"; };

	static C2CFile ms_C2CFileRepresentative;
	static GeometryFileType* getRepresentative();

protected:
	C2CFile();
};

#endif

