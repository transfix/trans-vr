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

// RawFile.h: interface for the RawFile class.
//
//////////////////////////////////////////////////////////////////////

/* $Id: PcdsFile.h 1498 2010-03-10 22:50:29Z transfix $ */

#ifndef __GEOMETRYFILETYPES__PCDSFILE_H__
#define __GEOMETRYFILETYPES__PCDSFILE_H__

#include <GeometryFileTypes/GeometryFileType.h>

//using std::string;

///\class PcdsFile PcdsFile.h
///\author Jose Rivera
///\brief This is a GeometryFileType instance for reading and writing pcds files (point cloud with scalar for each point)
class PcdsFile : public GeometryFileType  
{
public:
	virtual ~PcdsFile();

	virtual Geometry* loadFile(const string& fileName);
	virtual bool checkType(const string& fileName);
	virtual bool saveFile(const Geometry* geometry, const string& fileName);

	virtual string extension() { return "pcds"; };
	virtual string filter() { return "Pcds files (*.pcds)"; };

	static PcdsFile ms_PcdsFileRepresentative;
	static GeometryFileType* getRepresentative();

protected:
	PcdsFile();
};

#endif
