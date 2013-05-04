/*
  Copyright 2002-2008 The University of Texas at Austin
  
  Authors: Jose Rivera <transfix@ices.utexas.edu>
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

/* $Id: LinecFile.cpp 1787 2010-06-01 18:51:21Z transfix $ */

// LinecFile.cpp: implementation of the LinecFile class.
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include <iostream>
#include <fstream>
#include <GeometryFileTypes/LinecFile.h>
#include <cvcraw_geometry/Geometry.h>
//#include <qfileinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

#include <boost/scoped_ptr.hpp>

using namespace std;

LinecFile LinecFile::ms_LinecFileRepresentative;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

LinecFile::LinecFile()
{

}

LinecFile::~LinecFile()
{

}

Geometry* LinecFile::loadFile(const string& fileName)
{
  ifstream inf(fileName.c_str());
  if(!inf)
    {
      cerr << "Error opening " << fileName << endl;
      return 0;
    }
  
  unsigned int numVerts, numLines;
  inf >> numVerts >> numLines;
  if(!inf)
    {
      cerr << "Error reading num verts or num lines." << endl;
      return 0;
    }

  // initialize the geometry
  boost::scoped_ptr<Geometry> geometry(new Geometry());
  geometry->AllocateLines(numVerts, numLines);
  geometry->AllocateLineColors();

  for(unsigned int i = 0; i < numVerts; i++)
    {
      inf >> geometry->m_LineVerts[i*3+0]
	  >> geometry->m_LineVerts[i*3+1]
	  >> geometry->m_LineVerts[i*3+2]
	  >> geometry->m_LineColors[i*3+0]
	  >> geometry->m_LineColors[i*3+1]
	  >> geometry->m_LineColors[i*3+2];
      if(!inf)
	{
	  cerr << "Error reading vertex " << i << endl;
	  return 0;
	}
    }

  for(unsigned int i = 0; i < numLines; i++)
    {
      inf >> geometry->m_Lines[i*2+0]
	  >> geometry->m_Lines[i*2+1];
      if(!inf)
	{
	  cerr << "Error reading line " << i << endl;
	  return 0;
	}

      //check bounds
      if(geometry->m_Lines[i*2+0] >= numVerts ||
	 geometry->m_Lines[i*2+1] >= numVerts)
	{
	  cerr << "Line index out of bounds at line " << i << endl;
	  return 0;
	}
    }

  return new Geometry(*geometry.get());
}

bool LinecFile::checkType(const string& fileName)
{
  ifstream inf(fileName.c_str());
  if(!inf)
    return false;
  
  unsigned int numVerts, numLines;
  inf >> numVerts >> numLines;
  if(!inf)
    return false;

  for(unsigned int i = 0; i < numVerts; i++)
    {
      float x,y,z,r,g,b;
      inf >> x >> y >> z >> r >> g >> b;
      if(!inf)
	return false;
    }

  for(unsigned int i = 0; i < numLines; i++)
    {
      unsigned int line[2];
      inf >> line[0] >> line[1];
      if(!inf)
	return false;

      //check bounds
      if(line[0] >= numVerts ||
	 line[1] >= numVerts)
	return false;
    }

  return true;
}

bool LinecFile::saveFile(const Geometry* geometry, const string& fileName)
{
  ofstream outf(fileName.c_str());
  if(!outf)
    {
      cerr << "Error opening file " << fileName << endl;
      return false;
    }

  outf << geometry->m_NumLineVerts << " " << geometry->m_NumLines << endl;
  if(!outf)
    {
      cerr << "Error writing number of line verts or number of lines." << endl;
      return false;
    }
  
  for(unsigned int i = 0; i < geometry->m_NumLineVerts; i++)
    {
      outf << geometry->m_LineVerts[i*3+0] << " "
	   << geometry->m_LineVerts[i*3+1] << " "
	   << geometry->m_LineVerts[i*3+2] << " "
	   << geometry->m_LineColors[i*3+0] << " "
	   << geometry->m_LineColors[i*3+1] << " "
	   << geometry->m_LineColors[i*3+2] << endl;
      if(!outf)
	{
	  cerr << "Error writing vertex " << i << endl;
	  return false;
	}
    }

  for(unsigned int i = 0; i < geometry->m_NumLines; i++)
    {
      outf << geometry->m_Lines[i*2+0] << " "
	   << geometry->m_Lines[i*2+1] << endl;
      if(!outf)
	{
	  cerr << "Error writing lines " << i << endl;
	  return false;
	}
    }

  return true;
}

GeometryFileType* LinecFile::getRepresentative()
{
  return &ms_LinecFileRepresentative;
}

