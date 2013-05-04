/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
             Vinay Siddavanahalli <skvinay@cs.utexas.edu>
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

// RawGeometry.h: interface for the RawGeometry class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAWGEOMETRY_H__CFE1C218_1FAB_4918_A6B8_03533448CF48__INCLUDED_)
#define AFX_RAWGEOMETRY_H__CFE1C218_1FAB_4918_A6B8_03533448CF48__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <cvcraw_geometry/Geometry.h>
#include <qstring.h>

class RawGeometry : public Geometry  
{
public:
	RawGeometry(const char* FileName);
	virtual ~RawGeometry();
	void SaveAsGeometry(const char* FileName);
	void SaveGeometry(const char* FileName);

	bool isGood() const;
	bool isBad() const;
	QString ErrorString() const;

protected:
        static unsigned int CheckMax(unsigned int CurrentMax, unsigned int test);
        static unsigned int CheckMin(unsigned int CurrentMin, unsigned int test);

	bool Save(const char* FileName);
	bool Load(const char* FileName);
	QString m_FileName;
	bool m_bGoodFile;
	QString m_Error;
};

#endif // !defined(AFX_RAWGEOMETRY_H__CFE1C218_1FAB_4918_A6B8_03533448CF48__INCLUDED_)
