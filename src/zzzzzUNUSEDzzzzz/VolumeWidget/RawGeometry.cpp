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

// RawGeometry.cpp: implementation of the RawGeometry class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/RawGeometry.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qtextstream.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RawGeometry::RawGeometry(const char* FileName) : Geometry(), m_FileName(FileName)
{
	if (Load(FileName))
		m_bGoodFile = true;
	else 
		m_bGoodFile = false;
}

RawGeometry::~RawGeometry()
{

}

void RawGeometry::SaveAsGeometry(const char* FileName)
{
	m_FileName = FileName;
	Save(FileName);
}

void RawGeometry::SaveGeometry(const char* FileName)
{
	Save(m_FileName);
}

bool RawGeometry::Save(const char* FileName)
{
	return false;
}

bool RawGeometry::isGood() const
{
	return m_bGoodFile;
}

bool RawGeometry::isBad() const
{
	return !m_bGoodFile;
}

QString RawGeometry::ErrorString() const
{
	return m_Error;
}


unsigned int RawGeometry::CheckMax(unsigned int CurrentMax, unsigned int test)
{
	return (CurrentMax>test?CurrentMax:test);
}

unsigned int RawGeometry::CheckMin(unsigned int CurrentMin, unsigned int test)
{
	return (CurrentMin>test?CurrentMin:test);
}

bool RawGeometry::Load(const char* FileName)
{
	unsigned int numverts, numtris, c, min, max;
	QFile file(FileName);
	if (file.exists()) {
		file.open(IO_ReadOnly);
		QTextStream stream(&file);
		stream >> numverts >> numtris;
		this->AllocateTris(numverts, numtris);
		for (c=0; c<numverts*3; c++) {
			stream >> this->m_TriVerts[c];
		}

		// initialize min and max
		if (numtris>0) {
			stream >> this->m_Tris[0];
			max = min = m_Tris[0];
		}

		// read in the rest of the vertices
		for (c=1; c<numtris*3; c++) {
			stream >> this->m_Tris[c];
			max = CheckMax(max, m_Tris[c]);
			min = CheckMin(max, m_Tris[c]);
		}

		if (max == numverts) {
			if (min>0) {
				// file starts its vertex numbering at 1
				for (c=0; c<numtris*3; c++) {
					m_Tris[c]--;
				}
			}
			else {
				// bad file
				ClearGeometry();
				m_Error = "Bad File Format";
				return false;
			}
		}
		if (max>numverts) {
			// bad file
			ClearGeometry();
			m_Error = "Bad File Format";
			return false;
		}
		return true;
	}
	m_Error = "File did not exist";
	return false;
}

