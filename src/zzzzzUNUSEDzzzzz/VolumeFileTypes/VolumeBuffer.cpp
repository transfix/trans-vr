/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane 2002-2003 <thanea@ices.utexas.edu>
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

//////////////////////////////////////////////////////////////////////
// VolumeBuffer.cpp: implementation of the VolumeBuffer class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/VolumeBuffer.h>
//#include <qfile.h>
//#include <ByteSwapping.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeBuffer::VolumeBuffer()
{
	setDefaults();
}

VolumeBuffer::~VolumeBuffer()
{
	destroyMemory();
}

bool VolumeBuffer::allocateMemory(unsigned int width, unsigned int height, unsigned int depth, unsigned int voxelSize)
{
	if (width*height*depth*voxelSize>m_MemoryAllocated) { // must allocate more
		destroyMemory();
		if (forceAllocateMemory(width*height*depth*voxelSize)) {
			m_DimX = width;
			m_DimY = height;
			m_DimZ = depth;
			return true;
		}
		else {
			// failure
			m_DimX = 0;
			m_DimY = 0;
			m_DimZ = 0;
			return false;
		}
	}
	else { // we already have enough
			m_DimX = width;
			m_DimY = height;
			m_DimZ = depth;
			return true;
	}
}

void VolumeBuffer::setMin(double minX, double minY, double minZ)
{
	m_MinX = minX;
	m_MinY = minY;
	m_MinZ = minZ;
}

void VolumeBuffer::setMax(double maxX, double maxY, double maxZ)
{
	m_MaxX = maxX;
	m_MaxY = maxY;
	m_MaxZ = maxZ;
}

unsigned char* VolumeBuffer::getBuffer()
{
	return m_Buffer;
}

unsigned int VolumeBuffer::getWidth() const
{
	return m_DimX;
}

unsigned int VolumeBuffer::getHeight() const
{
	return m_DimY;
}

unsigned int VolumeBuffer::getDepth() const
{
	return m_DimZ;
}

double VolumeBuffer::getMinX() const
{
	return m_MinX;
}

double VolumeBuffer::getMinY() const
{
	return m_MinY;
}

double VolumeBuffer::getMinZ() const
{
	return m_MinZ;
}

double VolumeBuffer::getMaxX() const
{
	return m_MaxX;
}

double VolumeBuffer::getMaxY() const
{
	return m_MaxY;
}

double VolumeBuffer::getMaxZ() const
{
	return m_MaxZ;
}

void VolumeBuffer::setDefaults()
{
	m_Buffer = 0;
	m_MemoryAllocated = 0;
	m_DimX = 0;
	m_DimY = 0;
	m_DimZ = 0;
	setMin(0.0,0.0,0.0);
	setMax(1.0,1.0,1.0);
}

// this is no longer needed
#if 0
bool VolumeBuffer::saveBuffer(const char* filename) const
{
	QFile file(filename);
	if (file.open(IO_WriteOnly)) {
		// write header
		writeHeader(file);

		// write data
		file.writeBlock((char*)m_Buffer, sizeof(unsigned char)*m_DimX*m_DimY*m_DimZ);

		// close
		file.close();
		return true;
	}
	else {
		return false;
	}
}

void VolumeBuffer::writeHeader(QFile& file) const
{
	float floatVals[3];
	unsigned int uintVals[3];

	// write the mins
	floatVals[0] = (float)m_MinX;
	floatVals[1] = (float)m_MinY;
	floatVals[2] = (float)m_MinZ;
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	file.writeBlock((char*)floatVals, sizeof(float)*3);

	// write the maxs
	floatVals[0] = (float)m_MaxX;
	floatVals[1] = (float)m_MaxY;
	floatVals[2] = (float)m_MaxZ;
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	file.writeBlock((char*)floatVals, sizeof(float)*3);

	// write the numverts num cells
	uintVals[0] = (unsigned int)(m_DimX*m_DimY*m_DimZ);
	uintVals[1] = (unsigned int)((m_DimX-1)*(m_DimY-1)*(m_DimZ-1));
	if (isLittleEndian()) swapByteOrder(uintVals, 2);
	file.writeBlock((char*)uintVals, sizeof(unsigned int)*2);

	// write the dimensions
	uintVals[0] = m_DimX;
	uintVals[1] = m_DimY;
	uintVals[2] = m_DimZ;
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	file.writeBlock((char*)uintVals, sizeof(unsigned int)*3);

	// write the "origin"
	floatVals[0] = (float)m_MinX;
	floatVals[1] = (float)m_MinY;
	floatVals[2] = (float)m_MinZ;
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	file.writeBlock((char*)floatVals, sizeof(float)*3);

	// write the span
	floatVals[0] = (float)((double)(m_MaxX-m_MinX)/(m_DimX-1));
	floatVals[1] = (float)((double)(m_MaxY-m_MinY)/(m_DimY-1));
	floatVals[2] = (float)((double)(m_MaxZ-m_MinZ)/(m_DimZ-1));
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	file.writeBlock((char*)floatVals, sizeof(float)*3);

}
#endif

bool VolumeBuffer::forceAllocateMemory(unsigned int amount)
{
	m_Buffer = new unsigned char[amount];
	if (!m_Buffer) {
		m_MemoryAllocated = 0;
		return false;
	}
	else {
		m_MemoryAllocated = amount;
		return true;
	}
}

void VolumeBuffer::destroyMemory()
{
	delete [] m_Buffer;
	m_Buffer = 0;
	m_MemoryAllocated = 0;
	m_DimX = 0;
	m_DimY = 0;
	m_DimZ = 0;
}

