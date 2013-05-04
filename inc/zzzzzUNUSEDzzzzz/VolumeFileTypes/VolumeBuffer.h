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

// VolumeBuffer.h: interface for the VolumeBuffer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMEBUFFER_H__33096BC0_D397_43B7_B1E3_92690AB920F0__INCLUDED_)
#define AFX_VOLUMEBUFFER_H__33096BC0_D397_43B7_B1E3_92690AB920F0__INCLUDED_

class QFile;

///\class VolumeBuffer VolumeBuffer.h
///\author Anthony Thane
///\brief VolumeBuffer is a fancy buffer for holding volume data.
class VolumeBuffer  
{
public:
	VolumeBuffer();
	virtual ~VolumeBuffer();

///\fn bool allocateMemory(unsigned int width, unsigned int height, unsigned int depth, unsigned int voxelSize = 1)
///\brief Allocates the buffer needed to hold the volume data
///\param width The width of the volume
///\param height The height of the volume
///\param depth The depth of the volume
///\param voxelSize The number of bytes needed for each voxel (sample)
///\return A bool indicating sucess or failure
	bool allocateMemory(unsigned int width, unsigned int height, unsigned int depth, unsigned int voxelSize = 1);

///\fn void setMin(double minX, double minY, double minZ)
///\brief Sets the minimum extent of the volume
///\param minX The minimum X coordinate
///\param minY The minimum Y coordinate
///\param minZ The minimum Z coordinate
	void setMin(double minX, double minY, double minZ);
///\fn void setMax(double maxX, double maxY, double maxZ)
///\brief Sets the maximum extent of the volume
///\param maxX The maximum X coordinate
///\param maxY The maximum Y coordinate
///\param maxZ The maximum Z coordinate
	void setMax(double maxX, double maxY, double maxZ);

///\fn unsigned char* getBuffer()
///\brief Returns a pointer to the internal buffer
///\return A pointer to the buffer
	unsigned char* getBuffer();
///\fn unsigned int getWidth() const
///\brief Returns the width of the buffer in voxels
///\return The width of the buffer
	unsigned int getWidth() const;
///\fn unsigned int getHeight() const
///\brief Returns the height of the buffer in voxels
///\return The height of the buffer
	unsigned int getHeight() const;
///\fn unsigned int getDepth() const
///\brief Returns the depth of the buffer in voxels
///\return The depth of the buffer
	unsigned int getDepth() const;
///\fn double getMinX() const
///\brief Returns the minimum X coordinate of the bounding box (extent)
///\return The minimum X coordinate
	double getMinX() const;
///\fn double getMinY() const
///\brief Returns the minimum Y coordinate of the bounding box (extent)
///\return The minimum Y coordinate
	double getMinY() const;
///\fn double getMinZ() const
///\brief Returns the minimum Z coordinate of the bounding box (extent)
///\return The minimum Z coordinate
	double getMinZ() const;
///\fn double getMaxX() const
///\brief Returns the maximum X coordinate of the bounding box (extent)
///\return The maximum X coordinate
	double getMaxX() const;
///\fn double getMaxY() const
///\brief Returns the maximum Y coordinate of the bounding box (extent)
///\return The maximum Y coordinate
	double getMaxY() const;
///\fn double getMaxZ() const
///\brief Returns the maximum Y coordinate of the bounding box (extent)
///\return The maximum Y coordinate
	double getMaxZ() const;

	// obsolete
	//bool saveBuffer(const char* filename) const;

protected:
	void setDefaults();

	// obsolete
	//void writeHeader(QFile& file) const;

	bool forceAllocateMemory(unsigned int amount);
	void destroyMemory();

	unsigned int m_MemoryAllocated;

	unsigned int m_DimX, m_DimY, m_DimZ;
	double m_MinX, m_MinY, m_MinZ;
	double m_MaxX, m_MaxY, m_MaxZ;

	unsigned char* m_Buffer;
};

#endif // !defined(AFX_VOLUMEBUFFER_H__33096BC0_D397_43B7_B1E3_92690AB920F0__INCLUDED_)
