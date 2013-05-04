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

// SourceManager.h: interface for the SourceManager class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_SOURCEMANAGER_H__41217F7A_A50F_4B7D_A401_ACDE8120A5DF__INCLUDED_)
#define AFX_SOURCEMANAGER_H__41217F7A_A50F_4B7D_A401_ACDE8120A5DF__INCLUDED_

#include <VolumeFileTypes/VolumeSource.h>

///\class SourceManager SourceManager.h
///\author Anthony Thane
///\brief SourceManager manages access to a VolumeSource instance.
class SourceManager  
{
public:
	SourceManager();
	virtual ~SourceManager();

///\fn void setSource(VolumeSource* source)
///\brief Assigns a VolumeSource instance to be managed
///\param source A VolumeSource instance
	void setSource(VolumeSource* source);
///\fn void resetSource()
///\brief Deletes the current VolumeSource instance
	void resetSource();
///\fn VolumeSource* getSource()
///\brief Returns the current VolumeSource instance
///\return A VolumeSource pointer
	VolumeSource* getSource();
///\fn bool hasSource() const
///\brief Returns whether or not a source has been assigned to the object
///\return A bool indicating whether there is a source or not
	bool hasSource() const;

///\fn unsigned int getNumVerts() const
///\brief Returns the number of vertices (samples) in the volume
///\return The number of vertices
///\bug An unsigned int will definitely overflow for large volumes
	unsigned int getNumVerts() const;
///\fn unsigned int getNumCells() const
///\brief Returns the number of cells in the volume
///\return The number of cells
///\bug An unsigned int will definitely overflow for large volumes
	unsigned int getNumCells() const;

///\fn unsigned int getNumVars() const
///\brief Returns the number of variables in the attached source
///\return The number of variables 
	unsigned int getNumVars() const;
///\fn unsigned int getNumTimeSteps() const
///\brief Returns the number of time steps in the attached source
///\return The number of variables
	unsigned int getNumTimeSteps() const;

///\fn double getMinX() const
///\brief Returns the X value of the minimum extent
///\return The minimum X value
	double getMinX() const;
///\fn double getMinY() const
///\brief Returns the Y value of the minimum extent
///\return The minimum Y value
	double getMinY() const;
///\fn double getMinZ() const
///\brief Returns the Z value of the minimum extent
///\return The minimum Z value
	double getMinZ() const;
///\fn double getMaxX() const
///\brief Returns the X value of the maximum extent
///\return The maximum X value
	double getMaxX() const;
///\fn double getMaxY() const
///\brief Returns the Y value of the maximum extent
///\return The maximum Y value
	double getMaxY() const;
///\fn double getMaxZ() const
///\brief Returns the Z value of the maximum extent
///\return The maximum Z value
	double getMaxZ() const;
///\fn unsigned int getDimX() const
///\brief Returns the X dimension in samples
///\return The X dimension
	unsigned int getDimX() const;
///\fn unsigned int getDimY() const
///\brief Returns the Y dimension in samples
///\return The Y dimension
	unsigned int getDimY() const;
///\fn unsigned int getDimZ() const
///\brief Returns the Z dimension in samples
///\return The Z dimension
	unsigned int getDimZ() const;

///\fn inline double getCellSizeX() const
///\brief Returns the cell size (span) along the X axis
///\return The cell size along the X axis
	inline double getCellSizeX() const;
///\fn inline double getCellSizeY() const
///\brief Returns the cell size (span) along the Y axis
///\return The cell size along the Y axis
	inline double getCellSizeY() const;
///\fn inline double getCellSizeZ() const
///\brief Returns the cell size (span) along the Z axis
///\return The cell size along the Z axis
	inline double getCellSizeZ() const;

protected:
	VolumeSource* m_Source;
};

double SourceManager::getCellSizeX() const
{
	return (getMaxX()-getMinX())/(double)(getDimX()-1);
}

double SourceManager::getCellSizeY() const
{
	return (getMaxY()-getMinY())/(double)(getDimY()-1);
}

double SourceManager::getCellSizeZ() const
{
	return (getMaxZ()-getMinZ())/(double)(getDimZ()-1);
}

#endif // !defined(AFX_SOURCEMANAGER_H__41217F7A_A50F_4B7D_A401_ACDE8120A5DF__INCLUDED_)
