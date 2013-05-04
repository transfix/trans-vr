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
//
// SourceManager.cpp: implementation of the SourceManager class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/SourceManager.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SourceManager::SourceManager()
{
	m_Source = 0;
}

SourceManager::~SourceManager()
{
	delete m_Source;
}

void SourceManager::setSource(VolumeSource* source)
{
	delete m_Source;
	m_Source = source;
}

void SourceManager::resetSource()
{
	delete m_Source;
	m_Source = 0;
}

VolumeSource* SourceManager::getSource()
{
	return m_Source;
}

bool SourceManager::hasSource() const
{
	return (m_Source!=0);
}

unsigned int SourceManager::getNumVerts() const
{
	if (m_Source) {
		return m_Source->getNumVerts();
	}
	else {
		return 0.0;
	}
}

unsigned int SourceManager::getNumCells() const
{
	if (m_Source) {
		return m_Source->getNumCells();
	}
	else {
		return 0.0;
	}
}

unsigned int SourceManager::getNumVars() const
{
	if (m_Source) {
		return m_Source->getNumVars();
	}
	else {
		return 0.0;
	}
}

unsigned int SourceManager::getNumTimeSteps() const
{
	if (m_Source) {
		return m_Source->getNumTimeSteps();
	}
	else {
		return 0.0;
	}
}

double SourceManager::getMinX() const
{
	if (m_Source) {
		return m_Source->getMinX();
	}
	else {
		return 0.0;
	}
}

double SourceManager::getMinY() const
{
	if (m_Source) {
		return m_Source->getMinY();
	}
	else {
		return 0.0;
	}
}

double SourceManager::getMinZ() const
{
	if (m_Source) {
		return m_Source->getMinZ();
	}
	else {
		return 0.0;
	}
}

double SourceManager::getMaxX() const
{
	if (m_Source) {
		return m_Source->getMaxX();
	}
	else {
		return 1.0;
	}
}

double SourceManager::getMaxY() const
{
	if (m_Source) {
		return m_Source->getMaxY();
	}
	else {
		return 1.0;
	}
}

double SourceManager::getMaxZ() const
{
	if (m_Source) {
		return m_Source->getMaxZ();
	}
	else {
		return 1.0;
	}
}

unsigned int SourceManager::getDimX() const
{
	if (m_Source) {
		return m_Source->getDimX();
	}
	else {
		return 0;
	}
}

unsigned int SourceManager::getDimY() const
{
	if (m_Source) {
		return m_Source->getDimY();
	}
	else {
		return 0;
	}
}

unsigned int SourceManager::getDimZ() const
{
	if (m_Source) {
		return m_Source->getDimZ();
	}
	else {
		return 0;
	}
}


