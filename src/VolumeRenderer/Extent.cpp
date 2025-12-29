/*
  Copyright 2002-2003 The University of Texas at Austin

        Authors: Anthony Thane <thanea@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeLibrary.

  VolumeLibrary is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeLibrary is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// Extent.cpp: implementation of the Extent class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRenderer/Extent.h>

using namespace OpenGLVolumeRendering;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Extent::Extent() { setExtents(0, 0, 0, 1, 1, 1); }

Extent::Extent(double xMin, double yMin, double zMin, double xMax,
               double yMax, double zMax) {
  setExtents(xMin, yMin, zMin, xMax, yMax, zMax);
}

Extent::~Extent() {}

void Extent::setExtents(double minX, double minY, double minZ, double maxX,
                        double maxY, double maxZ) {
  m_MinX = minX;
  m_MaxX = maxX;
  m_MinY = minY;
  m_MaxY = maxY;
  m_MinZ = minZ;
  m_MaxZ = maxZ;
}
