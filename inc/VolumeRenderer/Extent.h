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

// Extent.h: interface for the Extent class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(                                                                \
    AFX_OPENGLEXTENT_H__DAAEF2B3_F7FE_48F8_88CF_41B8BF38BA02__INCLUDED_)
#define AFX_OPENGLEXTENT_H__DAAEF2B3_F7FE_48F8_88CF_41B8BF38BA02__INCLUDED_

namespace OpenGLVolumeRendering {

/** encapsulates a 3d min and max */
class Extent {
public:
  Extent();
  Extent(double xMin, double yMin, double zMin, double xMax, double yMax,
         double zMax);
  ~Extent();

  void setExtents(double xMin, double yMin, double zMin, double xMax,
                  double yMax, double zMax);

  double m_MinX;
  double m_MinY;
  double m_MinZ;
  double m_MaxX;
  double m_MaxY;
  double m_MaxZ;
};

}; // namespace OpenGLVolumeRendering

#endif // !defined(AFX_OPENGLEXTENT_H__DAAEF2B3_F7FE_48F8_88CF_41B8BF38BA02__INCLUDED_)
