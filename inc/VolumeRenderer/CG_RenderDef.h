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

#ifndef __VOLUME_LIBRARY_CG_RENDER_DEF_H__
#define __VOLUME_LIBRARY_CG_RENDER_DEF_H__

// point light position in eye coordinate
// (this means camera position is at (0,0,0))
// by default, light is at same position with camera
#define DEF_LIGHT_POS_EYECOOR_X 0.0
#define DEF_LIGHT_POS_EYECOOR_Y 0.0
#define DEF_LIGHT_POS_EYECOOR_Z 0.0

// point light color
#define DEF_LIGHT_COLOR_R 1.0
#define DEF_LIGHT_COLOR_G 1.0
#define DEF_LIGHT_COLOR_B 1.0

#endif // __VOLUME_LIBRARY_CG_RENDER_DEF_H__
