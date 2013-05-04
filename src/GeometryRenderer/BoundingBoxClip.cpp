/*
  Copyright 2012 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of libCVC.

  libCVC is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  libCVC is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: BoundingBoxClip.cpp 5692 2012-06-01 22:39:25Z transfix $ */

#include <GeometryRenderer/BoundingBoxClip.h>

#include <glew/glew.h>

namespace CVC_NAMESPACE
{
  void BoundingBoxClip::operator()()
  {
    CVC::BoundingBox bbox = boundingBox();
    
    double plane0[] = { 0.0, 0.0, -1.0, bbox.maxz };
    glClipPlane(GL_CLIP_PLANE0, plane0);
    glEnable(GL_CLIP_PLANE0);

    double plane1[] = { 0.0, 0.0, 1.0, -bbox.minz };
    glClipPlane(GL_CLIP_PLANE1, plane1);
    glEnable(GL_CLIP_PLANE1);

    double plane2[] = { 0.0, -1.0, 0.0, bbox.maxy };
    glClipPlane(GL_CLIP_PLANE2, plane2);
    glEnable(GL_CLIP_PLANE2);

    double plane3[] = { 0.0, 1.0, 0.0, -bbox.miny };
    glClipPlane(GL_CLIP_PLANE3, plane3);
    glEnable(GL_CLIP_PLANE3);

    double plane4[] = { -1.0, 0.0, 0.0, bbox.maxx };
    glClipPlane(GL_CLIP_PLANE4, plane4);
    glEnable(GL_CLIP_PLANE4);

    double plane5[] = { 1.0, 0.0, 0.0, -bbox.minx };
    glClipPlane(GL_CLIP_PLANE5, plane5);
    glEnable(GL_CLIP_PLANE5);
  }
}
