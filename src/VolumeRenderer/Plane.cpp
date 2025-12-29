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

// Plane.cpp: implementation of the Plane class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRenderer/Plane.h>
#include <math.h>

using namespace OpenGLVolumeRendering;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Plane::Plane() {}

Plane::Plane(double a, double b, double c, double d) {
  m_A = a;
  m_B = b;
  m_C = c;
  m_D = d;
}

Plane::~Plane() {}

double Plane::signedDistance(double x, double y, double z) const {
  return m_A * x + m_B * y + m_C * z - m_D;
}

void Plane::normalizeNormal() {
  double length = sqrt(m_A * m_A + m_B * m_B + m_C * m_C);
  m_A /= length;
  m_B /= length;
  m_C /= length;
  m_D /= length;
}
