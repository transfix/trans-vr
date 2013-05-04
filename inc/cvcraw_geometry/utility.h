/*
  Copyright 2008-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __CVCRAW_GEOMETRY_UTILITY_H__
#define __CVCRAW_GEOMETRY_UTILITY_H__

/*
 * Some typical vector math utility functions:
 *  cross, dot, normalize
 *
 * Change Log:
 * 04/02/2010 - Moved this code into it's own header from cvcraw_geometry.h
 */

#include <cmath>

namespace cvcraw_geometry
{
  namespace utility
  {
    template <class Vector_3>
      void cross(Vector_3& dest, const Vector_3& v1, const Vector_3& v2)
      {
	dest[0] = v1[1]*v2[2] - v1[2]*v2[1];
	dest[1] = v1[2]*v2[0] - v1[0]*v2[2];
	dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
      }
    
    template <class Vector_3>
      double dot(const Vector_3& v1, const Vector_3& v2)
      {
	return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
      }
    
    template <class Vector_3>
      void normalize(Vector_3& v)
      {
	double len = sqrt(static_cast<double>(v[0] * v[0] +
					      v[1] * v[1] +
					      v[2] * v[2]));
	if (len!=0.0)
	  {
	    v[0]/=len;
	    v[1]/=len;
	    v[2]/=len;
	  }
	else {
	  v[0] = 1.0;
	}
      }
  }
}

#endif
