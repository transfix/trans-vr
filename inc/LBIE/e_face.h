/*
  Copyright 2006 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of LBIE.

  LBIE is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  LBIE is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __E_FACE_H__
#define __E_FACE_H__

#include <boost/array.hpp>

namespace LBIE
{

typedef struct _edge_face {
  int faceidx;
  //int orientation[4];
  boost::array<int,4> orientation;
  int facebit;
} edge_face;

}

#endif

