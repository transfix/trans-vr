/*
  Copyright 2006-2007 The University of Texas at Austin

        Authors: Dr. Xu Guo Liang <xuguo@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of HLevelSet.

  HLevelSet is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  HLevelSet is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef MISC_H
#define MISC_H

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <stdlib.h>
/*
#ifdef __cplusplus
extern "C" {
#endif
*/

// template <typename T> T Distance_Two_Points3D(T* P1, T* P2);
template <typename T> T Distance_Two_points3D(T *P1, T *P2) {
  T result;

  result = sqrt((P1[0] - P2[0]) * (P1[0] - P2[0]) +
                (P1[1] - P2[1]) * (P1[1] - P2[1]) +
                (P1[2] - P2[2]) * (P1[2] - P2[2]));

  return (result);
}

/*----------------------------------------------------------------------------
Template of Max_Of_Two and Min_Of_Two which return the biggest and smallest of
three
-----------------------------------------------------------------------------*/

template <class T> T Max_Of_Two(T a, T b) {
  if (a >= b)
    return a;
  else
    return b;
}

template <class T> T Min_Of_Two(T a, T b) {
  if (a <= b)
    return a;
  else
    return b;
}

/*----------------------------------------------------------------------------
Template of Max_Of_Three, Min_Of_Three, Mid_Of_Three  which return the
biggest, smallest and middle of three
-----------------------------------------------------------------------------*/
template <class T> T Max_Of_Three(T a, T b, T c) {
  if ((a <= b) && (c <= b))
    return b;
  if ((a <= c) && (b <= c))
    return c;
  if ((b <= a) && (c <= a))
    return a;
}

template <class T> T Min_Of_Three(T a, T b, T c) {
  if ((a >= b) && (c >= b))
    return b;
  if ((a >= c) && (b >= c))
    return c;
  if ((b >= a) && (c >= a))
    return a;
}

template <class T> T Mid_Of_Three(T a, T b, T c) {
  if ((a >= b) && (b >= c))
    return b;
  if ((a >= c) && (c >= b))
    return c;
  if ((b >= a) && (a >= c))
    return a;
}

/*
#ifdef __cplusplus
}
#endif
*/
#endif
