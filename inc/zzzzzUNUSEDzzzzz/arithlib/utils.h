/*
  Copyright 2000-2003 The University of Texas at Austin

	Authors: Xiaoyu Zhang 2000-2002 <xiaoyu@ices.utexas.edu>
					 John Wiggins 2003 <prok@cs.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of iotree.

  iotree is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  iotree is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef C2C_UTILS_H
#define C2C_UTILS_H
#include <assert.h>

#ifdef WIN32
#include <arithlib/unitypes.h>
#else
#include <unistd.h>
#endif

float maximum(float * array, int);

float minimum(float * array, int);

float MyPower(int exp);

int Comp_Factor(float f);

int Comp_Bits(int maxv);

/**
 * Quantize a float number x to a int
 * @param x a float number in [-1.0, 1.0]
 * @param pow quantized int is in [0, pow)
 */
int fquantize(float x, u_int pow);

/**
 * Unquantize a int in [0, pow) to a float in [-1.0, 1.0]
 */
float unfquantize(int n, u_int pow);

#endif

