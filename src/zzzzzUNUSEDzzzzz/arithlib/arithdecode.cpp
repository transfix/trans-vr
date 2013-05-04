/*
  Copyright 2000-2003 The University of Texas at Austin

	Authors: Xiaoyu Zhang 2000-2002 <xiaoyu@ices.utexas.edu>
					 John Wiggins 2003 <prok@cs.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of iotree.

  iotree is free software; you can redistribute it and/or modify

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
#include <math.h>
#include <iostream>
#include <arithlib/encode.h>
#include <arithlib/utils.h>

using namespace std;

#define DEBUG_ADNo

static inline int SIGN(float x) {
    if (x > 0) return 1;
    if (x < 0) return 2;
    return 0;
}

static int zero_count = 0;

bool decode_vals(BitBuffer *bbuf, float *vals, int nv, float isoval)
{
    int i;
#ifdef ZP_CODEC
    bbuf->zp_decode();
#else
    bbuf->arith_decode();
#endif

    BIT *signs = new BIT[nv];

    for (i = 0; i < nv; i++) {
        signs[i] = 1+bbuf->get_bits(1);
    }

    // get the vals
    float favg = bbuf->get_a_float();
    float frad = bbuf->get_a_float();

    // get the first float
    vals[0] = bbuf->get_a_float(FLOAT_RES);
#ifdef DEBUG_AD
    cout << "avg = " << favg << ", rad = " << frad << endl;
    cout << "average = " << favg << endl;
    cout << "vals[0] = " << vals[0] << endl;
#endif

    float *delta = new float[nv-1];
    float delta_pow = MyPower(DELTA_RES);
    for (i = 0; i < nv-1; i++) {
        int quan = bbuf->get_bits(DELTA_RES);
        delta[i] = unfquantize(quan, (int)delta_pow);
    }
    for (i = 1; i < nv; i++) {
        vals[i] = vals[i-1] + delta[i-1];
    }

    float shift = 0.5/delta_pow;
    for (i = 0; i < nv; i++) {
        vals[i] = favg + 2*vals[i]*frad;
        if (signs[i] != SIGN(vals[i])) {
#ifdef _DEBUG
            printf("-------sign flipped: %f\n", vals[i]);
#endif
            if (signs[i] == 1) vals[i] = shift*2*frad;
            else if (signs[i] == 2) vals[i] = -shift*2*frad;
            else vals[i] = 0.0;
        }

        vals[i] += isoval;
#ifdef DEBUG_AD
        cout << "vals[" << i << "] = " << vals[i] << endl;
#endif
    }

    delete[] signs;
    delete[] delta;
    return true;
}

bool  decode_vals(BitBuffer *bbuf, u_short *vals, int nv, float isoval)
{
    int i;
#ifdef ZP_CODEC
    bbuf->zp_decode();
#else
    bbuf->arith_decode();
#endif    
    int res = USHORT_BIT;

    // get the first u_short
    vals[0] = bbuf->get_bits(res);

    int delta;
    for (i = 1; i < nv; i++) {
        delta = bbuf->get_bits(res);
        if (vals[i-1]+delta >= USHORT_POW) delta -= USHORT_POW;
        vals[i] = vals[i-1]+delta;
    }
    return true;
}

bool  decode_vals(BitBuffer *bbuf, u_char *vals, int nv, float isoval)
{
    int i;
#ifdef ZP_CODEC
    bbuf->zp_decode();
#else
    bbuf->arith_decode();
#endif
    
    int res = UCHAR_BIT;


    // get the first u_char
    vals[0] = bbuf->get_bits(res);

    int delta;
    for (i = 1; i < nv; i++) {
        delta = bbuf->get_bits(res);
        if (vals[i-1] + delta >= UCHAR_POW) delta -= UCHAR_POW;
        vals[i] = vals[i-1]+delta;
    }

/*  for(i = 0; i < nv; i++) {
      vals[i] = bbuf->get_bits(res);
  }
*/
    return true;
}

