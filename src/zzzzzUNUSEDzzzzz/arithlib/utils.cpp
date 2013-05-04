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
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <arithlib/utils.h>

float
MyPower(int exp)
{
   float tmp = 1;

   for (int k=0; k<exp; k++)
       tmp *= 2;

   return tmp;
}

float maximum(float * array, int n)
{
        assert(n >= 1);
        float max = array[0];
        for(int i = 1; i < n; i++) {
                if(array[i] > max) max = array[i];
        }
        return max;
}

float minimum(float* array, int n)
{
        assert(n >= 1);
        float min = array[0];
        for(int i = 1; i < n; i++) {
                if(array[i] < min) min = array[i];
        }
        return min;
}

int Comp_Factor(float f)
{
    int  ret=0;

    if (f<0.5)
       return 0;

    while(f>=0.5) {
        f = (float)(f/2.0);
        ret++;
    }

    return ret;
}             

int
Comp_Bits(int maxv)
{
   int tmp=2, ret=1;

   assert(maxv >= 0);
   if (maxv<0) {
      fprintf(stderr, "Comp_Bits(): argument should be positive\n");
      exit(1);
   }

   while(tmp<=maxv) {
       tmp = 2*tmp;
       ret++;
   }

   return ret;
}
  
int fquantize(float x, u_int pow)
{
	//cout << "x = " << x << endl;
	//assert(fabs(x) <= 1.0);
	if(x >= 1.0) {
		//printf("too large x: %f \n", x);
		x = 1.0;
	} else if(x <= -1.0){
		//printf("too small x: %f \n", x);
		x = -1.0;
	}
	//return (int)((x+1)*(pow-1)/2);
	float y = (x+1)/2;
	if(floor(y*pow) >= pow - 1) return pow-1;
	return (int)floor(y*pow) + (((y*pow - floor(y*pow))>0.5)? 1:0);
}

float unfquantize(int n, u_int pow)
{
#ifdef _DEBUG
    assert(n < (int)pow);
#endif
	return (float)(2*(n+0.5)/pow - 1.0);
}

