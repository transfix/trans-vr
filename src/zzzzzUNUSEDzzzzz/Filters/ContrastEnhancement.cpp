/*
  Copyright 2002-2005 The University of Texas at Austin
  
	Authors: Zeyun Yu <zeyun@cs.utexas.edu>
	         Jose Rivera <transfix@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdlib.h>

#include <Filters/ContrastEnhancement.h>

#ifndef max
#define max(x, y) ((x>y) ? (x):(y))
#endif
#ifndef min
#define min(x, y) ((x<y) ? (x):(y))
#endif

#define IndexVect(i,j,k) ((k)*xdim*ydim + (j)*xdim + (i))

int Filters::ContrastEnhancementSlice(unsigned int xdim, unsigned int ydim, float resistor, float *paramin, float *paramax, float *imgavg, int k)
{
  float *tmpmin, *tmpmax;
  float *lcmin, *lcmax;
  int i, j;

  tmpmin = (float*)malloc(sizeof(float)*xdim*ydim);
  tmpmax = (float*)malloc(sizeof(float)*xdim*ydim);
  lcmin = (float*)malloc(sizeof(float)*xdim*ydim);
  lcmax = (float*)malloc(sizeof(float)*xdim*ydim);
  
  for (j=0; j<(int)ydim; j++)
    for (i=0; i<(int)xdim; i++) {
      lcmin[j*xdim+i] = paramin[IndexVect(i,j,k)];
      lcmax[j*xdim+i] = paramax[IndexVect(i,j,k)];
      tmpmin[j*xdim+i] = paramin[IndexVect(i,j,k)];
      tmpmax[j*xdim+i] = paramax[IndexVect(i,j,k)];
    }
  

  /* Bottom-up */
  for (i=1; i<(int)xdim; i++) {
     imgavg[IndexVect(i,0,k)] += resistor*
                       (imgavg[IndexVect(i-1,0,k)]-imgavg[IndexVect(i,0,k)]);
     if (tmpmin[i-1] < tmpmin[i])
       tmpmin[i] += resistor*(tmpmin[i-1]-tmpmin[i]);
     if (tmpmax[i-1] > tmpmax[i])
       tmpmax[i] += resistor*(tmpmax[i-1]-tmpmax[i]);
 
   }
   
  for (i=xdim-2; i>=0; i--) {
    imgavg[IndexVect(i,0,k)] += resistor*
                       (imgavg[IndexVect(i+1,0,k)]-imgavg[IndexVect(i,0,k)]);
    if (tmpmin[i+1] < tmpmin[i])
       tmpmin[i] += resistor*(tmpmin[i+1]-tmpmin[i]);
    if (tmpmax[i+1] > tmpmax[i])
       tmpmax[i] += resistor*(tmpmax[i+1]-tmpmax[i]);
 
  }
  
  for (j=1; j<(int)ydim; j++) {

     for (i=0; i<(int)xdim; i++) {
       imgavg[IndexVect(i,j,k)] += resistor*
	               (imgavg[IndexVect(i,j-1,k)]-imgavg[IndexVect(i,j,k)]);
       if (tmpmin[(j-1)*xdim+i] < tmpmin[j*xdim+i])
	 tmpmin[j*xdim+i] += resistor*(tmpmin[(j-1)*xdim+i]-tmpmin[j*xdim+i]);
       if (tmpmax[(j-1)*xdim+i] > tmpmax[j*xdim+i])
	 tmpmax[j*xdim+i] += resistor*(tmpmax[(j-1)*xdim+i]-tmpmax[j*xdim+i]);
  
     }

     for (i=1; i<(int)xdim; i++) {
       imgavg[IndexVect(i,j,k)] += resistor*
	               (imgavg[IndexVect(i-1,j,k)]-imgavg[IndexVect(i,j,k)]);
       if (tmpmin[j*xdim+i-1] < tmpmin[j*xdim+i])
	 tmpmin[j*xdim+i] += resistor*(tmpmin[j*xdim+i-1]-tmpmin[j*xdim+i]);
       if (tmpmax[j*xdim+i-1] > tmpmax[j*xdim+i])
	 tmpmax[j*xdim+i] += resistor*(tmpmax[j*xdim+i-1]-tmpmax[j*xdim+i]);
 
     }
    
     for (i=xdim-2; i>=0; i--) {
       imgavg[IndexVect(i,j,k)] += resistor*
	               (imgavg[IndexVect(i+1,j,k)]-imgavg[IndexVect(i,j,k)]);
       if (tmpmin[j*xdim+i+1] < tmpmin[j*xdim+i])
	 tmpmin[j*xdim+i] += resistor*(tmpmin[j*xdim+i+1]-tmpmin[j*xdim+i]);
       if (tmpmax[j*xdim+i+1] > tmpmax[j*xdim+i])
	 tmpmax[j*xdim+i] += resistor*(tmpmax[j*xdim+i+1]-tmpmax[j*xdim+i]);
  
     }
  }



  /* Top-down */
  j=ydim-1;
  for (i=1; i<(int)xdim; i++) {
     imgavg[IndexVect(i,j,k)] += resistor*
                       (imgavg[IndexVect(i-1,j,k)]-imgavg[IndexVect(i,j,k)]);
     if (lcmin[j*xdim+i-1] < lcmin[j*xdim+i])
       lcmin[j*xdim+i] += resistor*(lcmin[j*xdim+i-1]-lcmin[j*xdim+i]);
     if (lcmax[j*xdim+i-1] > lcmax[j*xdim+i])
       lcmax[j*xdim+i] += resistor*(lcmax[j*xdim+i-1]-lcmax[j*xdim+i]);

   }
   
  for (i=xdim-2; i>=0; i--) {
    imgavg[IndexVect(i,j,k)] += resistor*
                       (imgavg[IndexVect(i+1,j,k)]-imgavg[IndexVect(i,j,k)]);
    if (lcmin[j*xdim+i+1] < lcmin[j*xdim+i])
       lcmin[j*xdim+i] += resistor*(lcmin[j*xdim+i+1]-lcmin[j*xdim+i]);
     if (lcmax[j*xdim+i+1] > lcmax[j*xdim+i])
       lcmax[j*xdim+i] += resistor*(lcmax[j*xdim+i+1]-lcmax[j*xdim+i]);
  
  }
  
  for (j=ydim-2; j>=0; j--) {

     for (i=0; i<(int)xdim; i++) {
       imgavg[IndexVect(i,j,k)] += resistor*
	               (imgavg[IndexVect(i,j+1,k)]-imgavg[IndexVect(i,j,k)]);
       if (lcmin[(j+1)*xdim+i] < lcmin[j*xdim+i])
	 lcmin[j*xdim+i] += resistor*(lcmin[(j+1)*xdim+i]-lcmin[j*xdim+i]);
       if (lcmax[(j+1)*xdim+i] > lcmax[j*xdim+i])
	 lcmax[j*xdim+i] += resistor*(lcmax[(j+1)*xdim+i]-lcmax[j*xdim+i]);

     }

     for (i=1; i<(int)xdim; i++) {
       imgavg[IndexVect(i,j,k)] += resistor*
	               (imgavg[IndexVect(i-1,j,k)]-imgavg[IndexVect(i,j,k)]);
       if (lcmin[j*xdim+i-1] < lcmin[j*xdim+i])
	 lcmin[j*xdim+i] += resistor*(lcmin[j*xdim+i-1]-lcmin[j*xdim+i]);
       if (lcmax[j*xdim+i-1] > lcmax[j*xdim+i])
	 lcmax[j*xdim+i] += resistor*(lcmax[j*xdim+i-1]-lcmax[j*xdim+i]);
   
     }
    
     for (i=xdim-2; i>=0; i--) {
       imgavg[IndexVect(i,j,k)] += resistor*
	               (imgavg[IndexVect(i+1,j,k)]-imgavg[IndexVect(i,j,k)]);
       if (lcmin[j*xdim+i+1] < lcmin[j*xdim+i])
	 lcmin[j*xdim+i] += resistor*(lcmin[j*xdim+i+1]-lcmin[j*xdim+i]);
       if (lcmax[j*xdim+i+1] > lcmax[j*xdim+i])
	 lcmax[j*xdim+i] += resistor*(lcmax[j*xdim+i+1]-lcmax[j*xdim+i]);
  
     }
  }

  
  for (j=0; j<(int)ydim; j++)
    for (i=0; i<(int)xdim; i++) {
      paramin[IndexVect(i,j,k)] = min(lcmin[j*xdim+i],tmpmin[j*xdim+i]);
      paramax[IndexVect(i,j,k)] = max(lcmax[j*xdim+i],tmpmax[j*xdim+i]);
    }

  free(tmpmin);
  free(tmpmax);
  free(lcmin);
  free(lcmax);
 
  return 0;
}
