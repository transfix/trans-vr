/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/**************************************
 * Zeyun Yu (zeyun@cs.utexas.edu)    *
 * Department of Computer Science    *
 * University of Texas at Austin     *
 **************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <VolMagick/VolMagick.h>

#include <TightCocone/segment.h>

#define _LITTLE_ENDIAN 1

namespace TightCocone
{

FILE *fp;
  //float maxraw;
  //float minraw;

float minext[3], maxext[3];
int nverts, ncells;
unsigned int dim[3];
float orig[3], span[3];


void swap_buffer(char *buffer, int count, int typesize);


void read_data(const VolMagick::Volume& vol/*char *input_name*/)
{
  //float c_float;
  //u_char c_unchar;
  //u_short c_unshort;
  int i,j,k;
  
  //struct stat filestat;
  //size_t size[3];
  //int datatype;
  //int found;
 
  
  XDIM = vol.XDim();
  YDIM = vol.YDim();
  ZDIM = vol.ZDim();
 
  //dataset->data = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  //dataset->tdata = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  dataset->data.reset(new float[XDIM*YDIM*ZDIM]);
  dataset->tdata.reset(new float[XDIM*YDIM*ZDIM]);

  /* reading the data */
  //maxraw = -99999999;
  //minraw = 99999999;

  VolMagick::Volume newvol(vol);
  newvol.map(0.0,255.0);

  for (i=0; i<ZDIM; i++) {
    for (j=0; j<YDIM; j++)
      for (k=0; k<XDIM; k++) {
	dataset->data[IndexVect(k,j,i)]=newvol(k,j,i);
	dataset->tdata[IndexVect(k,j,i)] = MAX_TIME;
      }
  }
  
  //maxraw = vol.max();
  //minraw = vol.min();

  //printf("minimum = %f,   maximum = %f \n",minraw,maxraw);
  
}



void write_data(char *out_seg)
{
  int i, j, k;
  int num;
  
  
  if ((fp=fopen(out_seg, "w")) == NULL){
    printf("write error....\n");
    exit(0);
  }
  
  num = 0;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
	if (bin_img[IndexVect(i,j,k)] == 0)
	  num++;
      }
  
  fprintf(fp,"%d \n",num);

  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
	if (bin_img[IndexVect(i,j,k)] == 0)
	  fprintf(fp,"%d %d %d \n",i,j,k);
      }

  fclose(fp);

}


void swap_buffer(char *buffer, int count, int typesize)
{
  char sbuf[4];
  int i;
  int temp = 1;
  unsigned char* chartempf = (unsigned char*) &temp;
  if(chartempf[0] > '\0') {
  
	// swapping isn't necessary on single byte data
	if (typesize == 1)
		return;
  
  
	for (i=0; i < count; i++)
    {
		memcpy(sbuf, buffer+(i*typesize), typesize);
      
		switch (typesize)
		{
			case 2:
			{
				buffer[i*typesize] = sbuf[1];
				buffer[i*typesize+1] = sbuf[0];
				break;
			}
			case 4:
			{
				buffer[i*typesize] = sbuf[3];
				buffer[i*typesize+1] = sbuf[2];
				buffer[i*typesize+2] = sbuf[1];
				buffer[i*typesize+3] = sbuf[0];
				break;
			}
			default:
				break;
		}
    }

  }
}

}
