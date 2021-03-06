/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory.h>
#include <sys/types.h>
#include <sys/stat.h>

#define _LITTLE_ENDIAN 1

#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))
#define max2(x, y) ((x>y) ? (x):(y))
#define min2(x, y) ((x<y) ? (x):(y))
#define PIE           3.1415926f

namespace SecStruct {

static int XDIM,YDIM,ZDIM;
static float maxraw;
static float minraw;

static float minext[3], maxext[3];
static int nverts, ncells;
static unsigned int dim[3];
static float orig[3], span[3];

void swap_buffer(char *buffer, int count, int typesize);
void write_rawiv_float(float*, FILE*);



void read_data(int *xd, int *yd, int *zd, float **data, 
	       float *span_t, float *orig_t, const char *input_name)
{
  float c_float;
  unsigned char c_unchar;
  unsigned short c_unshort;
  int i,j,k;
  float *dataset;
  
  struct stat filestat;
  size_t size[3];
  int datatype=0;
  int found;
  FILE *fp;
  
  size_t fread_return = 0;

  

  if ((fp=fopen(input_name, "rb"))==NULL){
    printf("read error...\n");
    exit(0);
  }
  stat(input_name, &filestat);

  /* reading RAWIV header */
  fread_return = fread(minext, sizeof(float), 3, fp);
  fread_return = fread(maxext, sizeof(float), 3, fp);
  fread_return = fread(&nverts, sizeof(int), 1, fp);
  fread_return = fread(&ncells, sizeof(int), 1, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
#endif  

  size[0] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
    nverts * sizeof(unsigned char);
  size[1] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
    nverts * sizeof(unsigned short);
  size[2] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
    nverts * sizeof(float);
  
  found = 0;
  for (i = 0; i < 3; i++)
    if (size[i] == (unsigned int)filestat.st_size)
      {
	if (found == 0)
	  {
	    datatype = i;
	    found = 1;
	  }
      }
  if (found == 0)
    {
      printf("Corrupted file or unsupported dataset type\n");
      exit(5);
    }
  
 
  fread_return = fread(dim, sizeof(unsigned int), 3, fp);
  fread_return = fread(orig, sizeof(float), 3, fp);
  fread_return = fread(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif 

  span_t[0] = span[0];
  span_t[1] = span[1];
  span_t[2] = span[2];
  orig_t[0] = minext[0]; //orig[0];
  orig_t[1] = minext[1]; //orig[1];
  orig_t[2] = minext[2]; //orig[2];
  
  XDIM = dim[0];
  YDIM = dim[1];
  ZDIM = dim[2];
  dataset = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  
  maxraw = -99999999.f;
  minraw = 99999999.f;
  
  if (datatype == 0) {
    printf("data type: unsigned char \n");
    for (i=0; i<ZDIM; i++)
      for (j=0; j<YDIM; j++)
	for (k=0; k<XDIM; k++) {
	  fread_return = fread(&c_unchar, sizeof(unsigned char), 1, fp);
	  dataset[IndexVect(k,j,i)]=(float)c_unchar;
	  
	  if (c_unchar > maxraw)
	    maxraw = c_unchar;
	  if (c_unchar < minraw)
	    minraw = c_unchar;
	}
  }
  else if (datatype == 1) {
    printf("data type: unsigned short \n");
    for (i=0; i<ZDIM; i++)
      for (j=0; j<YDIM; j++)
	for (k=0; k<XDIM; k++) {
	  fread_return = fread(&c_unshort, sizeof(unsigned short), 1, fp);
#ifdef _LITTLE_ENDIAN
	  swap_buffer((char *)&c_unshort, 1, sizeof(unsigned short));
#endif 
	  dataset[IndexVect(k,j,i)]=(float)c_unshort;
	  
	  if (c_unshort > maxraw)
	    maxraw = c_unshort;
	  if (c_unshort < minraw)
	    minraw = c_unshort;
	}
  }
  else if (datatype == 2) {
    printf("data type: float \n");
    for (i=0; i<ZDIM; i++) 
      for (j=0; j<YDIM; j++)
	for (k=0; k<XDIM; k++) {
	  fread_return = fread(&c_float, sizeof(float), 1, fp);
#ifdef _LITTLE_ENDIAN
	  swap_buffer((char *)&c_float, 1, sizeof(float));
#endif 
	  dataset[IndexVect(k,j,i)]=c_float;
	  
	  if (c_float > maxraw)
	    maxraw = c_float;
	  if (c_float < minraw)
	    minraw = c_float;
	}
  }
  
  else {
    printf("error\n");
    fclose(fp);
    exit(1);
  }
  
  fclose(fp);
  
  for (i=0; i<ZDIM; i++) 
    for (j=0; j<YDIM; j++)
      for (k=0; k<XDIM; k++)
	dataset[IndexVect(k,j,i)] = 255*(dataset[IndexVect(k,j,i)] - 
					 minraw)/(maxraw-minraw); 

  printf("minimum = %f,   maximum = %f \n",minraw,maxraw);
    
  /*
  float dx,dy,dz;
  float dist,dist1,dist2;
  dx = 0.266079;
  dy = -0.955360;
  dz = -0.128408;
  for (k=0; k<ZDIM; k++) 
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
        dataset[IndexVect(i,j,k)] = 0;
        dist = sqrt((i-XDIM/2)*(i-XDIM/2)+(j-YDIM/2)*(j-YDIM/2)+(k-ZDIM/2)*(k-ZDIM/2));
        dist1 = fabs((i-XDIM/2)*dx+(j-YDIM/2)*dy+(k-ZDIM/2)*dz);
        dist2 = sqrt(dist*dist-dist1*dist1);
        if (dist2 <= 15)
          dataset[IndexVect(i,j,k)] = 255*exp(-0.05*dist1*dist1);
        else
          dataset[IndexVect(i,j,k)] = 255*exp(-0.05*(dist1*dist1+(dist2-15)*(dist2-15)));
      }
  */


  *xd = XDIM;
  *yd = YDIM;
  *zd = ZDIM;
  *data = dataset;
  
}



void write_rawiv_char(unsigned char *result,FILE* fp)
{
  int i, j, k;
  unsigned char c_char;
  
  size_t fwrite_return = 0;
  

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif 
  
  fwrite_return = fwrite(minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif 

  for (k=0; k<ZDIM; k++) 
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {

	c_char = result[IndexVect(i,j,k)];
	  
#ifdef _LITTLE_ENDIAN
	  swap_buffer((char *)&c_char, 1, sizeof(unsigned char));
#endif 
	  fwrite_return = fwrite(&c_char, sizeof(unsigned char), 1, fp);
#ifdef _LITTLE_ENDIAN
	  swap_buffer((char *)&c_char, 1, sizeof(unsigned char));
#endif 
      }
  
  fclose(fp);
}

void write_rawiv_short(unsigned short *result,FILE* fp)
{
  int i, j, k;
  unsigned short c_short;
  
  size_t fwrite_return = 0;
  

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif 
  
  fwrite_return = fwrite(minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif 

  for (k=0; k<ZDIM; k++) 
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {

	c_short = result[IndexVect(i,j,k)];
	  
#ifdef _LITTLE_ENDIAN
	swap_buffer((char *)&c_short, 1, sizeof(unsigned short));
#endif 
	fwrite_return = fwrite(&c_short, sizeof(unsigned short), 1, fp);
#ifdef _LITTLE_ENDIAN
	swap_buffer((char *)&c_short, 1, sizeof(unsigned short));
#endif 
      }


  fclose(fp);
}

void write_rawiv_float(float *result,FILE* fp)
{
  int i, j, k;
  float c_float;
  
  size_t fwrite_return = 0;
  

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif 
  
  fwrite_return = fwrite(minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif 

  for (k=0; k<ZDIM; k++) 
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {

	c_float = result[IndexVect(i,j,k)];
	
#ifdef _LITTLE_ENDIAN
	swap_buffer((char *)&c_float, 1, sizeof(float));
#endif 
	fwrite_return = fwrite(&c_float, sizeof(float), 1, fp);
#ifdef _LITTLE_ENDIAN
	swap_buffer((char *)&c_float, 1, sizeof(float));
#endif 
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

};
