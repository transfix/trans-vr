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

static inline int big_endian()
{
  long one=1;
  return !(*((char *)(&one)));
}

#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))
#define max(x, y) ((x>y) ? (x):(y))
#define min(x, y) ((x<y) ? (x):(y))

namespace SegMed {

  static float maxraw;
  static float minraw;
  
  static float minext[3], maxext[3];
  static int nverts, ncells;
  static unsigned int dim[3];
  static float orig[3], span[3];

  void swap_buffer(char *buffer, int count, int typesize);

  void read_data(int *xd, int *yd, int *zd, 
		 float **data, const char *input_name)
  {
    float c_float;
    unsigned char c_unchar;
    unsigned short c_unshort;
    int i,j,k;
    int XDIM,YDIM,ZDIM;
    float *dataset;
  
    struct stat filestat;
    size_t size[3];
    int datatype=0;
    int found;
    FILE *fp;
    
    size_t fread_return = 0;

  

    if ((fp=fopen(input_name, "r"))==NULL){
      printf("read error...\n");
      exit(0);
    }
    stat(input_name, &filestat);

    /* reading RAWIV header */
    fread_return = fread(minext, sizeof(float), 3, fp);
    fread_return = fread(maxext, sizeof(float), 3, fp);
    fread_return = fread(&nverts, sizeof(int), 1, fp);
    fread_return = fread(&ncells, sizeof(int), 1, fp);

	if(!big_endian())
	{
		swap_buffer((char *)minext, 3, sizeof(float));
		swap_buffer((char *)maxext, 3, sizeof(float));
		swap_buffer((char *)&nverts, 1, sizeof(int));
		swap_buffer((char *)&ncells, 1, sizeof(int));
	}

    size[0] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
      nverts * sizeof(unsigned char);
    size[1] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
      nverts * sizeof(unsigned short);
    size[2] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
      nverts * sizeof(float);
  
    found = 0;
    for (i = 0; i < 3; i++)
      if ((long)size[i] == filestat.st_size)
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

	if(!big_endian())
	{
		swap_buffer((char *)dim, 3, sizeof(unsigned int));
		swap_buffer((char *)orig, 3, sizeof(float));
		swap_buffer((char *)span, 3, sizeof(float));
	}

    XDIM = dim[0];
    YDIM = dim[1];
    ZDIM = dim[2];
  
 
    dataset = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  

    /* reading the data */
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
		if(!big_endian())
			swap_buffer((char *)&c_unshort, 1, sizeof(unsigned short));
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
		if(!big_endian())
			swap_buffer((char *)&c_float, 1, sizeof(float));
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

    *xd = XDIM;
    *yd = YDIM;
    *zd = ZDIM;
    *data = dataset;
  
  }



  void write_data(int XDIM, int YDIM, int ZDIM, float *dataset, 
		  unsigned char *result, FILE* fp)
  {
    int i, j, k;
    unsigned char c;
    float c_float;
//    unsigned char c_unchar;

//    unsigned int	MagicNumW=0xBAADBEEF;
//    unsigned int	NumTimeStep, NumVariable;
//    float		MinXYZT[4], MaxXYZT[4];
//    unsigned char	VariableType[100];
//    char		*VariableName[100];
//    int m;
//    unsigned int xdim,ydim,zdim;
//    FILE *fp1;

	size_t fwrite_return = 0;

	if(!big_endian())
	{
		swap_buffer((char *)minext, 3, sizeof(float));
		swap_buffer((char *)maxext, 3, sizeof(float));
		swap_buffer((char *)&nverts, 1, sizeof(int));
		swap_buffer((char *)&ncells, 1, sizeof(int));
		swap_buffer((char *)dim, 3, sizeof(unsigned int));
		swap_buffer((char *)orig, 3, sizeof(float));
		swap_buffer((char *)span, 3, sizeof(float));
	}
  
    fwrite_return = fwrite(minext, sizeof(float), 3, fp);
    fwrite_return = fwrite(maxext, sizeof(float), 3, fp);
    fwrite_return = fwrite(&nverts, sizeof(int), 1, fp);
    fwrite_return = fwrite(&ncells, sizeof(int), 1, fp);
    fwrite_return = fwrite(dim, sizeof(unsigned int), 3, fp);
    fwrite_return = fwrite(orig, sizeof(float), 3, fp);
    fwrite_return = fwrite(span, sizeof(float), 3, fp);

	if(!big_endian())
	{
		swap_buffer((char *)minext, 3, sizeof(float));
		swap_buffer((char *)maxext, 3, sizeof(float));
		swap_buffer((char *)&nverts, 1, sizeof(int));
		swap_buffer((char *)&ncells, 1, sizeof(int));
		swap_buffer((char *)dim, 3, sizeof(unsigned int));
		swap_buffer((char *)orig, 3, sizeof(float));
		swap_buffer((char *)span, 3, sizeof(float));
	}

  
    for (k=0; k<ZDIM; k++) 
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) {
	  c = result[IndexVect(i,j,k)];
	  if (c == 1)
	    c_float = 255;
	  else if (c == 2)
	    c_float = 254;
	  else if (c == 3)
	    c_float = 253;
	  else if (c == 4)
	    c_float = 252;
	  else
	    c_float = dataset[IndexVect(i,j,k)]*0.98f;
	if(!big_endian())
	  swap_buffer((char *)&c_float, 1, sizeof(float));
	  fwrite_return = fwrite(&c_float, sizeof(float), 1, fp);
	}
  
    fclose(fp);
  

#if 0
  
    if ((fp1=fopen("heart_component_01.rawiv", "w"))==NULL){
      printf("read error...\n");
      exit(0); 
    }

	if(!big_endian())
	{
	swap_buffer((char *)minext, 3, sizeof(float));
    swap_buffer((char *)maxext, 3, sizeof(float));
    swap_buffer((char *)&nverts, 1, sizeof(int));
    swap_buffer((char *)&ncells, 1, sizeof(int));
    swap_buffer((char *)dim, 3, sizeof(unsigned int));
    swap_buffer((char *)orig, 3, sizeof(float));
    swap_buffer((char *)span, 3, sizeof(float));
	}
  
    fwrite(minext, sizeof(float), 3, fp1);
    fwrite(maxext, sizeof(float), 3, fp1);
    fwrite(&nverts, sizeof(int), 1, fp1);
    fwrite(&ncells, sizeof(int), 1, fp1);
    fwrite(dim, sizeof(unsigned int), 3, fp1);
    fwrite(orig, sizeof(float), 3, fp1);
    fwrite(span, sizeof(float), 3, fp1);

	if(!big_endian())
	{
    swap_buffer((char *)minext, 3, sizeof(float));
    swap_buffer((char *)maxext, 3, sizeof(float));
    swap_buffer((char *)&nverts, 1, sizeof(int));
    swap_buffer((char *)&ncells, 1, sizeof(int));
    swap_buffer((char *)dim, 3, sizeof(unsigned int));
    swap_buffer((char *)orig, 3, sizeof(float));
    swap_buffer((char *)span, 3, sizeof(float));
	} 
  
    for (k=0; k<ZDIM; k++) 
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) {
	  c = result[IndexVect(i,j,k)];
	  if (c == 1)
	    c_float = dataset[IndexVect(i,j,k)];
	  else
	    c_float = 0;
	if(!big_endian())
	  swap_buffer((char *)&c_float, 1, sizeof(float));
	  fwrite(&c_float, sizeof(float), 1, fp1);
	}
    fclose(fp1);
  
  
    if ((fp1=fopen("heart_component_02.rawiv", "w"))==NULL){
      printf("read error...\n");
      exit(0); 
    }

	if(!big_endian())
	{
    swap_buffer((char *)minext, 3, sizeof(float));
    swap_buffer((char *)maxext, 3, sizeof(float));
    swap_buffer((char *)&nverts, 1, sizeof(int));
    swap_buffer((char *)&ncells, 1, sizeof(int));
    swap_buffer((char *)dim, 3, sizeof(unsigned int));
    swap_buffer((char *)orig, 3, sizeof(float));
    swap_buffer((char *)span, 3, sizeof(float));
	}
  
    fwrite(minext, sizeof(float), 3, fp1);
    fwrite(maxext, sizeof(float), 3, fp1);
    fwrite(&nverts, sizeof(int), 1, fp1);
    fwrite(&ncells, sizeof(int), 1, fp1);
    fwrite(dim, sizeof(unsigned int), 3, fp1);
    fwrite(orig, sizeof(float), 3, fp1);
    fwrite(span, sizeof(float), 3, fp1);

	if(!big_endian())
	{
    swap_buffer((char *)minext, 3, sizeof(float));
    swap_buffer((char *)maxext, 3, sizeof(float));
    swap_buffer((char *)&nverts, 1, sizeof(int));
    swap_buffer((char *)&ncells, 1, sizeof(int));
    swap_buffer((char *)dim, 3, sizeof(unsigned int));
    swap_buffer((char *)orig, 3, sizeof(float));
    swap_buffer((char *)span, 3, sizeof(float));
	} 
  
    for (k=0; k<ZDIM; k++) 
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) {
	  c = result[IndexVect(i,j,k)];
	  if (c == 2)
	    c_float = dataset[IndexVect(i,j,k)];
	  else
	    c_float = 0;
	if(!big_endian())
	  swap_buffer((char *)&c_float, 1, sizeof(float));
	  fwrite(&c_float, sizeof(float), 1, fp1);
	}
    fclose(fp1);

  
    if ((fp1=fopen("heart_component_03.rawiv", "w"))==NULL){
      printf("read error...\n");
      exit(0); 
    }
#ifdef _LITTLE_ENDIAN
    swap_buffer((char *)minext, 3, sizeof(float));
    swap_buffer((char *)maxext, 3, sizeof(float));
    swap_buffer((char *)&nverts, 1, sizeof(int));
    swap_buffer((char *)&ncells, 1, sizeof(int));
    swap_buffer((char *)dim, 3, sizeof(unsigned int));
    swap_buffer((char *)orig, 3, sizeof(float));
    swap_buffer((char *)span, 3, sizeof(float));
#endif 
  
    fwrite(minext, sizeof(float), 3, fp1);
    fwrite(maxext, sizeof(float), 3, fp1);
    fwrite(&nverts, sizeof(int), 1, fp1);
    fwrite(&ncells, sizeof(int), 1, fp1);
    fwrite(dim, sizeof(unsigned int), 3, fp1);
    fwrite(orig, sizeof(float), 3, fp1);
    fwrite(span, sizeof(float), 3, fp1);

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
	  c = result[IndexVect(i,j,k)];
	  if (c == 3)
	    c_float = dataset[IndexVect(i,j,k)];
	  else
	    c_float = 0;
#ifdef _LITTLE_ENDIAN
	  swap_buffer((char *)&c_float, 1, sizeof(float));
#endif 
	  fwrite(&c_float, sizeof(float), 1, fp1);
	}
    fclose(fp1);

  
    if ((fp1=fopen("heart_component_04.rawiv", "w"))==NULL){
      printf("read error...\n");
      exit(0); 
    }
#ifdef _LITTLE_ENDIAN
    swap_buffer((char *)minext, 3, sizeof(float));
    swap_buffer((char *)maxext, 3, sizeof(float));
    swap_buffer((char *)&nverts, 1, sizeof(int));
    swap_buffer((char *)&ncells, 1, sizeof(int));
    swap_buffer((char *)dim, 3, sizeof(unsigned int));
    swap_buffer((char *)orig, 3, sizeof(float));
    swap_buffer((char *)span, 3, sizeof(float));
#endif 
  
    fwrite(minext, sizeof(float), 3, fp1);
    fwrite(maxext, sizeof(float), 3, fp1);
    fwrite(&nverts, sizeof(int), 1, fp1);
    fwrite(&ncells, sizeof(int), 1, fp1);
    fwrite(dim, sizeof(unsigned int), 3, fp1);
    fwrite(orig, sizeof(float), 3, fp1);
    fwrite(span, sizeof(float), 3, fp1);

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
	  c = result[IndexVect(i,j,k)];
	  if (c == 4)
	    c_float = dataset[IndexVect(i,j,k)];
	  else
	    c_float = 0;
#ifdef _LITTLE_ENDIAN
	  swap_buffer((char *)&c_float, 1, sizeof(float));
#endif 
	  fwrite(&c_float, sizeof(float), 1, fp1);
	}
    fclose(fp1);

  
    if ((fp1=fopen("heart_component_all.rawiv", "w"))==NULL){
      printf("read error...\n");
      exit(0); 
    }
#ifdef _LITTLE_ENDIAN
    swap_buffer((char *)minext, 3, sizeof(float));
    swap_buffer((char *)maxext, 3, sizeof(float));
    swap_buffer((char *)&nverts, 1, sizeof(int));
    swap_buffer((char *)&ncells, 1, sizeof(int));
    swap_buffer((char *)dim, 3, sizeof(unsigned int));
    swap_buffer((char *)orig, 3, sizeof(float));
    swap_buffer((char *)span, 3, sizeof(float));
#endif 
  
    fwrite(minext, sizeof(float), 3, fp1);
    fwrite(maxext, sizeof(float), 3, fp1);
    fwrite(&nverts, sizeof(int), 1, fp1);
    fwrite(&ncells, sizeof(int), 1, fp1);
    fwrite(dim, sizeof(unsigned int), 3, fp1);
    fwrite(orig, sizeof(float), 3, fp1);
    fwrite(span, sizeof(float), 3, fp1);

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
	  c = result[IndexVect(i,j,k)];
	  if (c >= 1 && c <= 4)
	    c_float = dataset[IndexVect(i,j,k)];
	  else
	    c_float = 0;
#ifdef _LITTLE_ENDIAN
	  swap_buffer((char *)&c_float, 1, sizeof(float));
#endif 
	  fwrite(&c_float, sizeof(float), 1, fp1);
	}
    fclose(fp1);




    if ((fp1=fopen("heart_segment.rawv", "w"))==NULL){
      printf("read error...\n");
      exit(0); 
    }
  
    NumTimeStep = 1;
    NumVariable = 4;
    MinXYZT[0]=0;
    MinXYZT[1]=0;
    MinXYZT[2]=0;
    MinXYZT[3]=0;
    MaxXYZT[0]=XDIM-1.f;
    MaxXYZT[1]=YDIM-1.f;
    MaxXYZT[2]=ZDIM-1.f;
    MaxXYZT[3]=1;
    VariableType[0] = 1;
    VariableName[0] = (char*)malloc(sizeof(char)*64);
    strcpy (VariableName[0], "red");
    VariableType[1] = 1;
    VariableName[1] = (char*)malloc(sizeof(char)*64);
    strcpy (VariableName[1], "green");
    VariableType[2] = 1;
    VariableName[2] = (char*)malloc(sizeof(char)*64);
    strcpy (VariableName[2], "blue");
    VariableType[3] = 1;
    VariableName[3] = (char*)malloc(sizeof(char)*64);
    strcpy (VariableName[3], "alpha");
  
    xdim = XDIM;
    ydim = YDIM;
    zdim = ZDIM;
#ifdef _LITTLE_ENDIAN
    swap_buffer((char *)&MagicNumW, 1, sizeof(unsigned int));
    swap_buffer((char *)&xdim, 1, sizeof(unsigned int));
    swap_buffer((char *)&ydim, 1, sizeof(unsigned int));
    swap_buffer((char *)&zdim, 1, sizeof(unsigned int));
    swap_buffer((char *)&NumTimeStep, 1, sizeof(unsigned int));
    swap_buffer((char *)&NumVariable, 1, sizeof(unsigned int));
    swap_buffer((char *)MinXYZT, 4, sizeof(float));
    swap_buffer((char *)MaxXYZT, 4, sizeof(float));
#endif 
  
    fwrite(&MagicNumW, sizeof(unsigned int), 1, fp1);
    fwrite(&xdim, sizeof(unsigned int), 1, fp1);
    fwrite(&ydim, sizeof(unsigned int), 1, fp1);
    fwrite(&zdim, sizeof(unsigned int), 1, fp1);
    fwrite(&NumTimeStep, sizeof(unsigned int), 1, fp1);
    fwrite(&NumVariable, sizeof(unsigned int), 1, fp1);
    fwrite(MinXYZT, sizeof(float), 4, fp1);
    fwrite(MaxXYZT, sizeof(float), 4, fp1);

#ifdef _LITTLE_ENDIAN
    swap_buffer((char *)&MagicNumW, 1, sizeof(unsigned int));
    swap_buffer((char *)&xdim, 1, sizeof(unsigned int));
    swap_buffer((char *)&ydim, 1, sizeof(unsigned int));
    swap_buffer((char *)&zdim, 1, sizeof(unsigned int));
    swap_buffer((char *)&NumTimeStep, 1, sizeof(unsigned int));
    swap_buffer((char *)&NumVariable, 1, sizeof(unsigned int));
    swap_buffer((char *)MinXYZT, 4, sizeof(float));
    swap_buffer((char *)MaxXYZT, 4, sizeof(float));
#endif 

    for (m=0; m<4; m++) {
      fwrite(&VariableType[m], sizeof(unsigned char), 1, fp1);
      fwrite(VariableName[m], sizeof(unsigned char), 64, fp1);
    }
  
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) {
	  c = result[IndexVect(i,j,k)];
	  if (c == 1) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 2) {
	    c_unchar = 0/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 3) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 4) {
	    c_unchar = 0/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else {
	    c_unchar = 0;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	}


    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) {
	  c = result[IndexVect(i,j,k)];
	  if (c == 1) {
	    c_unchar = 0/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 2) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 3) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 4) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else {
	    c_unchar = 0;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	}


    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) {
	  c = result[IndexVect(i,j,k)];
	  if (c == 1) {
	    c_unchar = 0/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 2) {
	    c_unchar = 0/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 3) {
	    c_unchar = 0/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 4) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else {
	    c_unchar = 0;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	}

    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) {
	  c = result[IndexVect(i,j,k)];
	  if (c == 1) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 2) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 3) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else if (c == 4) {
	    c_unchar = 255/*dataset[IndexVect(i,j,k)]/255.0*/;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	  else {
	    c_unchar = 0;
	    fwrite(&c_unchar, sizeof(unsigned char), 1, fp1);
	  }
	}

    fclose(fp1);
#endif
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
