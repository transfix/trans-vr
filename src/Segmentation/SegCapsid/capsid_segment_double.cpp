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

#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))
#define max2(x, y) ((x>y) ? (x):(y))
#define min2(x, y) ((x<y) ? (x):(y))
#define MAX_TIME      9999999.0f
#define PIE           3.1415926536f
#define ANGL1         1.1071487178f
#define ANGL2         2.0344439358f

#define ALPHA   0.1f

namespace SegCapsid {

typedef struct {
  unsigned short  *x;
  unsigned short  *y;
  unsigned short  *z;
  float *t;
  unsigned long size;
}MinHeapS;

typedef struct {
  float x;
  float y;
  float z;
}VECTOR;

typedef struct {
  unsigned short x;
  unsigned short y;
  unsigned short z;
}INTVEC;

static int XDIM, YDIM, ZDIM;
static int min_x,min_y,min_z;
static float min_t;
static MinHeapS* min_heap;
static float *tdata;
static unsigned short *seed_index;
static float t_low;
static float *dataset;
static VECTOR *FiveFold;

void GetMinimum2(void);
void InsertHeap2(int x, int y, int z);
void GetTime2(void);
float GetImgGradient(int, int, int);
VECTOR Rotate(float, float, float, float, float, float, int, int, int);
void FindDuplicate2(int, int, int, unsigned short, float);


void DoubleCapsidSegment(int xd,int yd,int zd,float tlow,float *img,unsigned short *result, 
			VECTOR *five_fold, float small_radius, float large_radius)
{
  int i,j,k;
  float radius;
  
  
  t_low = tlow;
  
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = img;
  seed_index = result;
  FiveFold = five_fold;
  
  min_heap=(MinHeapS*)malloc(sizeof(MinHeapS));
  min_heap->x = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->y = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->z = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->t = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  tdata = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  
  
  min_heap->size = 0;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	tdata[IndexVect(i,j,k)] = MAX_TIME;
	seed_index[IndexVect(i,j,k)] = 0;

	radius = (float)sqrt(double((i-XDIM/2)*(i-XDIM/2)+(j-YDIM/2)*(j-YDIM/2)+(k-ZDIM/2)*(k-ZDIM/2)));
	if (radius > large_radius-1 && radius < large_radius+1 &&
	    dataset[IndexVect(i,j,k)] > t_low) {
	  seed_index[IndexVect(i,j,k)] = 1;
	  min_heap->x[min_heap->size]=i;
	  min_heap->y[min_heap->size]=j;
	  min_heap->z[min_heap->size]=k;
	  min_heap->t[min_heap->size]=0;
	  min_heap->size ++;
	  tdata[IndexVect(i,j,k)] = 0;
	}
	else if (radius < small_radius+1 && radius > small_radius-1 &&
		 dataset[IndexVect(i,j,k)] > t_low) {
	  seed_index[IndexVect(i,j,k)] = 2;
	  min_heap->x[min_heap->size]=i;
	  min_heap->y[min_heap->size]=j;
	  min_heap->z[min_heap->size]=k;
	  min_heap->t[min_heap->size]=0;
	  min_heap->size ++;
	  tdata[IndexVect(i,j,k)] = 0;
	}
	
      }
  
  while (1){
    
    GetMinimum2();
    
    if (min_t > MAX_TIME || min_heap->size > (unsigned long)(XDIM*YDIM*ZDIM-100))
      break;
    GetTime2();
  }
  
  free(min_heap->x);
  free(min_heap->y);
  free(min_heap->z);
  free(min_heap->t);
  free(min_heap);
  free(tdata);
  
}



void GetMinimum2(void)
{
  int pointer, left, right;
  int x, y, z;
  float t;

  min_x = min_heap->x[0];
  min_y = min_heap->y[0];
  min_z = min_heap->z[0];
  min_t = min_heap->t[0];
  
  x=min_heap->x[min_heap->size-1];
  y=min_heap->y[min_heap->size-1];
  z=min_heap->z[min_heap->size-1];
  t=min_heap->t[min_heap->size-1];
  
  min_heap->size--;
  
  pointer=1;
  while ((unsigned long)pointer <= min_heap->size/2) {
    left=2*pointer;
    right=2*pointer+1;
    if ((min_heap->t[left-1] <= min_heap->t[right-1]) && (min_heap->t[left-1] < t)) {
      min_heap->x[pointer-1]=min_heap->x[left-1];
      min_heap->y[pointer-1]=min_heap->y[left-1];
      min_heap->z[pointer-1]=min_heap->z[left-1];
      min_heap->t[pointer-1]=min_heap->t[left-1];
      pointer=left;
    }
    else if ((min_heap->t[left-1] > min_heap->t[right-1]) && (min_heap->t[right-1] < t)){
      min_heap->x[pointer-1]=min_heap->x[right-1];
      min_heap->y[pointer-1]=min_heap->y[right-1];
      min_heap->z[pointer-1]=min_heap->z[right-1];
      min_heap->t[pointer-1]=min_heap->t[right-1];
      pointer=right;
    }
    else break;
  }

  min_heap->x[pointer-1]=x;
  min_heap->y[pointer-1]=y;
  min_heap->z[pointer-1]=z;
  min_heap->t[pointer-1]=t;

}
    
void InsertHeap2(int x, int y, int z)
{
  int pointer, parent;
  float t;

  t = tdata[IndexVect(x,y,z)];
  min_heap->size++;
  pointer=min_heap->size;

  while (pointer > 1) {
    if (pointer%2 == 0) {
      parent=pointer/2;
      if (t < min_heap->t[parent-1]) {
	min_heap->x[pointer-1]=min_heap->x[parent-1];
	min_heap->y[pointer-1]=min_heap->y[parent-1];
	min_heap->z[pointer-1]=min_heap->z[parent-1];
	min_heap->t[pointer-1]=min_heap->t[parent-1];
	pointer=parent;
      }
      else break;
    }
    else if (pointer%2 == 1){
      parent=(pointer-1)/2;
      if (t < min_heap->t[parent-1]) {
	min_heap->x[pointer-1]=min_heap->x[parent-1];
	min_heap->y[pointer-1]=min_heap->y[parent-1];
	min_heap->z[pointer-1]=min_heap->z[parent-1];
	min_heap->t[pointer-1]=min_heap->t[parent-1];
	pointer=parent;
      }
      else break;
    }
  }

  min_heap->x[pointer-1]=x;
  min_heap->y[pointer-1]=y;
  min_heap->z[pointer-1]=z;
  min_heap->t[pointer-1]=t;

}



void GetTime2(void)
{
  int tempt_x, tempt_y, tempt_z;
  int tx, ty, tz;
  float intens, value;
  unsigned char boundary;  
  unsigned short index;
  float t;


  index = seed_index[IndexVect(min_x,min_y,min_z)];
  

  tempt_x=max2(min_x-1,0);
  tempt_y=min_y;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
    
    boundary = 0;
    value=(float)(MAX_TIME+1.0);

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      t=(float)(MAX_TIME+1.0);
    else
      t=value;
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,index,t);
    
  }
  
  tempt_x=min2(min_x+1,XDIM-1);
  tempt_y=min_y;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
      
    boundary = 0;
    value=(float)(MAX_TIME+1.0);

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      t=(float)(MAX_TIME+1.0);
    else
      t=value;
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,index,t);
    
  }

  tempt_y=max2(min_y-1,0);
  tempt_x=min_x;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
        
    boundary = 0;
    value=(float)(MAX_TIME+1.0);

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      t=(float)(MAX_TIME+1.0);
    else
      t=value;
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,index,t);
    
  }

  tempt_y=min2(min_y+1,YDIM-1);
  tempt_x=min_x;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
       
    boundary = 0;
    value=(float)(MAX_TIME+1.0);

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      t=(float)(MAX_TIME+1.0);
    else
      t=value;
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,index,t);
    
  }
  
  tempt_z=max2(min_z-1,0);
  tempt_x=min_x;
  tempt_y=min_y;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
       
    boundary = 0;
    value=(float)(MAX_TIME+1.0);

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      t=(float)(MAX_TIME+1.0);
    else
      t=value;
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,index,t);
    
  }

  tempt_z=min2(min_z+1,ZDIM-1);
  tempt_x=min_x;
  tempt_y=min_y;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
      
    boundary = 0;
    value=(float)(MAX_TIME+1.0);

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] > 0 &&
	seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value=intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      t=(float)(MAX_TIME+1.0);
    else
      t=value;
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,index,t);
    
  }
}


void FindDuplicate2(int ax, int ay, int az, unsigned short index, float t)
{
  int i,j,k;
  int m,n;
  float theta, phi;
  float sx,sy,sz;
  float cx,cy,cz;
  float nx,ny,nz;
  VECTOR sv;
  

  if (dataset[IndexVect(ax,ay,az)] >= t_low &&
      seed_index[IndexVect(ax,ay,az)] == 0) {
    seed_index[IndexVect(ax,ay,az)] = index;
    tdata[IndexVect(ax,ay,az)] = t;
    InsertHeap2(ax,ay,az);
  }
  
  cx = FiveFold[0].x-XDIM/2;
  cy = FiveFold[0].y-YDIM/2;
  cz = FiveFold[0].z-ZDIM/2;
  theta = (float)atan2(cy,cx);
  phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
  for (n = 1; n < 5; n++) {
    sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
    i = (int)(sv.x+0.5);
    j = (int)(sv.y+0.5);
    k = (int)(sv.z+0.5);
    if (dataset[IndexVect(i,j,k)] >= t_low &&
	seed_index[IndexVect(i,j,k)] == 0) {
      seed_index[IndexVect(i,j,k)] = index;
      tdata[IndexVect(i,j,k)] = t;
      InsertHeap2(i,j,k);
    }
  }
  
  for (m = 1; m < 11; m++) {
    nx = FiveFold[m].x-XDIM/2;
    ny = FiveFold[m].y-YDIM/2;
    nz = FiveFold[m].z-ZDIM/2;
    sx = nz*cy-ny*cz;
    sy = nx*cz-nz*cx;
    sz = ny*cx-nx*cy;
    theta = (float)atan2(sy,sx);
    phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
    if (m < 6)
      sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL1,XDIM,YDIM,ZDIM);
    else
      sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL2,XDIM,YDIM,ZDIM);
    sx = sv.x;
    sy = sv.y;
    sz = sv.z;
    
    theta = (float)atan2(ny,nx);
    phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
    for (n = 0; n < 5; n++) {
      sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,XDIM,YDIM,ZDIM);
      i = (int)(sv.x+0.5);
      j = (int)(sv.y+0.5);
      k = (int)(sv.z+0.5);
      if (dataset[IndexVect(i,j,k)] >= t_low &&
	  seed_index[IndexVect(i,j,k)] == 0) {
	seed_index[IndexVect(i,j,k)] = index;
	tdata[IndexVect(i,j,k)] = t;
	InsertHeap2(i,j,k);
      }
    }
    }
  
  nx = FiveFold[1].x-XDIM/2;
  ny = FiveFold[1].y-YDIM/2;
  nz = FiveFold[1].z-ZDIM/2;
  sx = nz*cy-ny*cz;
  sy = nx*cz-nz*cx;
  sz = ny*cx-nx*cy;
  theta = (float)atan2(sy,sx);
  phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
  sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,PIE,XDIM,YDIM,ZDIM);
  i = (int)(sv.x+0.5);
  j = (int)(sv.y+0.5);
  k = (int)(sv.z+0.5);
  if (dataset[IndexVect(i,j,k)] >= t_low &&
      seed_index[IndexVect(i,j,k)] == 0) {
    seed_index[IndexVect(i,j,k)] = index;
    tdata[IndexVect(i,j,k)] = t;
    InsertHeap2(i,j,k);
  }
  sx = sv.x;
  sy = sv.y;
  sz = sv.z;
  
  nx = FiveFold[11].x-XDIM/2;
  ny = FiveFold[11].y-YDIM/2;
  nz = FiveFold[11].z-ZDIM/2;
  theta = (float)atan2(ny,nx);
  phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
  for (n = 1; n < 5; n++) {
    sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
    i = (int)(sv.x+0.5);
    j = (int)(sv.y+0.5);
    k = (int)(sv.z+0.5);
    if (dataset[IndexVect(i,j,k)] >= t_low &&
	seed_index[IndexVect(i,j,k)] == 0) {
      seed_index[IndexVect(i,j,k)] = index;
      tdata[IndexVect(i,j,k)] = t;
      InsertHeap2(i,j,k);
    }
  }
}

};
