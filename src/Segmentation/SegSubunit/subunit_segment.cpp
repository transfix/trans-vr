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

#define max2(x, y) ((x>y) ? (x):(y))
#define min2(x, y) ((x<y) ? (x):(y))
#define PIE           3.1415926f
#define ANGL1         1.107149f
#define ANGL2         2.034444f

#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))
#define MAX_TIME 9999999.0f
#define ALPHA   0.1f

namespace SegSubunit {

typedef struct {
  unsigned short *x;
  unsigned short *y;
  unsigned short *z;
  float *t;
  unsigned long size;
}MinHeapS;

typedef struct {
  float x;
  float y;
  float z;
}VECTOR;

typedef struct {
  float sx;
  float sy;
  float sz;
  float ex;
  float ey;
  float ez;
}DB_VECTOR;


typedef struct {
  unsigned short x;
  unsigned short y;
  unsigned short z;
}INTVEC;

static int min_x,min_y,min_z;
static float min_t;
static MinHeapS* min_heap;
static float t_low;
static int XDIM, YDIM, ZDIM;
static float *dataset;
static float *tdata;
static VECTOR *FiveFold;
static DB_VECTOR *LocalFold;
static int localfoldnum;
static unsigned short* seed_index;
static int numfold5;
static int localfold;
static INTVEC *dv;
static unsigned short *index_tmp;

void GetMinimum2(void);
void InsertHeap2(int, int, int);
void GetTime2(float*, float*);
void FindDuplicate2(int, int, int,float*, float*,  unsigned short, float);
VECTOR Rotate(float, float, float, float, float, float, float, float, float); //int, int, int);
char LocalRotate2(float, float, float,float*, float*, unsigned short, int*,int);
void Heap_Init2(float, float, float, float, float, float, float*, int,int);
float GetImgGradient(int, int, int);


void SubunitSegment(int xd, int yd, int zd, float* orig_tmp, float* span_tmp, float *data, unsigned short* result,
		    float tlow, VECTOR *five_fold, DB_VECTOR *local_fold, int fdnum, 
		    int numfd, int numfd5, int init_radius)
{
  int i,j,k;
  int num;
  unsigned short index;

  float max_tmp[3];
  

  t_low = tlow;
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;

  max_tmp[0]=orig_tmp[0]+span_tmp[0]*(XDIM-1);
  max_tmp[1]=orig_tmp[1]+span_tmp[1]*(YDIM-1);
  max_tmp[2]=orig_tmp[2]+span_tmp[2]*(ZDIM-1);

  dataset = data;
  FiveFold = five_fold;
  LocalFold = local_fold;
  localfoldnum = fdnum;
  localfold = 60*numfd;
  seed_index = result;
  numfold5 = numfd5;

  
  min_heap=(MinHeapS*)malloc(sizeof(MinHeapS));
  min_heap->x = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->y = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->z = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->t = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  tdata = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  
  
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	seed_index[IndexVect(i,j,k)] = 10000;
	tdata[IndexVect(i,j,k)] = (float)MAX_TIME;
      }


  index = 0;
  min_heap->size = 0;
  if (numfd5 > 0) {
    while (index < 12) {
//      Heap_Init2(XDIM/2.f,YDIM/2.f,ZDIM/2.f,FiveFold[index].x,FiveFold[index].y,
      Heap_Init2(max_tmp[0]/2,max_tmp[1]/2,max_tmp[2]/2, FiveFold[index].x,FiveFold[index].y,
		 FiveFold[index].z, max_tmp,index,init_radius);
      index++;
    }
  }



  num = 0;
  while (num < localfold) {
    if (LocalFold[num].sx != -9999) {
      Heap_Init2(LocalFold[num].sx,LocalFold[num].sy,LocalFold[num].sz,  
		LocalFold[num].ex,LocalFold[num].ey,LocalFold[num].ez, max_tmp,index,init_radius);
    }
    index++;
    num++;
  }
  
  dv = (INTVEC *)malloc(sizeof(INTVEC)*localfoldnum*60);
  index_tmp = (unsigned short *)malloc(sizeof(unsigned short)*localfoldnum*60);

  while (1){
    
    GetMinimum2();
  
    if (min_t > MAX_TIME || (int)min_heap->size > (int)(XDIM*YDIM*ZDIM-100))
      break;

    GetTime2(orig_tmp, span_tmp);
  
  }
  

  free(min_heap->x);
  free(min_heap->y);
  free(min_heap->z);
  free(min_heap->t);
  free(min_heap);
  free(tdata);
  free(dv);
  free(index_tmp);

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
  while (pointer <= (int)(min_heap->size/2)) {
    left=2*pointer;
    right=2*pointer+1;
    if ((min_heap->t[left-1] <= min_heap->t[right-1]) && 
	(min_heap->t[left-1] < t)) {
      min_heap->x[pointer-1]=min_heap->x[left-1];
      min_heap->y[pointer-1]=min_heap->y[left-1];
      min_heap->z[pointer-1]=min_heap->z[left-1];
      min_heap->t[pointer-1]=min_heap->t[left-1];
      pointer=left;
    }
    else if ((min_heap->t[left-1] > min_heap->t[right-1]) && 
	     (min_heap->t[right-1] < t)){
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



void GetTime2(float* orig_tmp, float* span_tmp)
{
  int tempt_x, tempt_y, tempt_z;
  int tx, ty, tz;
  float intens, value;
  unsigned short index;
  unsigned char boundary;  
  

  index = seed_index[IndexVect(min_x,min_y,min_z)];
  
  tempt_x=max2(min_x-1,0);
  tempt_y=min_y;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] >= 9999) {   
    value = (float)(MAX_TIME+1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
    

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index))
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      value=(float)(MAX_TIME+1.0);
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,orig_tmp, span_tmp, index,value);
    
  }

  
  tempt_x=min2(min_x+1,XDIM-1);
  tempt_y=min_y;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] >= 9999) {   
    value = (float)(MAX_TIME+1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
    

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index))
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      value=(float)(MAX_TIME+1.0);
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,orig_tmp, span_tmp,index,value);
    
  }


  tempt_y=max2(min_y-1,0);
  tempt_x=min_x;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] >= 9999) {   
    value = (float)(MAX_TIME+1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
    

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index))
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      value=(float)(MAX_TIME+1.0);
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,orig_tmp, span_tmp, index,value);
    
  }


  tempt_y=min2(min_y+1,YDIM-1);
  tempt_x=min_x;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] >= 9999) {   
    value = (float)(MAX_TIME+1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
    

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index))
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      value=(float)(MAX_TIME+1.0);
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,orig_tmp, span_tmp,index,value);
    
  }

  
  tempt_z=max2(min_z-1,0);
  tempt_x=min_x;
  tempt_y=min_y;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] >= 9999) {   
    value = (float)(MAX_TIME+1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
    

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index))
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      value=(float)(MAX_TIME+1.0);
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,orig_tmp, span_tmp, index,value);
    
  }


  tempt_z=min2(min_z+1,ZDIM-1);
  tempt_x=min_x;
  tempt_y=min_y;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] >= 9999) {   
    value = (float)(MAX_TIME+1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens));
    

    tx=max2(tempt_x-1,0);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index))
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tx=min2(tempt_x+1,XDIM-1);
    ty=tempt_y;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=max2(tempt_y-1,0);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    ty=min2(tempt_y+1,YDIM-1);
    tx=tempt_x;
    tz=tempt_z;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }
    
    tz=max2(tempt_z-1,0);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    tz=min2(tempt_z+1,ZDIM-1);
    tx=tempt_x;
    ty=tempt_y;
    if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	(seed_index[IndexVect(tx,ty,tz)] < 9999 &&
	 seed_index[IndexVect(tx,ty,tz)] != index)) 
      boundary = 1;
    else {
      if (value > intens+tdata[IndexVect(tx,ty,tz)])
	value = intens+tdata[IndexVect(tx,ty,tz)];
    }

    if (boundary == 1) 
      value=(float)(MAX_TIME+1.0);
    
    FindDuplicate2(tempt_x,tempt_y,tempt_z,orig_tmp, span_tmp, index,value);
    
  }

}



void FindDuplicate2(int ax, int ay, int az, float* orig_tmp, float* span_tmp, unsigned short index, float t)
{
  int i,j,k;
  int m,n;
  char flag;
  float theta, phi;
  float sx,sy,sz;
  float ex,ey,ez;
  int tmp_index;
  float cx,cy,cz;
  float nx,ny,nz;
  VECTOR sv;
  int start_index;

  float max_tmp[3];

  max_tmp[0]=orig_tmp[0]+span_tmp[0]*(XDIM-1);
  max_tmp[1]=orig_tmp[1]+span_tmp[1]*(YDIM-1);
  max_tmp[2]=orig_tmp[2]+span_tmp[2]*(ZDIM-1);

  
  if (index < 12*numfold5) {
    tmp_index = 0;
    flag = 1;
    
    cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
    cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
    cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
    if (index == 0) {
      sv.x = orig_tmp[0]+ span_tmp[0]*ax;
      sv.y = orig_tmp[1]+ span_tmp[1]*ay;
      sv.z = orig_tmp[2]+ span_tmp[2]*az;
      //sv.x = (float)ax;
      //sv.y = (float)ay;
      //sv.z = (float)az;
    }
    else if (index < 11) {
      nx = FiveFold[index].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[index].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[index].z-max_tmp[2]/2; //ZDIM/2;
      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      sv = Rotate(orig_tmp[0]+ span_tmp[0]*ax,orig_tmp[1]+ span_tmp[1]*ay,orig_tmp[2]+ span_tmp[2]*az,theta,phi,-PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;
      
      ex = ny*cz-nz*cy;
      ey = nz*cx-nx*cz;
      ez = nx*cy-ny*cx;
      theta = (float)atan2(ey,ex);
      phi = (float)atan2(ez,sqrt(ex*ex+ey*ey));
      if (index < 6)
	sv = Rotate(sx,sy,sz,theta,phi,ANGL1,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      else
	sv = Rotate(sx,sy,sz,theta,phi,ANGL2,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    }
    else if (index == 11) {
      sx = orig_tmp[0]+ span_tmp[0]*ax;
      sy = orig_tmp[1]+ span_tmp[1]*ay;
      sz = orig_tmp[2]+ span_tmp[2]*az;
 //     sx = (float)ax;
 //     sy = (float)ay;
  //    sz = (float)az;
      nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
      ex = ny*cz-nz*cy;
      ey = nz*cx-nx*cz;
      ez = nx*cy-ny*cx;
      theta = (float)atan2(ey,ex);
      phi = (float)atan2(ez,sqrt(ex*ex+ey*ey));
      sv = Rotate(sx,sy,sz,theta,phi,PIE,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    }

    
    ex = sv.x;
    ey = sv.y;
    ez = sv.z;
    index = 0;
    cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
    cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
    cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
    if (LocalRotate2(ex,ey,ez, orig_tmp, span_tmp, index, &tmp_index, 5) == 0) {
      flag = 0;
      return;
    }
    index++;
    
    for (m = 1; m < 11; m++) {
      nx = FiveFold[m].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[m].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[m].z-max_tmp[2]/2; //ZDIM/2;
      sx = nz*cy-ny*cz;
      sy = nx*cz-nz*cx;
      sz = ny*cx-nx*cy;
      theta = (float)atan2(sy,sx);
      phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
      if (m < 6)
	sv = Rotate(ex,ey,ez,theta,phi,ANGL1,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      else
	sv = Rotate(ex,ey,ez,theta,phi,ANGL2,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;

      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      sv = Rotate(sx,sy,sz,theta,phi,PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      if (LocalRotate2(sv.x,sv.y,sv.z, orig_tmp, span_tmp, index, &tmp_index, 5) == 0) {
	flag = 0;
	return;
      }
      index++;
    }

    nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
    ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
    nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
    sx = nz*cy-ny*cz;
    sy = nx*cz-nz*cx;
    sz = ny*cx-nx*cy;
    theta = (float)atan2(sy,sx);
    phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
    sv = Rotate(ex,ey,ez,theta,phi,PIE,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    if (LocalRotate2(sv.x,sv.y,sv.z, orig_tmp, span_tmp, index, &tmp_index, 5) == 0) {
      flag = 0;
      return;
    }
    index++;
    
    if (flag == 1) {
      for (m = 0; m < tmp_index; m++) {
	i = dv[m].x;
	j = dv[m].y;
	k = dv[m].z;
	
	if (seed_index[IndexVect(i,j,k)] >= 9999) {
	  seed_index[IndexVect(i,j,k)] = index_tmp[m];
	  tdata[IndexVect(i,j,k)] = t;
	  InsertHeap2(i,j,k);
	}
      } 
    }
  }
  else if (index < localfold+12*numfold5) {
    tmp_index = 0;
    flag = 1;
    
    start_index = (index-12*numfold5)/60;
    start_index *= 60;
    
    cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
    cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
    cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
    m = (index-start_index-12*numfold5)/5;
    n = index-start_index-12*numfold5 - 5*m;
    if (m == 0) {
      if (n == 0) {
      sv.x = orig_tmp[0]+ span_tmp[0]*ax;
      sv.y = orig_tmp[1]+ span_tmp[1]*ay;
      sv.z = orig_tmp[2]+ span_tmp[2]*az;
//	sv.x = (float)ax;
//	sv.y = (float)ay;
//	sv.z = (float)az;
      }
      else {
	theta = (float)atan2(cy,cx);
	phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
//	sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,-n*2.0f*PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	sv = Rotate(orig_tmp[0]+ span_tmp[0]*ax,orig_tmp[1]+ span_tmp[1]*ay,orig_tmp[2]+ span_tmp[2]*az,theta,phi,-n*2.0f*PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      }
    }
    else if (m < 11) {
      nx = FiveFold[m].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[m].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[m].z-max_tmp[2]/2; //ZDIM/2;
      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      sv = Rotate(orig_tmp[0]+ span_tmp[0]*ax,orig_tmp[1]+ span_tmp[1]*ay,orig_tmp[2]+ span_tmp[2]*az,theta,phi,-n*2.0f*PIE/5.0f-PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      //sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,-n*2.0f*PIE/5.0f-PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;
      
      ex = ny*cz-nz*cy;
      ey = nz*cx-nx*cz;
      ez = nx*cy-ny*cx;
      theta = (float)atan2(ey,ex);
      phi = (float)atan2(ez,sqrt(ex*ex+ey*ey));
      if (m < 6)
	sv = Rotate(sx,sy,sz,theta,phi,ANGL1,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      else
	sv = Rotate(sx,sy,sz,theta,phi,ANGL2,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    }
    else if (m == 11) {
      if (n == 0) {
      sx = orig_tmp[0]+ span_tmp[0]*ax;
      sy = orig_tmp[1]+ span_tmp[1]*ay;
      sz = orig_tmp[2]+ span_tmp[2]*az;
//	sx = (float)ax;
//	sy = (float)ay;
//	sz = (float)az;
      }
      else {
	nx = FiveFold[11].x-max_tmp[0]/2; //XDIM/2;
	ny = FiveFold[11].y-max_tmp[1]/2; //YDIM/2;
	nz = FiveFold[11].z-max_tmp[2]/2; //ZDIM/2;
	theta = (float)atan2(ny,nx);
	phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
	sv = Rotate(orig_tmp[0]+ span_tmp[0]*ax,orig_tmp[1]+ span_tmp[1]*ay,orig_tmp[2]+ span_tmp[2]*az,theta,phi,-n*2.0f*PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	//sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,-n*2.0f*PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	sx = sv.x;
	sy = sv.y;
	sz = sv.z;
      }
      nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
      ex = ny*cz-nz*cy;
      ey = nz*cx-nx*cz;
      ez = nx*cy-ny*cx;
      theta = (float)atan2(ey,ex);
      phi = (float)atan2(ez,sqrt(ex*ex+ey*ey));
      sv = Rotate(sx,sy,sz,theta,phi,PIE,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    }

    
    ex = sv.x;
    ey = sv.y;
    ez = sv.z;
    index = 12*numfold5+start_index;
    cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
    cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
    cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
    if (LocalRotate2(ex,ey,ez, orig_tmp, span_tmp, index, &tmp_index, localfoldnum) == 0) {
      flag = 0;
      return;
    }
    index++;
    
    theta = (float)atan2(cy,cx);
    phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
    for (n = 1; n < 5; n++) {
      sv = Rotate(ex,ey,ez,theta,phi,n*2.0f*PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      if (LocalRotate2(sv.x,sv.y,sv.z,orig_tmp, span_tmp,  index, &tmp_index, localfoldnum) == 0) {
	flag = 0;
	return;
      }
      index++;
    }
    
    for (m = 1; m < 11; m++) {
      nx = FiveFold[m].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[m].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[m].z-max_tmp[2]/2; //ZDIM/2;
      sx = nz*cy-ny*cz;
      sy = nx*cz-nz*cx;
      sz = ny*cx-nx*cy;
      theta = (float)atan2(sy,sx);
      phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
      if (m < 6)
	sv = Rotate(ex,ey,ez,theta,phi,ANGL1,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      else
	sv = Rotate(ex,ey,ez,theta,phi,ANGL2,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;

      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      for (n = 0; n < 5; n++) {
	sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	if (LocalRotate2(sv.x,sv.y,sv.z,orig_tmp, span_tmp,  index, &tmp_index, localfoldnum) == 0) {
	  flag = 0;
	  return;
	}
	index++;
      }
	 
    }

    nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
    ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
    nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
    sx = nz*cy-ny*cz;
    sy = nx*cz-nz*cx;
    sz = ny*cx-nx*cy;
    theta = (float)atan2(sy,sx);
    phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
    sv = Rotate(ex,ey,ez,theta,phi,PIE,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    if (LocalRotate2(sv.x,sv.y,sv.z, orig_tmp, span_tmp, index, &tmp_index, localfoldnum) == 0) {
      flag = 0;
      return;
    }
    index++;
    sx = sv.x;
    sy = sv.y;
    sz = sv.z;
    
    nx = FiveFold[11].x-max_tmp[0]/2; //XDIM/2;
    ny = FiveFold[11].y-max_tmp[1]/2; //YDIM/2;
    nz = FiveFold[11].z-max_tmp[2]/2; //ZDIM/2;
    theta = (float)atan2(ny,nx);
    phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
    for (n = 1; n < 5; n++) {
      sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f,max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      if (LocalRotate2(sv.x,sv.y,sv.z,orig_tmp, span_tmp, index, &tmp_index, localfoldnum) == 0) {
	flag = 0;
	return;
      }
      index++;
    }
    
    if (flag == 1) {
      for (m = 0; m < tmp_index; m++) {
	i = dv[m].x;
	j = dv[m].y;
	k = dv[m].z;
	if (seed_index[IndexVect(i,j,k)] >= 9999) {
	  seed_index[IndexVect(i,j,k)] = index_tmp[m];
	  tdata[IndexVect(i,j,k)] = t;
	  InsertHeap2(i,j,k);
	}
      }   
    }
  }
  
  else 
    printf("indexing errors %d .... \n",index);
  
}



char LocalRotate2(float ax, float ay, float az, float* orig_tmp, float* span_tmp, unsigned short index, 
		  int *t_index, int foldnum)
{
  int i,j,k;
  char flag;
  int tmp_index;
  float sx,sy,sz;
  float ex,ey,ez;
  float x,y,z;
  float xx,yy,zz;
  float a[3][3],b[3][3];
  float theta, phi;
  int m;
  float max_tmp[3];

  max_tmp[0]=orig_tmp[0]+span_tmp[0]*(XDIM-1);
  max_tmp[1]=orig_tmp[1]+span_tmp[1]*(YDIM-1);
  max_tmp[2]=orig_tmp[2]+span_tmp[2]*(ZDIM-1);


  if (foldnum == 5) {
    tmp_index = *t_index;
    flag = 1;

	i = (int)((ax-orig_tmp[0])/span_tmp[0]+0.5);
	j = (int)((ay-orig_tmp[1])/span_tmp[1]+0.5);
	k = (int)((az-orig_tmp[2])/span_tmp[2]+0.5);

   // i = (int)(ax+0.5);
   // j = (int)(ay+0.5);
   // k = (int)(az+0.5);
    if (seed_index[IndexVect(i,j,k)] < 9999 &&
	seed_index[IndexVect(i,j,k)] != index) 
      flag = 0;
    
    dv[tmp_index].x = i;
    dv[tmp_index].y = j;
    dv[tmp_index].z = k;
    index_tmp[tmp_index] = index;
    tmp_index++;
    
    ex = max_tmp[0]/2;
    ey = max_tmp[1]/2;
    ez = max_tmp[2]/2;
 //   ex = (float)(XDIM/2);
  //  ey = (float)(YDIM/2);
   // ez = (float)(ZDIM/2);
    sx = FiveFold[index].x;
    sy = FiveFold[index].y;
    sz = FiveFold[index].z;
    
    theta = (float)atan2(sy-ey,sx-ex);
    phi = (float)atan2(sz-ez, sqrt((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)));
    
    a[0][0] = (float)(cos(0.5*PIE-phi)*cos(theta));
    a[0][1] = (float)(cos(0.5*PIE-phi)*sin(theta));
    a[0][2] = (float)-sin(0.5*PIE-phi);
    a[1][0] = (float)-sin(theta);
    a[1][1] = (float)cos(theta);
    a[1][2] = 0.f;
    a[2][0] = (float)(sin(0.5*PIE-phi)*cos(theta));
    a[2][1] = (float)(sin(0.5*PIE-phi)*sin(theta));
    a[2][2] = (float)cos(0.5*PIE-phi);
    
    b[0][0] = (float)(cos(0.5*PIE-phi)*cos(theta));
    b[0][1] = (float)-sin(theta); 
    b[0][2] = (float)(sin(0.5*PIE-phi)*cos(theta)); 
    b[1][0] = (float)(cos(0.5*PIE-phi)*sin(theta));
    b[1][1] = (float)cos(theta);
    b[1][2] = (float)(sin(0.5*PIE-phi)*sin(theta));
    b[2][0] = (float)-sin(0.5*PIE-phi);
    b[2][1] = 0.f;
    b[2][2] = (float)cos(0.5*PIE-phi);
    
    x = ax-ex;
    y = ay-ey;
    z = az-ez;
    
    xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
    yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
    zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
    
    for (m = 1; m < 5; m++) {
      x = (float)(cos(2*PIE*(float)(m)/5.0)*xx - 
	sin(2*PIE*(float)(m)/5.0)*yy);
      y = (float)(sin(2*PIE*(float)(m)/5.0)*xx + 
	cos(2*PIE*(float)(m)/5.0)*yy);
      z = zz;
  
  	 // i = (int)(b[0][0]*x+b[0][1]*y+b[0][2]*z+0.5+ex);
    //  j = (int)(b[1][0]*x+b[1][1]*y+b[1][2]*z+0.5+ey);
     // k = (int)(b[2][0]*x+b[2][1]*y+b[2][2]*z+0.5+ez);
   
      i = (int)((b[0][0]*x+b[0][1]*y+b[0][2]*z+ex-orig_tmp[0])/span_tmp[0]+0.5);
      j = (int)((b[1][0]*x+b[1][1]*y+b[1][2]*z+ey-orig_tmp[0])/span_tmp[0]+0.5);
      k = (int)((b[2][0]*x+b[2][1]*y+b[2][2]*z+ez-orig_tmp[0])/span_tmp[0]+0.5);
      if (i < 0 || i >= XDIM || j < 0 || 
	  j >= YDIM || k < 0 || k >= ZDIM) {
	printf("check the correctness of local symmetry ...\n");
	exit(0);
      }
  
      if (seed_index[IndexVect(i,j,k)] < 9999 &&
	  seed_index[IndexVect(i,j,k)] != index) 
	flag = 0;
      
      dv[tmp_index].x = i;
      dv[tmp_index].y = j;
      dv[tmp_index].z = k;
      
      index_tmp[tmp_index] = index;
      tmp_index++;
    }
    
    *t_index = tmp_index;
    return(flag);
  }
  else if (LocalFold[index-12*numfold5].ex != -9999) {
    tmp_index = *t_index;
    flag = 1;
//    i = (int)(ax+0.5);
//    j = (int)(ay+0.5);
 //   k = (int)(az+0.5);
	i = (int)((ax-orig_tmp[0])/span_tmp[0]+0.5);
	j = (int)((ay-orig_tmp[1])/span_tmp[1]+0.5);
	k = (int)((az-orig_tmp[2])/span_tmp[2]+0.5);
    
    if (seed_index[IndexVect(i,j,k)] < 9999 &&
	seed_index[IndexVect(i,j,k)] != index) 
      flag = 0;
    
    dv[tmp_index].x = i;
    dv[tmp_index].y = j;
    dv[tmp_index].z = k;
    index_tmp[tmp_index] = index;
    tmp_index++;
    
    sx = LocalFold[index-12*numfold5].ex;
    sy = LocalFold[index-12*numfold5].ey;
    sz = LocalFold[index-12*numfold5].ez;
    ex = LocalFold[index-12*numfold5].sx;
    ey = LocalFold[index-12*numfold5].sy;
    ez = LocalFold[index-12*numfold5].sz;
    
    theta = (float)atan2(sy-ey,sx-ex);
    phi = (float)atan2(sz-ez, sqrt((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)));
    
    a[0][0] = (float)(cos(0.5*PIE-phi)*cos(theta));
    a[0][1] = (float)(cos(0.5*PIE-phi)*sin(theta));
    a[0][2] = (float)-sin(0.5*PIE-phi);
    a[1][0] = (float)-sin(theta);
    a[1][1] = (float)cos(theta);
    a[1][2] = 0.f;
    a[2][0] = (float)(sin(0.5*PIE-phi)*cos(theta));
    a[2][1] = (float)(sin(0.5*PIE-phi)*sin(theta));
    a[2][2] = (float)cos(0.5*PIE-phi);
    
    b[0][0] = (float)(cos(0.5*PIE-phi)*cos(theta));
    b[0][1] = (float)-sin(theta); 
    b[0][2] = (float)(sin(0.5*PIE-phi)*cos(theta)); 
    b[1][0] = (float)(cos(0.5*PIE-phi)*sin(theta));
    b[1][1] = (float)cos(theta);
    b[1][2] = (float)(sin(0.5*PIE-phi)*sin(theta));
    b[2][0] = (float)-sin(0.5*PIE-phi);
    b[2][1] = 0.f;
    b[2][2] = (float)cos(0.5*PIE-phi);
    
    x = ax-ex;
    y = ay-ey;
    z = az-ez;
    
    xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
    yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
    zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
    
    for (m = 1; m < foldnum; m++) {
      x = (float)(cos(2*PIE*(float)(m)/(float)(foldnum))*xx - 
	sin(2*PIE*(float)(m)/(float)(foldnum))*yy);
      y = (float)(sin(2*PIE*(float)(m)/(float)(foldnum))*xx + 
	cos(2*PIE*(float)(m)/(float)(foldnum))*yy);
      z = zz;
      
//      i = (int)(b[0][0]*x+b[0][1]*y+b[0][2]*z+0.5+ex);
//      j = (int)(b[1][0]*x+b[1][1]*y+b[1][2]*z+0.5+ey);
 //     k = (int)(b[2][0]*x+b[2][1]*y+b[2][2]*z+0.5+ez);
 
      i = (int)((b[0][0]*x+b[0][1]*y+b[0][2]*z+ex-orig_tmp[0])/span_tmp[0]+0.5);
      j = (int)((b[1][0]*x+b[1][1]*y+b[1][2]*z+ey-orig_tmp[0])/span_tmp[0]+0.5);
      k = (int)((b[2][0]*x+b[2][1]*y+b[2][2]*z+ez-orig_tmp[0])/span_tmp[0]+0.5);

 	if (i < 0 || i >= XDIM || j < 0 || 
	  j >= YDIM || k < 0 || k >= ZDIM) 
	return(0);
      
      if (seed_index[IndexVect(i,j,k)] < 9999 &&
	  seed_index[IndexVect(i,j,k)] != index) 
	flag = 0;
      
      dv[tmp_index].x = i;
      dv[tmp_index].y = j;
      dv[tmp_index].z = k;
      index_tmp[tmp_index] = index;
      tmp_index++;
      
    }
    
    *t_index = tmp_index;
    return(flag);
  }
  else 
    return(1);
  
}



void Heap_Init2(float sx, float sy, float sz, 
		float ex, float ey, float ez, float* max_tmp, int index, int radius)
{
  int i,j,k;
  int m,n,l;
  float dx,dy,dz;
  float step,length;
  
  
  length = (float)sqrt((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)+(sz-ez)*(sz-ez));
  
  dx = (ex-sx)/length;
  dy = (ey-sy)/length;
  dz = (ez-sz)/length;

 // step = 1.0;

	float stepp[3];
  stepp[0] = max_tmp[0]/(XDIM-1);
  stepp[1] = max_tmp[1]/(YDIM-1);
  stepp[2] = max_tmp[2]/(ZDIM-1);
  float unitstep[3];
  unitstep[0]= stepp[0];
  unitstep[1] = stepp[1];
  unitstep[2] = stepp[2];

	while(stepp[0] < length && stepp[1]< length && stepp[2]<length){
//  while (step < length) {
//    i = (int)(sx+step*dx+0.5);
//    j = (int)(sy+step*dy+0.5);
//    k = (int)(sz+step*dz+0.5);

    i = (int)((sx+stepp[0]*dx)/unitstep[0]+0.5);
    j = (int)((sy+stepp[1]*dy)/unitstep[1]+0.5);
    k = (int)((sz+stepp[2]*dz)/unitstep[2]+0.5);


    for (l = max2(0,k-radius); l <= min2(ZDIM-1,k+radius); l++)
      for (n = max2(0,j-radius); n <= min2(YDIM-1,j+radius); n++)
	for (m = max2(0,i-radius); m <= min2(XDIM-1,i+radius); m++) {
	  if (sqrt((m-i)*(m-i)+(n-j)*(n-j)+(l-k)*(l-k)) <= radius &&
	      dataset[IndexVect(m,n,l)] > t_low &&
	      seed_index[IndexVect(m,n,l)] == 10000) {
	    seed_index[IndexVect(m,n,l)] = index;
	    min_heap->x[min_heap->size]=m;
	    min_heap->y[min_heap->size]=n;
	    min_heap->z[min_heap->size]=l;
	    min_heap->t[min_heap->size]=0.0;
	    min_heap->size ++;
	    tdata[IndexVect(m,n,l)] = 0.0;
	  }
	}

//    step += 1.0;

	  stepp[0]+= unitstep[0];
	  stepp[1]+= unitstep[1];
	  stepp[2]+= unitstep[2];

  }
}



VECTOR Rotate(float sx, float sy, float sz, 
	      float theta, float phi, float angle, float xd, float yd, float zd)
{
  float x,y,z;
  float xx,yy,zz;
  float a[3][3],b[3][3];
  VECTOR tmp;


  a[0][0] = (float)(cos(0.5*PIE-phi)*cos(theta));
  a[0][1] = (float)(cos(0.5*PIE-phi)*sin(theta));
  a[0][2] = (float)-sin(0.5*PIE-phi);
  a[1][0] = (float)-sin(theta);
  a[1][1] = (float)cos(theta);
  a[1][2] = 0.f;
  a[2][0] = (float)(sin(0.5*PIE-phi)*cos(theta));
  a[2][1] = (float)(sin(0.5*PIE-phi)*sin(theta));
  a[2][2] = (float)cos(0.5*PIE-phi);

  b[0][0] = (float)(cos(0.5*PIE-phi)*cos(theta));
  b[0][1] = (float)-sin(theta); 
  b[0][2] = (float)(sin(0.5*PIE-phi)*cos(theta)); 
  b[1][0] = (float)(cos(0.5*PIE-phi)*sin(theta));
  b[1][1] = (float)cos(theta);
  b[1][2] = (float)(sin(0.5*PIE-phi)*sin(theta));
  b[2][0] = (float)-sin(0.5*PIE-phi);
  b[2][1] = 0.f;
  b[2][2] = (float)cos(0.5*PIE-phi);


  sx = sx-xd/2;
  sy = sy-yd/2;
  sz = sz-zd/2;
  
  x = a[0][0]*sx+a[0][1]*sy+a[0][2]*sz;
  y = a[1][0]*sx+a[1][1]*sy+a[1][2]*sz;
  z = a[2][0]*sx+a[2][1]*sy+a[2][2]*sz;
      
  xx = (float)(cos(angle)*x - sin(angle)*y);
  yy = (float)(sin(angle)*x + cos(angle)*y);
  zz = z;
	
  tmp.x = b[0][0]*xx+b[0][1]*yy+b[0][2]*zz+xd/2;
  tmp.y = b[1][0]*xx+b[1][1]*yy+b[1][2]*zz+yd/2;
  tmp.z = b[2][0]*xx+b[2][1]*yy+b[2][2]*zz+zd/2;
  
  return(tmp);
  
}

};

