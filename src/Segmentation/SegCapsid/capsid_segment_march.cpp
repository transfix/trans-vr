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
#define BETA    0.03f

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

typedef struct {
  float sx;
  float sy;
  float sz;
  float ex;
  float ey;
  float ez;
}DB_VECTOR;

typedef struct {
  float trans;
  float rotat;
  float angle;
}CVM;  

static int XDIM, YDIM, ZDIM;
static int min_x,min_y,min_z;
static float min_t;
static MinHeapS* min_heap;
static float *tdata;
static unsigned short *seed_index;
static float t_low;
static float *dataset;
static float *sym_score;
static VECTOR *FiveFold;
static DB_VECTOR *LocalFold;
static CVM *LocalcoVar;
static int LocalFoldnum;
static int LocalAxisnum;
static INTVEC *dv;

void GetMinimum3(void);
void InsertHeap3(int x, int y, int z);
void GetTime3(void);
float GetScoreGradient(int, int, int);
float GetImgGradient(int, int, int);
VECTOR Rotate(float, float, float, float, float, float, int, int, int);
void FindDuplicate3(int, int, int, unsigned short, float);
VECTOR FindCopy3(float,float,float,int,int);
VECTOR CoVarRotate(float, float, float, float, float, float);
char LocalRotate3(float, float, float, unsigned short, int *, INTVEC *, int);


void CapsidSegmentMarch(int xd,int yd,int zd, float tlow, float *img, float *sym_scr,
	   unsigned short *result, VECTOR *five_fold, DB_VECTOR *local_fold,CVM* p_coVar,
	   int p_axisnum,int p_fdnum,float small_radius, float large_radius)
{
  int i,j,k;
  float radius;
  
  
  t_low = tlow;
  
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = img;
  sym_score = sym_scr;
  seed_index = result;
  FiveFold = five_fold;
  LocalFold = local_fold;
  LocalcoVar = p_coVar;
  LocalFoldnum = p_fdnum;
  LocalAxisnum = p_axisnum;
  
  min_heap=(MinHeapS*)malloc(sizeof(MinHeapS));
  min_heap->x = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->y = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->z = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
  min_heap->t = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  tdata = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  
  dv = (INTVEC *)malloc(sizeof(INTVEC)*LocalFoldnum*LocalAxisnum*60);
  
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
    
    GetMinimum3();
    
    if (min_t > MAX_TIME || min_heap->size > (unsigned long)(XDIM*YDIM*ZDIM-100))
      break;
    GetTime3();
  }
  
  free(dv);
  free(min_heap->x);
  free(min_heap->y);
  free(min_heap->z);
  free(min_heap->t);
  free(min_heap);
  free(tdata);
  
}



void GetMinimum3(void)
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
    
void InsertHeap3(int x, int y, int z)
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



void GetTime3(void)
{
  int tempt_x, tempt_y, tempt_z;
  int tx, ty, tz;
  float intens,intens1, value;
  unsigned char boundary;  
  unsigned short index;
  float t;


  index = seed_index[IndexVect(min_x,min_y,min_z)];
  

  tempt_x=max2(min_x-1,0);
  tempt_y=min_y;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetScoreGradient(tempt_x, tempt_y, tempt_z);
    intens1 = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens+BETA*intens1));
    
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
    
    FindDuplicate3(tempt_x,tempt_y,tempt_z,index,t);
    
  }
  
  tempt_x=min2(min_x+1,XDIM-1);
  tempt_y=min_y;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetScoreGradient(tempt_x, tempt_y, tempt_z);
    intens1 = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens+BETA*intens1));
      
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
    
    FindDuplicate3(tempt_x,tempt_y,tempt_z,index,t);
    
  }

  tempt_y=max2(min_y-1,0);
  tempt_x=min_x;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetScoreGradient(tempt_x, tempt_y, tempt_z);
    intens1 = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens+BETA*intens1));
        
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
    
    FindDuplicate3(tempt_x,tempt_y,tempt_z,index,t);
    
  }

  tempt_y=min2(min_y+1,YDIM-1);
  tempt_x=min_x;
  tempt_z=min_z;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetScoreGradient(tempt_x, tempt_y, tempt_z);
    intens1 = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens+BETA*intens1));
       
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
    
    FindDuplicate3(tempt_x,tempt_y,tempt_z,index,t);
    
  }
  
  tempt_z=max2(min_z-1,0);
  tempt_x=min_x;
  tempt_y=min_y;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetScoreGradient(tempt_x, tempt_y, tempt_z);
    intens1 = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens+BETA*intens1));
       
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
    
    FindDuplicate3(tempt_x,tempt_y,tempt_z,index,t);
    
  }

  tempt_z=min2(min_z+1,ZDIM-1);
  tempt_x=min_x;
  tempt_y=min_y;
  if (seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] == 0) {   
    intens = GetScoreGradient(tempt_x, tempt_y, tempt_z);
    intens1 = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA*intens+BETA*intens1));
      
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
    
    FindDuplicate3(tempt_x,tempt_y,tempt_z,index,t);
    
  }
}


void FindDuplicate3(int ax, int ay, int az, unsigned short index, float t)
{
  int i,j,k;
  int m,n,num;
  float theta, phi;
  float sx,sy,sz;
  float ex,ey,ez;
  float cx,cy,cz;
  float nx,ny,nz;
  VECTOR sv;
  int tmp_index;
  float a1,b1,c1;
  float dist,min_dist,tmp;
  int start_index, min_index=0;
  char flag;
  float x,y,z;
  float xx,yy,zz;
  

  if (index == 1) {
  
    min_dist = 99999.0f;
    for (m = 0; m < LocalAxisnum*60; m++) {
      if (LocalFold[m].sx != -9999) {
	cx = 0.5f*(LocalFold[m].sx+LocalFold[m].ex);
	cy = 0.5f*(LocalFold[m].sy+LocalFold[m].ey);
	cz = 0.5f*(LocalFold[m].sz+LocalFold[m].ez);
	if ((ax-XDIM/2)*(cx-XDIM/2)+
	    (ay-YDIM/2)*(cy-YDIM/2)+
	    (az-ZDIM/2)*(cz-ZDIM/2) > 0) {
	  a1 = LocalFold[m].sx-LocalFold[m].ex;
	  b1 = LocalFold[m].sy-LocalFold[m].ey;
	  c1 = LocalFold[m].sz-LocalFold[m].ez;
	  tmp = (float)sqrt(a1*a1+b1*b1+c1*c1);
	  x = (float)ax-LocalFold[m].ex;
	  y = (float)ay-LocalFold[m].ey;
	  z = (float)az-LocalFold[m].ez;
	  nx = b1*z-c1*y;
	  ny = x*c1-a1*z;
	  nz = a1*y-x*b1;
	  dist = (float)(sqrt(nx*nx+ny*ny+nz*nz)/tmp);
	  if (dist < min_dist) {
	    min_dist = dist;
	    min_index = m;
	  }
	}
      }
    }
    start_index = (min_index/60)*60;
    cx = FiveFold[0].x-XDIM/2;
    cy = FiveFold[0].y-YDIM/2;
    cz = FiveFold[0].z-ZDIM/2;
    m = (min_index-start_index)/5;
    n = min_index-start_index - 5*m;
    if (m == 0) {
      if (n == 0) {
	sv.x = (float)ax;
	sv.y = (float)ay;
	sv.z = (float)az;
      }
      else {
	theta = (float)atan2(cy,cx);
	phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
	sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,-n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
      }
    }
    else if (m < 11) {
      nx = FiveFold[m].x-XDIM/2;
      ny = FiveFold[m].y-YDIM/2;
      nz = FiveFold[m].z-ZDIM/2;
      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,-n*2.0f*PIE/5.0f-PIE/5.0f,XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;
      
      ex = ny*cz-nz*cy;
      ey = nz*cx-nx*cz;
      ez = nx*cy-ny*cx;
      theta = (float)atan2(ey,ex);
      phi = (float)atan2(ez,sqrt(ex*ex+ey*ey));
      if (m < 6)
	sv = Rotate(sx,sy,sz,theta,phi,ANGL1,XDIM,YDIM,ZDIM);
      else
	sv = Rotate(sx,sy,sz,theta,phi,ANGL2,XDIM,YDIM,ZDIM);
    }
    else if (m == 11) {
      if (n == 0) {
	sx = (float)ax;
	sy = (float)ay;
	sz = (float)az;
      }
      else {
	nx = FiveFold[11].x-XDIM/2;
	ny = FiveFold[11].y-YDIM/2;
	nz = FiveFold[11].z-ZDIM/2;
	theta = (float)atan2(ny,nx);
	phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
	sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,-n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
	sx = sv.x;
	sy = sv.y;
	sz = sv.z;
      }
      nx = FiveFold[1].x-XDIM/2;
      ny = FiveFold[1].y-YDIM/2;
      nz = FiveFold[1].z-ZDIM/2;
      ex = ny*cz-nz*cy;
      ey = nz*cx-nx*cz;
      ez = nx*cy-ny*cx;
      theta = (float)atan2(ey,ex);
      phi = (float)atan2(ez,sqrt(ex*ex+ey*ey));
      sv = Rotate(sx,sy,sz,theta,phi,PIE,XDIM,YDIM,ZDIM);
    }
    xx = sv.x;
    yy = sv.y;
    zz = sv.z;
    
    
    tmp_index = 0;
    flag = 1;
    
    for (num = 0; num < LocalAxisnum; num++) {
      if (num != min_index/60) {
	sv = FindCopy3(xx,yy,zz, min_index/60,num);
	x = sv.x;
	y = sv.y;
	z = sv.z;
      }
      else {
	x = xx;
	y = yy;
	z = zz;
      }
      
      start_index = num*60;
      cx = FiveFold[0].x-XDIM/2;
      cy = FiveFold[0].y-YDIM/2;
      cz = FiveFold[0].z-ZDIM/2;
      if (LocalRotate3(x,y,z,start_index, &tmp_index, dv, LocalFoldnum) == 0) {
	flag = 0;
	return;
      }
      start_index++;
      
      theta = (float)atan2(cy,cx);
      phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
      for (n = 1; n < 5; n++) {
	sv = Rotate(x,y,z,theta,phi,n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
	if (LocalRotate3(sv.x,sv.y,sv.z,start_index, &tmp_index, dv, LocalFoldnum) == 0) {
	  flag = 0;
	  return;
	}
	start_index++;
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
	  sv = Rotate(x,y,z,theta,phi,ANGL1,XDIM,YDIM,ZDIM);
	else
	  sv = Rotate(x,y,z,theta,phi,ANGL2,XDIM,YDIM,ZDIM);
	sx = sv.x;
	sy = sv.y;
	sz = sv.z;
	
	theta = (float)atan2(ny,nx);
	phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
	for (n = 0; n < 5; n++) {
	  sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,XDIM,YDIM,ZDIM);
	  if (LocalRotate3(sv.x,sv.y,sv.z,start_index, &tmp_index, dv, LocalFoldnum) == 0) {
	    flag = 0;
	    return;
	  }
	  start_index++;
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
      sv = Rotate(x,y,z,theta,phi,PIE,XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;
      if (LocalRotate3(sv.x,sv.y,sv.z,start_index, &tmp_index, dv, LocalFoldnum) == 0) {
	flag = 0;
	return;
      }
      start_index++;
      
      nx = FiveFold[11].x-XDIM/2;
      ny = FiveFold[11].y-YDIM/2;
      nz = FiveFold[11].z-ZDIM/2;
      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      for (n = 1; n < 5; n++) {
	sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
	if (LocalRotate3(sv.x,sv.y,sv.z,start_index, &tmp_index, dv, LocalFoldnum) == 0) {
	  flag = 0;
	  return;
	}
	start_index++;
      }
    }
    
    if (flag == 1) {
      for (m = 0; m < tmp_index; m++) {
	i = dv[m].x;
	j = dv[m].y;
	k = dv[m].z;
	seed_index[IndexVect(i,j,k)] = index;
	tdata[IndexVect(i,j,k)] = t;
	InsertHeap3(i,j,k);
      }   
    }
  }
  else if (index == 2) {
    if (dataset[IndexVect(ax,ay,az)] >= t_low &&
	seed_index[IndexVect(ax,ay,az)] == 0) {
      seed_index[IndexVect(ax,ay,az)] = index;
      tdata[IndexVect(ax,ay,az)] = t;
      InsertHeap3(ax,ay,az);
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
	InsertHeap3(i,j,k);
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
	  InsertHeap3(i,j,k);
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
      InsertHeap3(i,j,k);
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
	InsertHeap3(i,j,k);
      }
    }
  }
  else 
    printf("indexing errors %d .... \n",index);
   
}


char LocalRotate3(float ax, float ay, float az, unsigned short index, int *t_index,
		 INTVEC *dv, int foldnum)
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

  
  if (LocalFold[index].ex == -9999)
    return(1);

  else {
    tmp_index = *t_index;
    flag = 1;
    i = (int)(ax+0.5);
    j = (int)(ay+0.5);
    k = (int)(az+0.5);
    
    if (seed_index[IndexVect(i,j,k)] == 2) 
      flag = 0;
    
    if (seed_index[IndexVect(i,j,k)] == 0 &&
	dataset[IndexVect(i,j,k)] > t_low) {
      dv[tmp_index].x = i;
      dv[tmp_index].y = j;
      dv[tmp_index].z = k;
      tmp_index++;
    }
    
    ex = LocalFold[index].ex;
    ey = LocalFold[index].ey;
    ez = LocalFold[index].ez;
    sx = LocalFold[index].sx;
    sy = LocalFold[index].sy;
    sz = LocalFold[index].sz;
    
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
      
      i = (int)(b[0][0]*x+b[0][1]*y+b[0][2]*z+0.5+ex);
      j = (int)(b[1][0]*x+b[1][1]*y+b[1][2]*z+0.5+ey);
      k = (int)(b[2][0]*x+b[2][1]*y+b[2][2]*z+0.5+ez);
      if (i < 0 || i >= XDIM || j < 0 || 
	  j >= YDIM || k < 0 || k >= ZDIM) 
	return(0);
      
      if (seed_index[IndexVect(i,j,k)] == 2) 
	flag = 0;
      
      if (seed_index[IndexVect(i,j,k)] == 0 &&
	  dataset[IndexVect(i,j,k)] > t_low) {
	dv[tmp_index].x = i;
	dv[tmp_index].y = j;
	dv[tmp_index].z = k;
	tmp_index++;
      }
    }
    
    *t_index = tmp_index;
    return(flag);
  }
}



VECTOR FindCopy3(float ax,float ay,float az,int index1,int index2)
{
  DB_VECTOR temp1, temp2;
  float alpha,tmp;
  float x,y,z;
  float rotat,trans;
  float theta,phi;
  float t_theta,t_phi;
  float fx,fy,fz;
  float gx,gy,gz;
  float px,py,pz;
  float dx,dy,dz;
  VECTOR sv;
  

  temp1.sx = LocalFold[index1*60].sx;
  temp1.sy = LocalFold[index1*60].sy;
  temp1.sz = LocalFold[index1*60].sz;
  temp1.ex = LocalFold[index1*60].ex;
  temp1.ey = LocalFold[index1*60].ey;
  temp1.ez = LocalFold[index1*60].ez;
  temp2.sx = LocalFold[index2*60].sx;
  temp2.sy = LocalFold[index2*60].sy;
  temp2.sz = LocalFold[index2*60].sz;
  temp2.ex = LocalFold[index2*60].ex;
  temp2.ey = LocalFold[index2*60].ey;
  temp2.ez = LocalFold[index2*60].ez;
   
  alpha = LocalcoVar[index1*LocalAxisnum+index2].angle;
  rotat = LocalcoVar[index1*LocalAxisnum+index2].rotat;
  trans = LocalcoVar[index1*LocalAxisnum+index2].trans;
  
  gx = temp2.sx-temp2.ex;
  gy = temp2.sy-temp2.ey;
  gz = temp2.sz-temp2.ez;
  fx = temp1.sx-temp1.ex;
  fy = temp1.sy-temp1.ey;
  fz = temp1.sz-temp1.ez;
  px = fy*gz-fz*gy;
  py = fz*gx-fx*gz;
  pz = fx*gy-fy*gx;
  t_theta = (float)atan2(py,px);
  t_phi = (float)atan2(pz, sqrt(px*px+py*py));
  theta = (float)atan2(gy,gx);
  phi = (float)atan2(gz, sqrt(gx*gx+gy*gy));
  tmp = (float)sqrt(gx*gx+gy*gy+gz*gz);
  dx = gx/tmp;
  dy = gy/tmp;
  dz = gz/tmp;
  
  x = ax-temp1.ex;
  y = ay-temp1.ey;
  z = az-temp1.ez;
	
  sv = CoVarRotate(x,y,z,t_theta,t_phi,alpha);
  x = sv.x;
  y = sv.y;
  z = sv.z;
  sv = CoVarRotate(x,y,z,theta,phi,rotat);
  x = sv.x+temp2.ex+dx*trans;
  y = sv.y+temp2.ey+dy*trans;
  z = sv.z+temp2.ez+dz*trans;

  sv.x = x;
  sv.y = y;
  sv.z = z;

  return(sv);
}


float GetScoreGradient(int x, int y, int z)
{
  int i,j,k;
  float grad_x,grad_y,grad_z;
  float gradient;


  grad_x=0.0;
  for (j=max2(y-1,0); j<=min2(y+1,YDIM-1); j++) 
    for (k=max2(z-1,0); k<=min2(z+1,ZDIM-1); k++) {
      grad_x += sym_score[IndexVect(min2(x+1,XDIM-1),j,k)]-
	sym_score[IndexVect(max2(x-1,0),j,k)];
      if (j==y || k==z)
	grad_x += sym_score[IndexVect(min2(x+1,XDIM-1),j,k)]-
	  sym_score[IndexVect(max2(x-1,0),j,k)];
      if (j==y && k==z)
	grad_x += 2.0f*(sym_score[IndexVect(min2(x+1,XDIM-1),j,k)]-
		       sym_score[IndexVect(max2(x-1,0),j,k)]);
    }
  
  grad_y=0.0;
  for (i=max2(x-1,0); i<=min2(x+1,XDIM-1); i++) 
    for (k=max2(z-1,0); k<=min2(z+1,ZDIM-1); k++) {
      grad_y += sym_score[IndexVect(i,min2(y+1,YDIM-1),k)]-
	sym_score[IndexVect(i,max2(y-1,0),k)];
      if (i==x || k==z)
	grad_y += sym_score[IndexVect(i,min2(y+1,YDIM-1),k)]-
	  sym_score[IndexVect(i,max2(y-1,0),k)];
      if (i==x && k==z)
	grad_y += 2.0f*(sym_score[IndexVect(i,min2(y+1,YDIM-1),k)]-
		       sym_score[IndexVect(i,max2(y-1,0),k)]);
    }
  
  grad_z=0.0;
  for (i=max2(x-1,0); i<=min2(x+1,XDIM-1); i++) 
    for (j=max2(y-1,0); j<=min2(y+1,YDIM-1); j++) { 
      grad_z += sym_score[IndexVect(i,j,min2(z+1,ZDIM-1))]-
	sym_score[IndexVect(i,j,max2(z-1,0))];
      if (i==x || j==y)
	grad_z += sym_score[IndexVect(i,j,min2(z+1,ZDIM-1))]-
	  sym_score[IndexVect(i,j,max2(z-1,0))];
      if (i==x && j==y)
	grad_z += 2.0f*(sym_score[IndexVect(i,j,min2(z+1,ZDIM-1))]-
		       sym_score[IndexVect(i,j,max2(z-1,0))]);
    }
 
  gradient=(float)sqrt(grad_x*grad_x+grad_y*grad_y+grad_z*grad_z);
  return(gradient/16.0f);
}

};
