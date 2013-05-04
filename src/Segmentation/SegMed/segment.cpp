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

#define max(x, y) ((x>y) ? (x):(y))
#define min(x, y) ((x<y) ? (x):(y))

#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))
#define MAX_TIME 9999999.0f
#define ALPHA    0.2f
#define BETA     0.5f

namespace SegMed {

  typedef struct {
    int *x;
    int *y;
    int *z;
    float *t;
    int size;
  }MinHeapS;

  typedef struct {
    float x;
    float y;
    float z;
  }VECTOR;

  typedef struct CriticalPoint CPNT;
  struct CriticalPoint{
    int x;
    int y;
    int z;
    CPNT *next;
  };

  typedef struct {
    int x;
    int y;
    int z;
  }INTVEC;


  typedef struct {
    unsigned short x;
    unsigned short y;
    unsigned short z;
  }SVECTOR;


  static int min_x,min_y,min_z;
  static float min_t;
  static MinHeapS* T_min_heap;
  static unsigned char T_index;
  static unsigned char *seed_index;
  static float t_low;
  static int XDIM, YDIM, ZDIM;
  static float *dataset;
  static float *T_tdata;
  static float *ImgGrad;
  static CPNT *critical_start;
  static int SeedNum;
  static int PhaseNum;
  static int start_ptr,end_ptr;
  static INTVEC *IndexArray;

  static SVECTOR *stack;
  

  void T_GetMinimum(void);
  void T_InsertHeap(int, int, int);
  void T_GetTime(void);
  unsigned char SearchIndex(int*, int*, int*);
  void InitialTime(int, int, int);
  void RegionGrowing(int, int, int, int);


  void Segment(int xd, int yd, int zd, float *data, float *edge_mag, 
	       unsigned char *result, CPNT *critical,float tlow)
  {
    int i,j,k;
//    float MaxSeed;
    int MaxSeed_x, MaxSeed_y, MaxSeed_z;
//    unsigned char index;
//    CPNT *critical_tmp;
    unsigned long step;
//    int x,y,z;
//    int num1,num2;


    XDIM = xd;
    YDIM = yd;
    ZDIM = zd;
    dataset = data;
    ImgGrad = edge_mag;
    critical_start = critical;
    t_low = tlow;
    seed_index = result;


    T_min_heap=(MinHeapS*)malloc(sizeof(MinHeapS));
    T_min_heap->x = (int *)malloc(sizeof(int)*XDIM*YDIM*ZDIM);
    T_min_heap->y = (int *)malloc(sizeof(int)*XDIM*YDIM*ZDIM);
    T_min_heap->z = (int *)malloc(sizeof(int)*XDIM*YDIM*ZDIM);
    T_min_heap->t = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
    T_tdata = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
    stack = (SVECTOR*)malloc(sizeof(SVECTOR)*XDIM*YDIM*ZDIM);
  

    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++) {
	  seed_index[IndexVect(i,j,k)] = 255;
	  T_tdata[IndexVect(i,j,k)] = MAX_TIME;
	}

    /*
      SeedNum = 0;
      critical_tmp = critical_start;
      while (critical_tmp != NULL) {
      i = critical_tmp->x;
      j = critical_tmp->y;
      k = critical_tmp->z;
      seed_index[IndexVect(i,j,k)] = 254;
    
      SeedNum++;
      critical_tmp = critical_tmp->next;
      }
	
      printf("SeedNum = %d....\n",SeedNum);
    */

    PhaseNum = 1;

    T_min_heap->size = 0;
  
  
    MaxSeed_x = 179;
    MaxSeed_y = 53;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=1;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 224;
    MaxSeed_y = 100;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=2;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 106;
    MaxSeed_y = 59;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=3;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 96;
    MaxSeed_y = 101;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=3;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 145;
    MaxSeed_y = 89;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=2;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 150;
    MaxSeed_y = 135;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=4;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 187;
    MaxSeed_y = 188;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=2;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 87;
    MaxSeed_y = 123;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=4;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
  
    MaxSeed_x = 135;
    MaxSeed_y = 164;
    MaxSeed_z = 210;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=11;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
  
    MaxSeed_x = 133;
    MaxSeed_y = 167;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=11;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
 

    MaxSeed_x = 147;
    MaxSeed_y = 18;
    MaxSeed_z = 185;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=11;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  
    MaxSeed_x = 131;
    MaxSeed_y = 158;
    MaxSeed_z = 320;
    seed_index[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)]=11;
    /*RegionGrowing(MaxSeed_x,MaxSeed_y,MaxSeed_z,1);*/
    T_min_heap->x[T_min_heap->size]=MaxSeed_x;
    T_min_heap->y[T_min_heap->size]=MaxSeed_y;
    T_min_heap->z[T_min_heap->size]=MaxSeed_z;
    T_min_heap->t[T_min_heap->size]=0.0;
    T_min_heap->size ++;
    T_tdata[IndexVect(MaxSeed_x,MaxSeed_y,MaxSeed_z)] = 0.0;
  

    printf("SeedNum = %d....\n",SeedNum);
  
    /*
      IndexArray = (INTVEC*)malloc(sizeof(INTVEC)*XDIM*YDIM*ZDIM);
  
      while (SeedNum > 0) {

      index = SearchIndex(&i,&j,&k);
      if (index == 255) 
      break;
    
      RegionGrowing(i,j,k,index);

      if (SeedNum%100 == 0)
      printf("SeedNum = %d,  Index = %d \n",SeedNum,index);
    
      }
  

      printf("\n\nBegin the fast marching ....\n");
      T_min_heap->size = 0;
      for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
      T_tdata[IndexVect(i,j,k)] = MAX_TIME;
      if (seed_index[IndexVect(i,j,k)] < 254) {
      T_min_heap->x[T_min_heap->size]=i;
      T_min_heap->y[T_min_heap->size]=j;
      T_min_heap->z[T_min_heap->size]=k;
      T_min_heap->t[T_min_heap->size]=0.0;
      T_min_heap->size ++;
      T_tdata[IndexVect(i,j,k)] = 0.0;
      }
      }
    */
  
    PhaseNum = 2;
    step = 0;
    while (1){
    
      T_GetMinimum();
      if (step % 100000 == 0)
	printf("step: %lu;  min_t: %f \n",step,min_t);

      if (min_t >= MAX_TIME)
	break;
      T_GetTime();
    
      step++;
    }
  
    /*
      for (z=0; z<ZDIM; z++)
      for (y=0; y<YDIM; y++) 
      for (x=0; x<XDIM; x++) {
      if (seed_index[IndexVect(x,y,z)] == 2) {
      num1 = 0;
      num2 = 0;
      for (k=max(z-2,0); k<=min(z+2,ZDIM-1); k++) 
      for (j=max(y-2,0); j<=min(y+2,YDIM-1); j++) 
      for (i=max(x-2,0); i<=min(x+2,XDIM-1); i++) {
      if ((i-x)*(i-x)+(j-y)*(j-y)+(k-z)*(k-z) <= 4) {
      if (seed_index[IndexVect(i,j,k)] == 1) 
      num1 ++;
      else if (seed_index[IndexVect(i,j,k)] == 2) 
      num2 ++;
      }
      }
      if (num1 > num2)
      seed_index[IndexVect(x,y,z)] = 11;
      }
      }

      for (z=0; z<ZDIM; z++)
      for (y=0; y<YDIM; y++) 
      for (x=0; x<XDIM; x++) {
      if (seed_index[IndexVect(x,y,z)] == 11) 
      seed_index[IndexVect(x,y,z)] = 1;
      }
    */
  
    free(T_min_heap->x);
    free(T_min_heap->y);
    free(T_min_heap->z);
    free(T_min_heap->t);
    free(T_min_heap);
    free(T_tdata);
    free(IndexArray);
  }




  void RegionGrowing(int tx, int ty, int tz, int index)
  {
    int i,j,k;
    int x,y,z;
    long stack_size;
    float min_inten,max_inten;

  
    min_inten = dataset[IndexVect(tx,ty,tz)]-0.5f;
    max_inten = dataset[IndexVect(tx,ty,tz)]+0.5f;
  
    stack[0].x = tx;
    stack[0].y = ty;
    stack[0].z = tz;
  
    stack_size = 1;
  
    while (stack_size > 0) {
      stack_size--;
      x = stack[stack_size].x;
      y = stack[stack_size].y;
      z = stack[stack_size].z;

      for (k=max(z-1,0); k<=min(z+1,ZDIM-1); k++) 
	for (j=max(y-1,0); j<=min(y+1,YDIM-1); j++) 
	  for (i=max(x-1,0); i<=min(x+1,XDIM-1); i++) {
	    if (dataset[IndexVect(i,j,k)] >= min_inten &&
		dataset[IndexVect(i,j,k)] <= max_inten &&
		seed_index[IndexVect(i,j,k)] >= 254) {
	      if (seed_index[IndexVect(i,j,k)] == 254)
		SeedNum--;
	      seed_index[IndexVect(i,j,k)] = index;
	    
	      stack[stack_size].x = i;
	      stack[stack_size].y = j;
	      stack[stack_size].z = k;
	      stack_size++;
	    }
	  }
    }   
  }



  void InitialTime(int xx, int yy, int zz)
  {
    int i,j,k;
    int x, y, z;

  
    T_tdata[IndexVect(xx,yy,zz)] = MAX_TIME;
    start_ptr = 0;
    end_ptr = 1;
    IndexArray[0].x = xx;
    IndexArray[0].y = yy;
    IndexArray[0].z = zz;
  
    while (start_ptr < end_ptr) {
      x = IndexArray[start_ptr].x;
      y = IndexArray[start_ptr].y;
      z = IndexArray[start_ptr].z;
      start_ptr ++;
	
      for (k=max(z-1,0); k<=min(z+1,ZDIM-1); k++) 
	for (j=max(y-1,0); j<=min(y+1,YDIM-1); j++)
	  for (i=max(x-1,0); i<=min(x+1,XDIM-1); i++) {
	    if (T_tdata[IndexVect(i,j,k)] != MAX_TIME) {
	      T_tdata[IndexVect(i,j,k)] = MAX_TIME;
	      IndexArray[end_ptr].x = i;
	      IndexArray[end_ptr].y = j;
	      IndexArray[end_ptr].z = k;
	      end_ptr ++;
	    }
	  }
    }
  }





  unsigned char SearchIndex(int *m, int *n, int *l)
  {
    int i,j,k;
    int x0,y0,z0;
    int x1,y1,z1;
    float t;
    CPNT *critical_tmp;
    CPNT *critical_prv;
  
  
    critical_tmp = critical_start;
    while (critical_tmp != NULL) {
      i = critical_tmp->x;
      j = critical_tmp->y;
      k = critical_tmp->z;
      InitialTime(i,j,k);
      critical_tmp = critical_tmp->next;
    }

  

    T_min_heap->size = 0;
    critical_tmp = critical_start;
    critical_prv = critical_start;
    while (critical_tmp != NULL) {
      i = critical_tmp->x;
      j = critical_tmp->y;
      k = critical_tmp->z;
    
      if (seed_index[IndexVect(i,j,k)] == 254) {
	T_min_heap->x[T_min_heap->size]=i;
	T_min_heap->y[T_min_heap->size]=j;
	T_min_heap->z[T_min_heap->size]=k;
	T_min_heap->t[T_min_heap->size]=0.0;
	T_tdata[IndexVect(i,j,k)]=0.0;
	T_min_heap->size ++;

	if (critical_tmp != critical_start)
	  critical_prv = critical_prv->next;
      }
      else {
	if (critical_tmp == critical_start) {
	  critical_start = critical_start->next;
	  critical_prv = critical_start;
	}
	else 
	  critical_prv->next = critical_tmp->next;
      }
    
      critical_tmp = critical_tmp->next;
    }
  
    T_index = 255;
    while (1){
    
      T_GetMinimum();
    
      if (min_t >= MAX_TIME) 
	return(255);
    
      T_GetTime();
    
      if (T_index < 254) 
	break;
    }
  
  
    x0 = min_x;
    y0 = min_y;
    z0 = min_z;
    x1 = x0;
    y1 = y0;
    z1 = z0;
    t = T_tdata[IndexVect(x0,y0,z0)]; 
  
    while (t > 0) {
    
      for (k=max(0,z0-1); k<=min(ZDIM-1,z0+1); k++) 
	for (j=max(0,y0-1); j<=min(YDIM-1,y0+1); j++) 
	  for (i=max(0,x0-1); i<=min(XDIM-1,x0+1); i++) {
	    if (T_tdata[IndexVect(i,j,k)] < t) {
	      t = T_tdata[IndexVect(i,j,k)];
	      x1 = i;
	      y1 = j;
	      z1 = k;
	    }
	  }
      if (x0 == x1 && y0 == y1 && z0 == z1) 
	return(255);
      else {
	x0 = x1;
	y0 = y1;
	z0 = z1;
	t = T_tdata[IndexVect(x0,y0,z0)]; 
      }
    }
    (*m) = x0;
    (*n) = y0;
    (*l) = z0;
    return(T_index);

  }


   
  void T_GetMinimum(void)
  {
    int pointer, left, right;
    int x, y, z;
    float t;

    min_x = T_min_heap->x[0];
    min_y = T_min_heap->y[0];
    min_z = T_min_heap->z[0];
    min_t = T_min_heap->t[0];
  
    x=T_min_heap->x[T_min_heap->size-1];
    y=T_min_heap->y[T_min_heap->size-1];
    z=T_min_heap->z[T_min_heap->size-1];
    t=T_min_heap->t[T_min_heap->size-1];
  
    T_min_heap->size--;
  
    pointer=1;
    while (pointer <= T_min_heap->size/2) {
      left=2*pointer;
      right=2*pointer+1;
      if ((T_min_heap->t[left-1] <= T_min_heap->t[right-1]) && (T_min_heap->t[left-1] < t)) {
	T_min_heap->x[pointer-1]=T_min_heap->x[left-1];
	T_min_heap->y[pointer-1]=T_min_heap->y[left-1];
	T_min_heap->z[pointer-1]=T_min_heap->z[left-1];
	T_min_heap->t[pointer-1]=T_min_heap->t[left-1];
	pointer=left;
      }
      else if ((T_min_heap->t[left-1] > T_min_heap->t[right-1]) && (T_min_heap->t[right-1] < t)){
	T_min_heap->x[pointer-1]=T_min_heap->x[right-1];
	T_min_heap->y[pointer-1]=T_min_heap->y[right-1];
	T_min_heap->z[pointer-1]=T_min_heap->z[right-1];
	T_min_heap->t[pointer-1]=T_min_heap->t[right-1];
	pointer=right;
      }
      else break;
    }

    T_min_heap->x[pointer-1]=x;
    T_min_heap->y[pointer-1]=y;
    T_min_heap->z[pointer-1]=z;
    T_min_heap->t[pointer-1]=t;

  }
    
  void T_InsertHeap(int x, int y, int z)
  {
    int pointer, parent;
    float t;

    t = T_tdata[IndexVect(x,y,z)];
    T_min_heap->size++;
    pointer=T_min_heap->size;

    while (pointer > 1) {
      if (pointer%2 == 0) {
	parent=pointer/2;
	if (t < T_min_heap->t[parent-1]) {
	  T_min_heap->x[pointer-1]=T_min_heap->x[parent-1];
	  T_min_heap->y[pointer-1]=T_min_heap->y[parent-1];
	  T_min_heap->z[pointer-1]=T_min_heap->z[parent-1];
	  T_min_heap->t[pointer-1]=T_min_heap->t[parent-1];
	  pointer=parent;
	}
	else break;
      }
      else if (pointer%2 == 1){
	parent=(pointer-1)/2;
	if (t < T_min_heap->t[parent-1]) {
	  T_min_heap->x[pointer-1]=T_min_heap->x[parent-1];
	  T_min_heap->y[pointer-1]=T_min_heap->y[parent-1];
	  T_min_heap->z[pointer-1]=T_min_heap->z[parent-1];
	  T_min_heap->t[pointer-1]=T_min_heap->t[parent-1];
	  pointer=parent;
	}
	else break;
      }
    }

    T_min_heap->x[pointer-1]=x;
    T_min_heap->y[pointer-1]=y;
    T_min_heap->z[pointer-1]=z;
    T_min_heap->t[pointer-1]=t;

  }



  void T_GetTime(void)
  {
    int tempt_x, tempt_y, tempt_z;
    int tx, ty, tz;
    float intens, value;
    char boundary;
    unsigned char index=0;

     

    if (PhaseNum == 2)
      index = seed_index[IndexVect(min_x,min_y,min_z)];
  
    tempt_x=max(min_x-1,0);
    tempt_y=min_y;
    tempt_z=min_z;
    if (PhaseNum == 1 &&
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] < 254) {
      T_index = seed_index[IndexVect(tempt_x,tempt_y,tempt_z)];
      return;
    }
    else if (T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]==MAX_TIME) {   
    
      value=MAX_TIME;
      boundary = 0;

      tx=max(tempt_x-1,0);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tx=min(tempt_x+1,XDIM-1);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=max(tempt_y-1,0);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=min(tempt_y+1,YDIM-1);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=max(tempt_z-1,0);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=min(tempt_z+1,ZDIM-1);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      if (boundary == 1) 
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=MAX_TIME+1;
      else
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=value;
    
      if (PhaseNum == 2)
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] = index;

      T_InsertHeap(tempt_x,tempt_y,tempt_z);   
    
    }
  
    tempt_x=min(min_x+1,XDIM-1);
    tempt_y=min_y;
    tempt_z=min_z;
    if (PhaseNum == 1 &&
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] < 254) {
      T_index = seed_index[IndexVect(tempt_x,tempt_y,tempt_z)];
      return;
    }
    else if (T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]==MAX_TIME) {
    
      value=MAX_TIME;
      boundary = 0;

      tx=max(tempt_x-1,0);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tx=min(tempt_x+1,XDIM-1);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=max(tempt_y-1,0);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=min(tempt_y+1,YDIM-1);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=max(tempt_z-1,0);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=min(tempt_z+1,ZDIM-1);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      if (boundary == 1) 
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=MAX_TIME+1;
      else
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=value;
    
      if (PhaseNum == 2)
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] = index;

      T_InsertHeap(tempt_x,tempt_y,tempt_z);  
    
    }

    tempt_y=max(min_y-1,0);
    tempt_x=min_x;
    tempt_z=min_z;
    if (PhaseNum == 1 &&
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] < 254) {
      T_index = seed_index[IndexVect(tempt_x,tempt_y,tempt_z)];
      return;
    }
    else if (T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]==MAX_TIME) {
    
      value=MAX_TIME;
      boundary = 0;

      tx=max(tempt_x-1,0);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tx=min(tempt_x+1,XDIM-1);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=max(tempt_y-1,0);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=min(tempt_y+1,YDIM-1);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=max(tempt_z-1,0);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=min(tempt_z+1,ZDIM-1);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      if (boundary == 1) 
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=MAX_TIME+1;
      else
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=value;
    
      if (PhaseNum == 2)
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] = index;

      T_InsertHeap(tempt_x,tempt_y,tempt_z);  
    
    }

    tempt_y=min(min_y+1,YDIM-1);
    tempt_x=min_x;
    tempt_z=min_z;
    if (PhaseNum == 1 &&
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] < 254) {
      T_index = seed_index[IndexVect(tempt_x,tempt_y,tempt_z)];
      return;
    }
    else if (T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]==MAX_TIME) {
    
      value=MAX_TIME;
      boundary = 0;

      tx=max(tempt_x-1,0);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tx=min(tempt_x+1,XDIM-1);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=max(tempt_y-1,0);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=min(tempt_y+1,YDIM-1);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=max(tempt_z-1,0);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=min(tempt_z+1,ZDIM-1);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      if (boundary == 1) 
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=MAX_TIME+1;
      else
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=value;
    
      if (PhaseNum == 2)
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] = index;

      T_InsertHeap(tempt_x,tempt_y,tempt_z);  
    
    }
  
    tempt_z=max(min_z-1,0);
    tempt_x=min_x;
    tempt_y=min_y;
    if (PhaseNum == 1 &&
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] < 254) {
      T_index = seed_index[IndexVect(tempt_x,tempt_y,tempt_z)];
      return;
    }
    else if (T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]==MAX_TIME) {
    
      value=MAX_TIME;
      boundary = 0;

      tx=max(tempt_x-1,0);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tx=min(tempt_x+1,XDIM-1);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=max(tempt_y-1,0);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=min(tempt_y+1,YDIM-1);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=max(tempt_z-1,0);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=min(tempt_z+1,ZDIM-1);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      if (boundary == 1) 
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=MAX_TIME+1;
      else
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=value;
    
      if (PhaseNum == 2)
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] = index;

      T_InsertHeap(tempt_x,tempt_y,tempt_z);  
    
    }

    tempt_z=min(min_z+1,ZDIM-1);
    tempt_x=min_x;
    tempt_y=min_y;
    if (PhaseNum == 1 &&
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] < 254) {
      T_index = seed_index[IndexVect(tempt_x,tempt_y,tempt_z)];
      return;
    }
    else if (T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]==MAX_TIME) {
    
      value=MAX_TIME;
      boundary = 0;

      tx=max(tempt_x-1,0);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tx=min(tempt_x+1,XDIM-1);
      ty=tempt_y;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=max(tempt_y-1,0);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      ty=min(tempt_y+1,YDIM-1);
      tx=tempt_x;
      tz=tempt_z;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=max(tempt_z-1,0);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      tz=min(tempt_z+1,ZDIM-1);
      tx=tempt_x;
      ty=tempt_y;
      if (dataset[IndexVect(tx,ty,tz)] < t_low ||
	  (PhaseNum == 2 &&
	   seed_index[IndexVect(tx,ty,tz)] < 254 &&
	   seed_index[IndexVect(tx,ty,tz)] != index)) 
	boundary = 1;
      else {
	intens=0.01f*(float)exp(ALPHA*ImgGrad[IndexVect(tx,ty,tz)]+ 
			BETA*fabs(dataset[IndexVect(tx,ty,tz)]-
				  dataset[IndexVect(tempt_x,tempt_y,tempt_z)]));
	if (value > intens+T_tdata[IndexVect(tx,ty,tz)])
	  value=intens+T_tdata[IndexVect(tx,ty,tz)];
      }

      if (boundary == 1) 
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=MAX_TIME+1;
      else
	T_tdata[IndexVect(tempt_x,tempt_y,tempt_z)]=value;
    
      if (PhaseNum == 2)
	seed_index[IndexVect(tempt_x,tempt_y,tempt_z)] = index;

      T_InsertHeap(tempt_x,tempt_y,tempt_z);  
    
    }

  }

};
