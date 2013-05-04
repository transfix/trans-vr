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
#define PIE 		  3.1415926f
#define ANGL1		  1.107149f
#define ANGL2		  2.034444f

#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))
#define MAX_TIME 9999999.0f
#define ALPHA	0.1f
#define modifyTime(tx,ty,tz) if (dataset[IndexVect(tx,ty,tz)] < t_low ||\
								(seed_index[IndexVect(tx,ty,tz)] < 254 &&\
							 seed_index[IndexVect(tx,ty,tz)] != index)){\
								boundary = 1;\
							 }else if (time > intens+tdata[IndexVect(tx,ty,tz)]){\
								time=intens+tdata[IndexVect(tx,ty,tz)];\
							 }

namespace SegCapsid {

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

typedef struct CriticalPoint CPNT;
struct CriticalPoint{
	unsigned short x;
	unsigned short y;
	unsigned short z;
	CPNT *next;
};
	
typedef struct {
	int x;
	int y;
	int z;
}INTVEC;

static int min_x,min_y,min_z;
static float min_t;
static MinHeapS* min_heap; //stores our binary tree
static unsigned char *seed_index;  
static float t_low; //lower bound threshold value
static int XDIM, YDIM, ZDIM;
static float *dataset;
static float *tdata; //time data
static VECTOR *FiveFold;

void GetMinimum(void);
void InsertHeap(int, int, int);
void GetTime(void);
void set_index_and_time(int,int,int,unsigned char,float);
VECTOR Rotate(float, float, float, float, float, float, int, int, int);
float GetImgGradient(int, int, int);

/***********************************************************************
We take two initial seed points and a threshold value tlow as user input.
The first seed point must be in the interrior of the virus, the second seed
point must be on the capsid shell.	The threshold value is used as a lower
bound for the density data, all density values below this threshold will 
be thrown out.

Refer to the attached reame file for a further explanation into this algorithm.*/
void SingleCapsidSegment(int xd,int yd,int zd,float *data,float tlow,VECTOR *five_fold,
	int seed_x1, int seed_y1, int seed_z1,
	int seed_x2, int seed_y2, int seed_z2)
{
	int i,j,k;
	
	
	t_low = tlow;
	XDIM = xd;
	YDIM = yd;
	ZDIM = zd;
	dataset = data;
	FiveFold = five_fold;
	
	
	
	min_heap=(MinHeapS*)malloc(sizeof(MinHeapS));
	min_heap->x = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
	min_heap->y = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
	min_heap->z = (unsigned short *)malloc(sizeof(unsigned short)*XDIM*YDIM*ZDIM);
	min_heap->t = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
	tdata = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
	seed_index = (unsigned char *)malloc(sizeof(unsigned char)*XDIM*YDIM*ZDIM);
	
	//initialize seed_index[] and tdata[]
	for (k=0; k<ZDIM; k++)
		for (j=0; j<YDIM; j++) 
			for (i=0; i<XDIM; i++) {
				seed_index[IndexVect(i,j,k)] = 255;
				tdata[IndexVect(i,j,k)] = MAX_TIME;
			}
			
			//Process both of our given initial seed points.
			min_heap->size = 0;
			set_index_and_time(seed_x1,seed_y1,seed_z1,1,0);
			set_index_and_time(seed_x2,seed_y2,seed_z2,2,0);
			
			//The main work is done here.
			while (1){
				GetMinimum();
				
				if (min_t > MAX_TIME || min_heap->size > (unsigned long)(XDIM*YDIM*ZDIM-100))
					break;
				GetTime();
				
			}
			
			/*****************************************************************************
			At this point, our algorithm has completed.  All voxels with seed_index 2 have
			been determined to lie on the capsid shell.*/
			for (k=0; k<ZDIM; k++)
				for (j=0; j<YDIM; j++) 
					for (i=0; i<XDIM; i++) {
						if (seed_index[IndexVect(i,j,k)] != 2)
							dataset[IndexVect(i,j,k)] = 0; 
					}
					
					
					free(min_heap->x);
					free(min_heap->y);
					free(min_heap->z);
					free(min_heap->t);
					free(min_heap);
					free(tdata);
					free(seed_index);
}

	
void set_index_and_time(int ax, int ay,int az, unsigned char index, float t)
{
	int i,j,k;
	int m,n;
	float cx,cy,cz;
	float mx,my,mz;
	float sx,sy,sz;
	float theta,phi;
	VECTOR sv;
	
	
	/***********************************************************************
	(cx,cy,cz) is set to be a point on the first symmetry axis in FiveFold[].  
	We whift the point so that the axis goes through the origin.*/
	cx = FiveFold[0].x-XDIM/2;
	cy = FiveFold[0].y-YDIM/2;
	cz = FiveFold[0].z-ZDIM/2;
	i = ax;
	j = ay;
	k = az;
	/********************************************
	If the seed_index has not previously been set
	then set it equal to the value passed to the algorithm*/
	if (seed_index[IndexVect(i,j,k)] == 255) {
		seed_index[IndexVect(i,j,k)] = index;
		tdata[IndexVect(i,j,k)] = t;
		InsertHeap(i,j,k); 
	}
	/******************************************************************************
	theta and phi are the spherical angles corresponding to the voxel (cx,cy,cz).*/
	theta = (float)atan2(cy,cx);
	phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
	/************************************************************************************
	(i,j,k) is set to be the initial point passed to the function rotated about the first
	5-fold axis.*/
	for (n = 1; n < 5; n++) {
		sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
		i = (int)(sv.x+0.5);
		j = (int)(sv.y+0.5);
		k = (int)(sv.z+0.5);
		/******************************************************************************
		It is possible for the rotation to return a point outside our data set.  If the
		rotation gives us a valid point and if the seed_index has not been previously
		set then we set the seed_index and the tdata.*/
		if(i >= 0 && i < XDIM && j >= 0 && j < YDIM && k >= 0 && k < ZDIM &&
			dataset[IndexVect(i,j,k)] > 0) {
			if (seed_index[IndexVect(i,j,k)] == 255) {
				seed_index[IndexVect(i,j,k)] = index;
				tdata[IndexVect(i,j,k)] = t;
				InsertHeap(i,j,k);
			}
		}
	}
	
	for (m = 1; m < 11; m++) {
		/***********************************************************
		(mx,my,mz) is the m'th 5-fold axis centered at the origin.*/
		mx = FiveFold[m].x-XDIM/2;
		my = FiveFold[m].y-YDIM/2;
		mz = FiveFold[m].z-ZDIM/2;
		/***********************************
		(sx,sy,sz) = (cx,cy,cz)X(mx,my,mz)*/
		sx = mz*cy-my*cz;
		sy = mx*cz-mz*cx;
		sz = my*cx-mx*cy;
		theta = (float)atan2(sy,sx);
		phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
		if (m < 6)
			sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL1,XDIM,YDIM,ZDIM);
		else
			sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL2,XDIM,YDIM,ZDIM);
		/**********************************************************************
		(sx,sy,sz) = (ax,ay,az) rotated about (cx,cy,cz)X(mx,my,mz) by angle.*/
		sx = sv.x;
		sy = sv.y;
		sz = sv.z;
		
		theta = (float)atan2(my,mx);
		phi = (float)atan2(mz, sqrt(mx*mx+my*my));
		for (n = 0; n < 5; n++) {
			sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,XDIM,YDIM,ZDIM);
			/*************************************************************************
			(i,j,k) = (ax,ay,az) rotated about (cx,xy,xz)X(mx,my,mz) by angle and then
			rotated about (mx,my,mz)*/
			i = (int)(sv.x+0.5);
			j = (int)(sv.y+0.5);
			k = (int)(sv.z+0.5);
			if(i >= 0 && i < XDIM && j >= 0 && j < YDIM && k >= 0 && k < ZDIM && 
				dataset[IndexVect(i,j,k)] > 0) {
				if (seed_index[IndexVect(i,j,k)] == 255) {
					seed_index[IndexVect(i,j,k)] = index;
					tdata[IndexVect(i,j,k)] = t;
					InsertHeap(i,j,k); 
				}
			}
		}
		
	}
	
	/********************************
	(mx,my,mz) = second 5-fold axis*/
	mx = FiveFold[1].x-XDIM/2;
	my = FiveFold[1].y-YDIM/2;
	mz = FiveFold[1].z-ZDIM/2;
	/***********************************
	(sx,sy,sz) = (cx,cy,cz)X(mx,my,mz)*/
	sx = mz*cy-my*cz;
	sy = mx*cz-mz*cx;
	sz = my*cx-mx*cy;
	theta = (float)atan2(sy,sx);
	phi = (float)atan2(sz,sqrt(sx*sx+sy*sy));
	sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,PIE,XDIM,YDIM,ZDIM);
	sx = sv.x;
	sy = sv.y;
	sz = sv.z;
	/************************************************************************
	(i,j,k) = (ax,ay,az) rotated about (cx,cy,cz)X(mx,my,mz) by 180 degrees*/
	i = (int)(sx+0.5);
	j = (int)(sy+0.5);
	k = (int)(sz+0.5);
	if(i >= 0 && i < XDIM && j >= 0 && j < YDIM && k >= 0 && k < ZDIM &&
		dataset[IndexVect(i,j,k)] > 0) {
		if (seed_index[IndexVect(i,j,k)] == 255) {
			seed_index[IndexVect(i,j,k)] = index;
			tdata[IndexVect(i,j,k)] = t;
			InsertHeap(i,j,k); 
		}
	}
	/******************************
	(mx,my,mz) = last 5-fold axis*/
	mx = FiveFold[11].x-XDIM/2;
	my = FiveFold[11].y-YDIM/2;
	mz = FiveFold[11].z-ZDIM/2;
	theta = (float)atan2(my,mx);
	phi = (float)atan2(mz, sqrt(mx*mx+my*my));
	for (n = 1; n < 5; n++) {
		sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
		/******************************************************************************************
		(i,j,k) = (cx,cy,cz)X[(cx,cy,cz)X(second 5-fold axis)] rotated about the last 5-fold axis*/
		i = (int)(sv.x+0.5);
		j = (int)(sv.y+0.5);
		k = (int)(sv.z+0.5);
		if (seed_index[IndexVect(i,j,k)] == 255) {
			seed_index[IndexVect(i,j,k)] = index;
			tdata[IndexVect(i,j,k)] = t;
			InsertHeap(i,j,k); 
		}
	}
}

/*****************************************************************************
We set (min_x, min_y, min_z) to be the minimum voxel stored in our binary tree.
We then remove this point and adjust the rest of the tree accordingly.  
The tree is sorted such that the time data for each parent is less than the
time data for its children.  So the minimum voxel is the root of the entire
tree.*/
void GetMinimum(void)
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


/***********************************************************************
This funtion simply inserts the voxel (x,y,z) into the appropriate place
in our binary tree.  The tree is sorted such that the time data for each
parent is less than the time data for its children.*/
void InsertHeap(int x, int y, int z)
{
	int pointer, parent;
	float t;
	
	t = tdata[IndexVect(x,y,z)];
	min_heap->size++;
	pointer=min_heap->size;
	
	while (pointer > 1) {
		/************************************************************
		If time data for the new voxel is less than the time data for
		the current parent, then move the parent down to the child
		position and set pointer to index the empty slot for the
		previous parent*/
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
	/**********************************************************
	Insert the new point into the empty slot left behind by the
	previous parent.*/
	min_heap->x[pointer-1]=x;
	min_heap->y[pointer-1]=y;
	min_heap->z[pointer-1]=z;
	min_heap->t[pointer-1]=t;
	
}



void GetTime(void)
{
	int temp_x, temp_y, temp_z;
	float intens, time;
	char boundary; //boolean flag variable
	unsigned char index;
	
	
	index = seed_index[IndexVect(min_x,min_y,min_z)];
	
	temp_x=max(min_x-1,0);
	temp_y=min_y;
	temp_z=min_z;
	/*****************************************************************
	If seed_index has not been previously set by set_index_and_time()*/
	if (seed_index[IndexVect(temp_x,temp_y,temp_z)] >= 254) {	
		boundary = 0;
		time = (float)(MAX_TIME+1.0);
		intens = GetImgGradient(temp_x, temp_y, temp_z);
		intens = (float)(exp(ALPHA*intens)); //currently ALPHA set to 1.0f
		//intens = EXP[maximum density gradient at (temp_x,temp_y,temp_z)]
		
		/*
		We now calculate the time for voxel (min_x-1,min_y,min_z).
		If (min_x-1,min_y,min_z) is next to a voxel that is below tlow
		or if it is next to a voxel whose index has been set to a value
		other than the index of (min_x,min_y,min_z), then we set the time
		to MAX_TIME.  Otherwise we compute for each voxel next to 
		(min_x-1,min_y,min_z), [intens + timedata for voxel next to (min_x-1,
		min_y,min_z)] and take the minimum of all these values as the time
		for (min_x-1, min_y, min_z).  Then we call set_index_and_time()
		to set the index and time of (min_x-1, min_y, min_z).*/
		modifyTime(max(temp_x-1,0), temp_y, temp_z);
		if(boundary == 0){
			modifyTime(min(temp_x+1,XDIM-1), temp_y, temp_z);
			if(boundary == 0){
				modifyTime(temp_x, max(temp_y-1,0), temp_z);
				if(boundary == 0){
					modifyTime(temp_x, min(temp_y+1,YDIM-1), temp_z);
					if(boundary == 0){
						modifyTime(temp_x, temp_y, max(temp_z-1,0));
						if(boundary == 0){
							modifyTime(temp_x, temp_y, min(temp_z+1,ZDIM-1));
						}
					}
				}
			}
		}	 
		if (boundary == 1){
			time=(float)(MAX_TIME+1.0);
		}
		set_index_and_time(temp_x,temp_y,temp_z,index,time);
	}
	
	/*******************************************************************
	same as above except we calculate the time for (min_x+1,min_y,min_z)
	instead of (min_x-1,min_y,min_z).*/
	temp_x=min(min_x+1,XDIM-1);
	temp_y=min_y;
	temp_z=min_z;
	if (seed_index[IndexVect(temp_x,temp_y,temp_z)] >= 254) {	
		boundary = 0;
		time = (float)(MAX_TIME+1.0);
		intens = GetImgGradient(temp_x, temp_y, temp_z);
		intens = (float)(exp(ALPHA*intens));
		
		modifyTime(max(temp_x-1,0), temp_y, temp_z)
			if(boundary == 0){
				modifyTime(min(temp_x+1,XDIM-1), temp_y, temp_z);
				if(boundary == 0){
					modifyTime(temp_x, max(temp_y-1,0), temp_z);
					if(boundary == 0){
						modifyTime(temp_x, min(temp_y+1,YDIM-1), temp_z);
						if(boundary == 0){
							modifyTime(temp_x, temp_y, max(temp_z-1,0));
							if(boundary == 0){
								modifyTime(temp_x, temp_y, min(temp_z+1,ZDIM-1));
							}
						}
					}
				}
			}	 
			if (boundary == 1){
				time=(float)(MAX_TIME+1.0);
			}
			set_index_and_time(temp_x,temp_y,temp_z,index,time);
	}
	
	temp_y=max(min_y-1,0);
	temp_x=min_x;
	temp_z=min_z;
	if (seed_index[IndexVect(temp_x,temp_y,temp_z)] >= 254) {	
		boundary = 0;
		time = (float)(MAX_TIME+1.0);
		intens = GetImgGradient(temp_x, temp_y, temp_z);
		intens = (float)(exp(ALPHA*intens));
		
		modifyTime(max(temp_x-1,0), temp_y, temp_z);
		if(boundary == 0){
			modifyTime(min(temp_x+1,XDIM-1), temp_y, temp_z);
			if(boundary == 0){
				modifyTime(temp_x, max(temp_y-1,0), temp_z);
				if(boundary == 0){
					modifyTime(temp_x, min(temp_y+1,YDIM-1), temp_z);
					if(boundary == 0){
						modifyTime(temp_x, temp_y, max(temp_z-1,0));
						if(boundary == 0){
							modifyTime(temp_x, temp_y, min(temp_z+1,ZDIM-1));
						}
					}
				}
			}
		}	 
		if (boundary == 1){
			time=(float)(MAX_TIME+1.0);
		}
		set_index_and_time(temp_x,temp_y,temp_z,index,time);
	}
	
	temp_y=min(min_y+1,YDIM-1);
	temp_x=min_x;
	temp_z=min_z;
	if (seed_index[IndexVect(temp_x,temp_y,temp_z)] >= 254) {	
		boundary = 0;
		time = (float)(MAX_TIME+1.0);
		intens = GetImgGradient(temp_x, temp_y, temp_z);
		intens = (float)(exp(ALPHA*intens));
		
		modifyTime(max(temp_x-1,0), temp_y, temp_z);
		if(boundary == 0){
			modifyTime(min(temp_x+1,XDIM-1), temp_y, temp_z);
			if(boundary == 0){
				modifyTime(temp_x, max(temp_y-1,0), temp_z);
				if(boundary == 0){
					modifyTime(temp_x, min(temp_y+1,YDIM-1), temp_z);
					if(boundary == 0){
						modifyTime(temp_x, temp_y, max(temp_z-1,0));
						if(boundary == 0){
							modifyTime(temp_x, temp_y, min(temp_z+1,ZDIM-1));
						}
					}
				}
			}
		}	 
		if (boundary == 1){
			time=(float)(MAX_TIME+1.0);
		}
		set_index_and_time(temp_x,temp_y,temp_z,index,time);
	}
	
	temp_z=max(min_z-1,0);
	temp_x=min_x;
	temp_y=min_y;
	if (seed_index[IndexVect(temp_x,temp_y,temp_z)] >= 254) {	
		boundary = 0;
		time = (float)(MAX_TIME+1.0);
		intens = GetImgGradient(temp_x, temp_y, temp_z);
		intens = (float)(exp(ALPHA*intens));
		
		modifyTime(max(temp_x-1,0), temp_y, temp_z);
		if(boundary == 0){
			modifyTime(min(temp_x+1,XDIM-1), temp_y, temp_z);
			if(boundary == 0){
				modifyTime(temp_x, max(temp_y-1,0), temp_z);
				if(boundary == 0){
					modifyTime(temp_x, min(temp_y+1,YDIM-1), temp_z);
					if(boundary == 0){
						modifyTime(temp_x, temp_y, max(temp_z-1,0));
						if(boundary == 0){
							modifyTime(temp_x, temp_y, min(temp_z+1,ZDIM-1));
						}
					}
				}
			}
		}	 
		if (boundary == 1){
			time=(float)(MAX_TIME+1.0);
		}
		set_index_and_time(temp_x,temp_y,temp_z,index,time);
	}
	
	temp_z=min(min_z+1,ZDIM-1);
	temp_x=min_x;
	temp_y=min_y;
	if (seed_index[IndexVect(temp_x,temp_y,temp_z)] >= 254) {	
		boundary = 0;
		time = (float)(MAX_TIME+1.0);
		intens = GetImgGradient(temp_x, temp_y, temp_z);
		intens = (float)(exp(ALPHA*intens));
		
		modifyTime(max(temp_x-1,0), temp_y, temp_z);
		if(boundary == 0){
			modifyTime(min(temp_x+1,XDIM-1), temp_y, temp_z);
			if(boundary == 0){
				modifyTime(temp_x, max(temp_y-1,0), temp_z);
				if(boundary == 0){
					modifyTime(temp_x, min(temp_y+1,YDIM-1), temp_z);
					if(boundary == 0){
						modifyTime(temp_x, temp_y, max(temp_z-1,0));
						if(boundary == 0){
							modifyTime(temp_x, temp_y, min(temp_z+1,ZDIM-1));
						}
					}
				}
			}
		}	 
		if (boundary == 1){
			time=(float)(MAX_TIME+1.0);
		}
		set_index_and_time(temp_x,temp_y,temp_z,index,time);
	}
	
}

};
