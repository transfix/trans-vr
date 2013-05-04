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
#include <vector>
#include <stack>


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
/*-Jesse-***************************************************************
min_heap[] is an array that stores a binary tree.  Each node in the tree 
is a voxel and the tree is sorted such that the time data of each parent 
is less than the time data of its children.*/
static MinHeapS* min_heap;
/*-Jesse-*******************************************************
seed_index[] is an array that stores a seed value for each voxel.  
The seed value is used to designate each voxel as being a member
of a segmented unit*/
static unsigned char *seed_index;  
static float t_low; //lower bound threshold value
static int XDIM, YDIM, ZDIM;
static float *dataset;
/*-Jesse-**************************************************************
tdata[] is an array that stores time data for each voxel.  This is used 
in our implementation of the fast marching algorithm.*/
static float *tdata;
static VECTOR *FiveFold;

void GetMinimum(void);
void InsertHeap(int, int, int);
void GetTime(void);
void set_index_and_time(int,int,int,unsigned char,float);
VECTOR Rotate(float, float, float, float, float, float, int, int, int);
//VECTOR RotateDouble(double, double, double, double, double, double);
float GetImgGradient(int, int, int);
void bubbleStopperSingle(unsigned char *segmentIndex, unsigned int maxVoxelRadius);

/*-Jesse-***************************************************************
We take two initial seed points and a threshold value tlow as user input.
The first seed point must be in the interrior of the virus, the second seed
point must be on the capsid shell.	The threshold value is used as a lower
bound for the density data, density values below this threshold will 
be thrown out.

Refer to the attached readme file for a further explanation into this algorithm.*/
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

	/*-Jesse-************************************************************
	Mark all voxels with density below the minimum threshold for deletion.
	Then run bubbleStopper to save some of the voxels that we may want to keep.  
	Then delete the voxels that are still marked and proceed with the rest of the
	algorithm.*/
	for (k=0; k<ZDIM; k++){
		for (j=0; j<YDIM; j++){
			for (i=0; i<XDIM; i++) {
				if(dataset[IndexVect(i,j,k)] < tlow){
					seed_index[IndexVect(i,j,k)] = 0;
				}else
					seed_index[IndexVect(i,j,k)] = 2;
			}
		}
	}
//	bubbleStopperSingle(seed_index, 5);
	for (k=0; k<ZDIM; k++){
		for (j=0; j<YDIM; j++){
			for (i=0; i<XDIM; i++){
				if (seed_index[IndexVect(i,j,k)] != 2)
					dataset[IndexVect(i,j,k)] = 0; 
			}
		}
	}
	/*-Jesse- Density values below tlow have been deleted, we now
	proceed with the rest of the algorithm.
	**************************************/

	/*-Jesse-***************************
	initialize seed_index[] and tdata[]*/
	for (k=0; k<ZDIM; k++){
		for (j=0; j<YDIM; j++){
			for (i=0; i<XDIM; i++) {
				seed_index[IndexVect(i,j,k)] = 255;
				tdata[IndexVect(i,j,k)] = MAX_TIME;
			}
		}
	}

	/*-Jesse-**************************************
	Process both of our given initial seed points.*/
	min_heap->size = 0;
	set_index_and_time(seed_x1,seed_y1,seed_z1,1,0);
	set_index_and_time(seed_x2,seed_y2,seed_z2,2,0);

	/*-Jesse-************************************************************************
	The main work is done here.  We repeatedly call GetMinimum() and GetTime() untill 
	either the voxel with the lowest tdata has a value greater than MAX_TIME or the 
	size of our min_heap[] grows too large.*/
	while (1){
		GetMinimum();
		if (min_t > MAX_TIME || min_heap->size > (unsigned long)(XDIM*YDIM*ZDIM-100)){
			break;
		}
		GetTime();
	}

	free(min_heap->x);
	free(min_heap->y);
	free(min_heap->z);
	free(min_heap->t);
	free(min_heap);
	free(tdata);

	/*-Jesse-***********************************************************
	This call to bubbleStopper can probably be removed, but I havn't yet
	tested it.*/
//	bubbleStopperSingle(seed_index, 5);

	/*-Jesse-*********************************************************************
	At this point, our algorithm has completed.  All voxels with seed_index 2 have
	been determined to lie on the capsid shell.*/
	for (k=0; k<ZDIM; k++){
		for (j=0; j<YDIM; j++){
			for (i=0; i<XDIM; i++){
				if (seed_index[IndexVect(i,j,k)] != 2)
					dataset[IndexVect(i,j,k)] = 0; 
			}
		}
	}
	free(seed_index);
}

/*-Jesse-**************************************************************************
set_index_and_time() takes the voxel passed to it, (ax,ay,az), and if its seedIndex 
has not previously been set then we set its seedIndex and tdata to the corresponding 
values passed to the function and insert the point into the heap.  We then take 
advantage of the capsid's icosahedron properties and compute the other 59 symmetrical 
points.  For each of these points if its seedIndex has not been set then we set its 
seedIndex and tdata to the corresponding values passed to the function, and insert 
into the heap.*/
void set_index_and_time(int ax, int ay,int az, unsigned char index, float t)
{
	int i,j,k;
	int m,n;
	double cx,cy,cz;
	double mx,my,mz;
	double sx,sy,sz;
	double theta,phi;
	VECTOR sv;
	
	
	/*-Jesse-***************************************************************
	(cx,cy,cz) is set to be a point on the first symmetry axis in FiveFold[].  
	We whift the point so that the axis goes through the origin.*/
	cx = FiveFold[0].x-XDIM/2;
	cy = FiveFold[0].y-YDIM/2;
	cz = FiveFold[0].z-ZDIM/2;
	i = ax;
	j = ay;
	k = az;
	/*-Jesse-*******************************************************************
	If the seed_index has not previously been set, then set seed_index and tdata
	to the appropriate values passed to the algorithm.  Then insert into the heap.*/
	if (seed_index[IndexVect(i,j,k)] == 255) {
		seed_index[IndexVect(i,j,k)] = index;
		tdata[IndexVect(i,j,k)] = t;
		InsertHeap(i,j,k); 
	}
	/*-Jesse-***********************************************************************
	theta and phi are the spherical angles corresponding (cx,cy,cz) being treated as
	a voxel instead of an axis of rotation.*/
	theta = atan2(cy,cx);
	phi = atan2(cz, sqrt(cx*cx+cy*cy));
	/*-Jesse-**********************************************************************
	(i,j,k) is set to be the initial point passed to the function rotated about the
	5-fold axis stored in (cx,cy,cz).*/
	for (n = 1; n < 5; n++) {
		sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,n*2.0f*PIE/5.0f, XDIM, YDIM, ZDIM);
		i = (int)(sv.x+0.5);
		j = (int)(sv.y+0.5);
		k = (int)(sv.z+0.5);
		/*-Jesse-**********************************************************************
		It is possible for the rotation to return a point outside our data set.  If the
		rotation gives us a valid point and if the seed_index has not been previously
		set then we set seed_index, tdata, and insert into the heap.*/
		if(i >= 0 && i < XDIM && j >= 0 && j < YDIM && k >= 0 && k < ZDIM &&
			dataset[IndexVect(i,j,k)] > 0) {
			if (seed_index[IndexVect(i,j,k)] == 255) {
				seed_index[IndexVect(i,j,k)] = index;
				tdata[IndexVect(i,j,k)] = t;
				InsertHeap(i,j,k);
			}
		}
	}
	
	/*-Jesse-*************************************************************************
	We now make use of the icosahedral symmetry to find the 59 points symmetric to the
	point passed to set_index_and_time().  We have already found 4 of these points by
	rotating about (cx,cy,cz) above.  We now wish to translate our point to the other
	axes of rotation and then rotate about each axis.
	
	In detail, suppose we have a voxel V1 that we have just rotated about axis A1 and 
	we now wish to rotate about another axis A2.  We first find the cross product of 
	A1XA2 and the angle between A1 and A2.  We rotate the voxel V1 about A1XA2 by the 
	angle and we get a corresponding voxel V2 in relation to A2.  We can then rotate V2 
	about A2.  We then test if the resulting voxels are above tlow, if they are then we 
	set their densities to be negative.*/
	for (m = 1; m < 11; m++) {
		/*-Jesse-***************************************************
		(mx,my,mz) is the m'th 5-fold axis centered at the origin.*/
		mx = FiveFold[m].x-XDIM/2;
		my = FiveFold[m].y-YDIM/2;
		mz = FiveFold[m].z-ZDIM/2;
		/*-Jesse-***************************
		(sx,sy,sz) = (cx,cy,cz)X(mx,my,mz)*/
		sx = mz*cy-my*cz;
		sy = mx*cz-mz*cx;
		sz = my*cx-mx*cy;
		theta = atan2(sy,sx);
		phi = atan2(sz,sqrt(sx*sx+sy*sy));
		if (m < 6)
			sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL1,XDIM,YDIM,ZDIM);
		else
			sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL2,XDIM,YDIM,ZDIM);
		/*-Jesse-**************************************************************
		(sx,sy,sz) = (ax,ay,az) rotated about (cx,cy,cz)X(mx,my,mz) by angle.*/
		sx = sv.x;
		sy = sv.y;
		sz = sv.z;
		
		theta = atan2(my,mx);
		phi = atan2(mz, sqrt(mx*mx+my*my));
		for (n = 0; n < 5; n++) {
			sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,XDIM,YDIM,ZDIM);
			/*-Jesse-*****************************************************************
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
	
	/*-Jesse-************************
	(mx,my,mz) = second 5-fold axis*/
	mx = FiveFold[1].x-XDIM/2;
	my = FiveFold[1].y-YDIM/2;
	mz = FiveFold[1].z-ZDIM/2;
	/*-Jesse-***************************
	(sx,sy,sz) = (cx,cy,cz)X(mx,my,mz)*/
	sx = mz*cy-my*cz;
	sy = mx*cz-mz*cx;
	sz = my*cx-mx*cy;
	theta = atan2(sy,sx);
	phi = atan2(sz,sqrt(sx*sx+sy*sy));
	sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,PIE,XDIM,YDIM,ZDIM);
	sx = sv.x;
	sy = sv.y;
	sz = sv.z;
	/*-Jesse-****************************************************************
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
	/*-Jesse-**********************
	(mx,my,mz) = last 5-fold axis*/
	mx = FiveFold[11].x-XDIM/2;
	my = FiveFold[11].y-YDIM/2;
	mz = FiveFold[11].z-ZDIM/2;
	theta = atan2(my,mx);
	phi = atan2(mz, sqrt(mx*mx+my*my));
	for (n = 1; n < 5; n++) {
		sv = Rotate(sx,sy,sz,theta,phi,n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
		/*-Jesse-**********************************************************************************
		(i,j,k) = (cx,cy,cz)X[(cx,cy,cz)X(second 5-fold axis)] rotated about the last 5-fold axis*/
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

/*-Jesse-*********************************************************************
We set (min_x, min_y, min_z) to be the minimum voxel stored in our binary tree.
We then remove this voxel and adjust the rest of the tree accordingly.  
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


/*-Jesse-***************************************************************
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
		/*-Jesse-****************************************************
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
	/*-Jesse-**************************************************
	Insert the new point into the empty slot left behind by the
	previous parent.*/
	min_heap->x[pointer-1]=x;
	min_heap->y[pointer-1]=y;
	min_heap->z[pointer-1]=z;
	min_heap->t[pointer-1]=t;
	
}

/*-Jesse-**********************************************************************
This function iterates through every voxel adjacent to minVoxel, we'll refer to 
these voxels as adjacentVoxel.  For each adjacentVoxel, we compute its marching 
time and then call set_index_and_time(adjacentVoxel, index of minVoxel, marchingTime).  
To compute marchingTime, we first compute the maximum density gradient at the 
adjacentVoxel.  Then for each voxel adjacent to adjacentVoxel we compute 
Time = [exp(0.1*gradient) + timedata for the voxel adjacent to adjecentVoxel] and 
set marchingTime to be the minimum of all the Time values.  Note that if the 
timedata for the voxel adjacent to the adjacentVoxel has not previously been set 
by any call to set_index_and_time(), then the timedata will be the original 
default value, which is MAX_TIME.*/
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
	/*-Jesse-**********************************************************
	If seed_index has not been previously set by set_index_and_time()*/
	if (seed_index[IndexVect(temp_x,temp_y,temp_z)] >= 254) {	
		boundary = 0;
		time = (float)(MAX_TIME+1.0);
		/*-Jesse-********************************************************
		intens = EXP[0.1 * maximum density gradient at (temp_x,temp_y,temp_z)]*/
		intens = GetImgGradient(temp_x, temp_y, temp_z);
		intens = (float)(exp(ALPHA*intens)); //currently ALPHA set to 0.1f
		
		/*-Jesse-************************************************
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
	
	/*-Jesse-***********************************************************
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

/*-Jesse-******************************************************************
This algorithm is intended to clean up the data after it has been processed 
by removing bubbles.  segmentIndex is an array that stores an index value for 
each voxel that determines a partitioning of the data.  Suppose we have a small
pocket of voxels with index 2 completely surrounded by voxels with index 0,
then bubbleStopper will set the indices of the voxels in the pocket to be 0.
If the pocket is found to have a diamater larger than maxVoxelDiamater then we 
do nothing.  Also, suppose we find a small bubble with indices 0 that borders 
on a region with index 1, and then we discover that it also borders on a region
with index 2.  Since it is unclear whether we should set the pocket index to 1 or 2
we leave it untouched as 0.*/
void bubbleStopperSingle(unsigned char *segmentIndex, unsigned int maxVoxelDiamater){
	int i, j, k, n, m, p;
	int tempBorder_i, tempBorder_j, tempBorder_k;
	double maxBubbleSize = (4/3)*PIE*pow((double)maxVoxelDiamater/2,3);
	/*-Jesse-**********************************************************
	thisIndex stores the index value from segmentIndex[] for the voxels
	in the current bubble*/
	unsigned char thisIndex;
	/*-Jesse-************************************************************
	borderIndex stores the index value from segmentIndex[] for the voxels
	bordering on the outside of the current bubble.  If borderIndex is
	the same as thisIndex then we have not found a voxel on the border 
	with a different index as the voxels inside the bubble*/
	unsigned char borderIndex;
	/*-Jesse-*****************************************************************
	borderIndices_(x,y,z) stores the indices of the voxels lying on the border
	of the bubble that we still want to expand from.  If there is a voxel on 
	the border that we do not wish to expand from then the voxel is not stored
	in borderIndices_(x,y,z).*/
	std::stack<int> borderIndices_x;
	std::stack<int> borderIndices_y;
	std::stack<int> borderIndices_z;
	std::stack<int> bubbleIndices;
	/*-Jesse-
	visited<> is used to prevent a bubble from growing into voxels that are already
	included in the current bubble.  As we add a voxel to the bubble, we set the 
	voxel's corresponding entry in visited<> to be true.*/
	std::vector<bool> visited(XDIM*YDIM*ZDIM, false);

	for (k=0; k<ZDIM; k++){
		for (j=0; j<YDIM; j++){
			for (i=0; i<XDIM; i++){
				/*-Jesse-************************************
				Start growing a bubble from the voxel (i,j,k).
				*/
				visited.at(IndexVect(i,j,k)) = true;
				thisIndex = segmentIndex[IndexVect(i,j,k)];
				borderIndex = segmentIndex[IndexVect(i,j,k)];
				bubbleIndices.push(IndexVect(i,j,k));
				borderIndices_x.push(i);
				borderIndices_y.push(j);
				borderIndices_z.push(k);
				/*-Jesse-**********************************************
				While there are still voxels on the border that we want
				to expand from.*/
				while(!borderIndices_x.empty()){
					tempBorder_i = borderIndices_x.top();
					tempBorder_j = borderIndices_y.top();
					tempBorder_k = borderIndices_z.top();
					borderIndices_x.pop();
					borderIndices_y.pop();
					borderIndices_z.pop();
					/*-Jesse-***********************************************
					For all voxels around the voxel on the border that we're
					currently expanding from*/
					for(p=max(0,tempBorder_k-1); p<min(ZDIM,tempBorder_k+2); p++){
						for(m=max(0,tempBorder_j-1); m<min(YDIM,tempBorder_j+2); m++){
							for(n=max(0,tempBorder_i-1); n<min(XDIM,tempBorder_i+2); n++){
								/*-Jesse-*************************************************
								(n,m,p) is the voxel we're currently trying to expand into.
								If (n,m,p) != the current border voxel that we're trying to
								expand away from.*/
								if(n!=tempBorder_i || m!=tempBorder_j || p!=tempBorder_k){
									/*-Jesse-*************************************************
									If the index of (n,m,p) is the same as the bubble's index*/
									if(segmentIndex[IndexVect(n,m,p)] == thisIndex){
										/*-Jesse**********************************
										If the bubble has spread beyond the maximum diamater*/
										if( sqrt((double)((n-i)*(n-i)+(m-j)*(m-j)+(p-k)*(p-k))) > maxVoxelDiamater/2){
											/*-Jesse-***********************************
											stop growing the bubble and leave it intact*/
											borderIndex = thisIndex;
											p = ZDIM;
											m = YDIM;
											n = XDIM;
											while(!borderIndices_x.empty()){
												borderIndices_x.pop();
												borderIndices_y.pop();
												borderIndices_z.pop();
											}
											while(!bubbleIndices.empty()){
												visited.at(bubbleIndices.top()) = false;
												bubbleIndices.pop();
											}
										/*-Jesse-**************************************
										Else the bubble is within the maximum diamater*/
										}else{
											/*-Jesse-*******************************
											If (n,m,p) is not already in the bubble*/
											if( !visited.at(IndexVect(n,m,p)) ){
												visited.at(IndexVect(n,m,p)) = true;
												bubbleIndices.push(IndexVect(n,m,p));
												borderIndices_x.push(n);
												borderIndices_y.push(m);
												borderIndices_z.push(p);
												/*-Jesse-*************************
												If the bubble has grown too large*/
												if(bubbleIndices.size() > maxBubbleSize){
													/*-Jesse-***********************************
													stop growing the bubble and leave it intact*/
													borderIndex = thisIndex;
													p = ZDIM;
													m = YDIM;
													n = XDIM;
													while(!borderIndices_x.empty()){
														borderIndices_x.pop();
														borderIndices_y.pop();
														borderIndices_z.pop();
													}
													while(!bubbleIndices.empty()){
														visited.at(bubbleIndices.top()) = false;
														bubbleIndices.pop();
													}
												}
											}
										}
									/*-Jesse-*******************************************************
									else the index of (n,m,p) is not the same as the bubble's index*/
									}else{
										if(borderIndex == thisIndex){
											borderIndex = segmentIndex[IndexVect(n,m,p)];
										}else if(borderIndex != segmentIndex[IndexVect(n,m,p)]){
											/*-Jesse-***********************************
											stop growing the bubble and leave it intact*/
											borderIndex = thisIndex;
											p = ZDIM;
											m = YDIM;
											n = XDIM;
											while(!borderIndices_x.empty()){
												borderIndices_x.pop();
												borderIndices_y.pop();
												borderIndices_z.pop();
											}
											while(!bubbleIndices.empty()){
												visited.at(bubbleIndices.top()) = false;
												bubbleIndices.pop();
											}
										}
									}
								}
							}
						}
					}
				}
				/*-Jesse-**********************************************************************
				At this point, we're done growing the bubble.  If borderIndex != thisIndex then
				we have a bubble and we set the segmentIndex of every voxel in the bubble 
				to match the index of the surrounding voxels.  If borderIndex = thisIndex then we 
				do not have a bubble and we do nothing.*/
				if(borderIndex != thisIndex){
					while(!bubbleIndices.empty()){
						segmentIndex[bubbleIndices.top()] = borderIndex;
						visited.at(bubbleIndices.top()) = false;
						bubbleIndices.pop();
					}
				}
			}
		}
	}
}

};
