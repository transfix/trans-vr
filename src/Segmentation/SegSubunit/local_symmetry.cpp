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
#include <time.h>

#define max2(x, y) ((x>y) ? (x):(y))
#define min2(x, y) ((x<y) ? (x):(y))
#define PIE           3.1415926536f
#define ANGL1         1.1071487178f
#define ANGL2         2.0344439358f
#define PolyNum       15
#define MINDIST       3   


#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))

namespace SegSubunit {

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
}INTVECT;


typedef struct CriticalPoint CPNT;
struct CriticalPoint{
  unsigned short x;
  unsigned short y;
  unsigned short z;
  CPNT *next;
};

static CPNT *critical_start;
static int XDIM, YDIM, ZDIM;
static float *dataset;
static float density[10];
static VECTOR *FiveFold;
static VECTOR *TempFold;
static DB_VECTOR *SixFold;
static DB_VECTOR *ThreeFold;
static DB_VECTOR *LocalFold;
static float min_radius;
static unsigned short *classify;
static float symmetry_score;

static int v_num, t_num;
static VECTOR *vertex;
static INTVECT *triangle;


void LocalSymmetryDetect(double, double, int, float*);
float LocalSymmetryScore(float, float, float, float, float, float, float*,float*, int);
float LocalSymmetryScore2(float, float, float, float, float, float, float*, float*, int,int);
void SearchNeighbors(float*, float*, float*, float*, float*, float*, float*, float*,
		       float *, int, float);
void SearchNeighbors2(float*, float*, float*, float*, float*, float*, float*, float*, 
		       float *, int, float,int);
VECTOR Rotate(float, float, float, float, float, float, float, float, float); //int, int, int);
void DrawLine(float, float, float, float, float, float, float);
void NeighborOrder(VECTOR *, int);
void LocalRefinement(float *,float *,float *,float *,float *,float *,float*, float*, int);
float GetIntersection(float, float, float, float*);
float GetLength(float, float, float, float*);
float GetLocalLength(float, float, float, float, float, float, float*, float*, float);

//float* max_tmp;

void LocalSymmetryRefine(int xd,int yd,int zd,float *data,CPNT *critical,
     unsigned short *p_classify, VECTOR *five_fold, int numfold5,DB_VECTOR *local_fold,
     FILE *fp,FILE *fp2,int localnumfold,int numaxis,float *span_tmp,float *orig_tmp,float tlow)
{
  int d,n,m,num;
  float temp;
  float cx,cy,cz;
  float fx,fy,fz;
  float gx,gy,gz;
  float ax,ay,az;
  float score,radius;
  VECTOR *fold_tmp;
  int count, min_index;
  float distance, min_dist;
  float theta,phi;
  float px,py,pz;
  float qx,qy,qz;
  float nx,ny,nz;
  VECTOR sv1,sv2,sv;

  float* max_tmp;
  max_tmp=  (float *)malloc(sizeof(float)*3);

  max_tmp[0] = orig_tmp[0] + (xd-1)*span_tmp[0];
  max_tmp[1] = orig_tmp[1] + (yd-1)*span_tmp[1];
  max_tmp[2] = orig_tmp[2] + (zd-1)*span_tmp[2];

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  critical_start = critical;
  classify = p_classify;
  LocalFold = local_fold;
  FiveFold = five_fold;


  vertex = (VECTOR*)malloc(sizeof(VECTOR)*5000*PolyNum);
  triangle = (INTVECT*)malloc(sizeof(INTVECT)*5000*PolyNum);
  v_num = 0;
  t_num = 0;
  
  for (n=0; n<12; n++) {
    cx = FiveFold[n].x;
    cy = FiveFold[n].y;
    cz = FiveFold[n].z;
    DrawLine(cx,cy,cz,max_tmp[0]/2 /*XDIM/2*/,max_tmp[1]/2/*YDIM/2*/,max_tmp[2]/2/*ZDIM/2*/,3.f);    
  }
  
  fold_tmp = (VECTOR*)malloc(sizeof(VECTOR)*60);
  for (n = 0; n < 12; n++) {
    cx = FiveFold[n].x;
    cy = FiveFold[n].y;
    cz = FiveFold[n].z;
    for (m = 0; m < 12; m++) {
      fold_tmp[m].x = FiveFold[m].x;
      fold_tmp[m].y = FiveFold[m].y;
      fold_tmp[m].z = FiveFold[m].z;
    }
    count = 0;
    while (count < 5) {
      min_dist = 999999.0;
      for (m = 0; m < 12; m++) {
	if (m != n) {
	  ax = fold_tmp[m].x;
	  ay = fold_tmp[m].y;
	  az = fold_tmp[m].z;
	  distance = (float)sqrt((cx-ax)*(cx-ax)+(cy-ay)*(cy-ay)+(cz-az)*(cz-az));
	  if (distance < min_dist) {
	    min_index = m;
	    min_dist = distance;
	  }
	}
      }
      if (n < min_index) 
	DrawLine(cx,cy,cz,fold_tmp[min_index].x,
		 fold_tmp[min_index].y,fold_tmp[min_index].z,3);
      fold_tmp[min_index].x = -9999.0;
      fold_tmp[min_index].y = -9999.0;
      fold_tmp[min_index].z = -9999.0;
      count++;
    }
  }
  free(fold_tmp);

   for (d = 0; d < numaxis; d++) {

    fx = LocalFold[d*60].sx;
    fy = LocalFold[d*60].sy;
    fz = LocalFold[d*60].sz;
    gx = LocalFold[d*60].ex;
    gy = LocalFold[d*60].ey;
    gz = LocalFold[d*60].ez;

    temp = sqrt((fx-gx)*(fx-gx)+(fy-gy)*(fy-gy)+(fz-gz)*(fz-gz));
    cx = (fx-gx)/temp;
    cy = (fy-gy)/temp;
    cz = (fz-gz)/temp;
    temp = GetLocalLength(gx,gy,gz,fx,fy,fz,orig_tmp, span_tmp, tlow);
    if (temp > 0) {
      ax = gx+temp*cx;
      ay = gy+temp*cy;
      az = gz+temp*cz;
      fx = ax+cx*30.0;
      fy = ay+cy*30.0;
      fz = az+cz*30.0;
      gx = ax-cx*30.0;
      gy = ay-cy*30.0;
      gz = az-cz*30.0;
    }
    else {
      ax = 0.5*(gx+fx);
      ay = 0.5*(gy+fy);
      az = 0.5*(gz+fz);
      fx = ax+cx*30.0;
      fy = ay+cy*30.0;
      fz = az+cz*30.0;
      gx = ax-cx*30.0;
      gy = ay-cy*30.0;
      gz = az-cz*30.0;
    }

    score = -LocalSymmetryScore2(fx,fy,fz,gx,gy,gz,orig_tmp, span_tmp, localnumfold,d*60+numfold5*12);
 
    radius = 2.0f;
    while (radius > 0) {
      score = -score;
      while (score > 0) 
	SearchNeighbors2(&fx,&fy,&fz,&gx,&gy,&gz,orig_tmp, span_tmp, &score,localnumfold,radius,d*60+numfold5*12);
      radius -= 0.2f;
    }

    temp = sqrt((fx-gx)*(fx-gx)+(fy-gy)*(fy-gy)+(fz-gz)*(fz-gz));
    cx = (fx-gx)/temp;
    cy = (fy-gy)/temp;
    cz = (fz-gz)/temp;
    temp = GetLocalLength(gx,gy,gz,fx,fy,fz,orig_tmp, span_tmp, tlow);
    if (temp > 0) {
      ax = gx+temp*cx;
      ay = gy+temp*cy;
      az = gz+temp*cz;
      fx = ax+cx*30.0;
      fy = ay+cy*30.0;
      fz = az+cz*30.0;
      gx = ax-cx*30.0;
      gy = ay-cy*30.0;
      gz = az-cz*30.0;
    }
    else {
      ax = 0.5*(gx+fx);
      ay = 0.5*(gy+fy);
      az = 0.5*(gz+fz);
      fx = ax+cx*30.0;
      fy = ay+cy*30.0;
      fz = az+cz*30.0;
      gx = ax-cx*30.0;
      gy = ay-cy*30.0;
      gz = az-cz*30.0;
    }

    LocalFold[d*60].sx = fx;
    LocalFold[d*60].sy = fy;
    LocalFold[d*60].sz = fz;
    LocalFold[d*60].ex = gx;
    LocalFold[d*60].ey = gy;
    LocalFold[d*60].ez = gz;
    
    DrawLine(fx,fy,fz,gx,gy,gz,1.0f);
    
    score = LocalSymmetryScore2(fx,fy,fz, gx,gy,gz,orig_tmp, span_tmp, localnumfold,d*60+numfold5*12);
    fprintf(fp, "%f %f %f %f %f %f %f\n",fx,fy,fz, gx,gy,gz,symmetry_score);
     
    num = 1;
    cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
    cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
    cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
    theta = (float)atan2(cy,cx);
    phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
    for (n = 1; n < 5; n++) {
      if (LocalFold[d*60+num].sx != -9999.0f) {
	sv = Rotate(fx,fy,fz,theta,phi,n*2.0f*PIE/5.0f, max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	px = sv.x;
	py = sv.y;
	pz = sv.z;
	sv = Rotate(gx,gy,gz,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	qx = sv.x;
	qy = sv.y;
	qz = sv.z;
	
	LocalFold[d*60+num].sx = px;
	LocalFold[d*60+num].sy = py;
	LocalFold[d*60+num].sz = pz;
	LocalFold[d*60+num].ex = qx;
	LocalFold[d*60+num].ey = qy;
	LocalFold[d*60+num].ez = qz;
	DrawLine(px,py,pz,qx,qy,qz,1.0f);
	fprintf(fp, "%f %f %f %f %f %f %f\n",px,py,pz, qx,qy,qz,symmetry_score);
      }
      else 
	fprintf(fp, "%f %f %f %f %f %f %f\n",-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,symmetry_score);

      num ++;
    }
 
 	for (m = 1; m < 11; m++) {
      nx = FiveFold[m].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[m].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[m].z-max_tmp[2]/2; //ZDIM/2;
      ax = nz*cy-ny*cz;
      ay = nx*cz-nz*cx;
      az = ny*cx-nx*cy;
      theta = (float)atan2(ay,ax);
      phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
      if (m < 6) {
	sv1 = Rotate(fx,fy,fz,theta,phi,ANGL1,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	sv2 = Rotate(gx,gy,gz,theta,phi,ANGL1, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
      }
      else {
	sv1 = Rotate(fx,fy,fz,theta,phi,ANGL2,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	sv2 = Rotate(gx,gy,gz,theta,phi,ANGL2,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      }
      
      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      for (n = 0; n < 5; n++) {
		if (LocalFold[d*60+num].sx != -9999.0f) {
	  	  sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	  	  px = sv.x;
	  	  py = sv.y;
	      pz = sv.z;
	      sv = Rotate(sv2.x,sv2.y,sv2.z,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	      qx = sv.x;
	      qy = sv.y;
	      qz = sv.z;
	  
	      LocalFold[d*60+num].sx = px;
	      LocalFold[d*60+num].sy = py;
	      LocalFold[d*60+num].sz = pz;
	 	  LocalFold[d*60+num].ex = qx;
	      LocalFold[d*60+num].ey = qy;
	      LocalFold[d*60+num].ez = qz;
	      DrawLine(px,py,pz,qx,qy,qz,1.0f);
	      fprintf(fp, "%f %f %f %f %f %f %f\n",px,py,pz, qx,qy,qz,symmetry_score);
	    }
	    else 
	     fprintf(fp, "%f %f %f %f %f %f %f\n",-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,symmetry_score);

	   num ++;
       }
    }

    nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
    ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
    nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
    ax = nz*cy-ny*cz;
    ay = nx*cz-nz*cx;
    az = ny*cx-nx*cy;
    theta = (float)atan2(ay,ax);
    phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
    sv1 = Rotate(fx,fy,fz,theta,phi,PIE, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
    px = sv1.x;
    py = sv1.y;
    pz = sv1.z;
    sv2 = Rotate(gx,gy,gz,theta,phi,PIE,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    qx = sv2.x;
    qy = sv2.y;
    qz = sv2.z; 
    if (LocalFold[d*60+num].sx != -9999.0f) {
      LocalFold[d*60+num].sx = px;
      LocalFold[d*60+num].sy = py;
      LocalFold[d*60+num].sz = pz;
      LocalFold[d*60+num].ex = qx;
      LocalFold[d*60+num].ey = qy;
      LocalFold[d*60+num].ez = qz;
      DrawLine(px,py,pz,qx,qy,qz,1.0f);
      fprintf(fp, "%f %f %f %f %f %f %f\n",px,py,pz, qx,qy,qz,symmetry_score);
    }
    else 
      fprintf(fp, "%f %f %f %f %f %f %f\n",-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,symmetry_score);
    
    num ++;
    
    nx = FiveFold[11].x-max_tmp[0]/2; //XDIM/2;
    ny = FiveFold[11].y-max_tmp[1]/2; //YDIM/2;
    nz = FiveFold[11].z-max_tmp[2]/2; //ZDIM/2;
    theta = (float)atan2(ny,nx);
    phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
    for (n = 1; n < 5; n++) {
      if (LocalFold[d*60+num].sx != -9999.0f) {
	sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	px = sv.x;
	py = sv.y;
	pz = sv.z;
	sv = Rotate(sv2.x,sv2.y,sv2.z,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	qx = sv.x;
	qy = sv.y;
	qz = sv.z;
	
	LocalFold[d*60+num].sx = px;
	LocalFold[d*60+num].sy = py;
	LocalFold[d*60+num].sz = pz;
	LocalFold[d*60+num].ex = qx;
	LocalFold[d*60+num].ey = qy;
	LocalFold[d*60+num].ez = qz;
	DrawLine(px,py,pz,qx,qy,qz,1.0f);
	fprintf(fp, "%f %f %f %f %f %f %f\n",px,py,pz, qx,qy,qz,symmetry_score);
      }
      else 
	fprintf(fp, "%f %f %f %f %f %f %f\n",-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,-9999.0f,symmetry_score);
      
      num ++;
    }
    
  }
  fclose(fp);
  
  
  fprintf(fp2, "%d %d\n",v_num,t_num);
  for (n = 0; n < v_num; n++) 
    fprintf(fp2, "%f %f %f\n",vertex[n].x, vertex[n].y,vertex[n].z);
  //  fprintf(fp2, "%f %f %f\n",vertex[n].x*span_tmp[0]+orig_tmp[0],
//	    vertex[n].y*span_tmp[1]+orig_tmp[1],
//	    vertex[n].z*span_tmp[2]+orig_tmp[2]);
  for (n = 0; n < t_num; n++) 
    fprintf(fp2, "%d %d %d\n",triangle[n].x,triangle[n].y,triangle[n].z);
  fclose(fp2);

  free(triangle);
  free(vertex);
  
}




void LocalSymmetry(int xd,int yd,int zd,float* orig_tmp, float* span_tmp, float *data,CPNT *critical,int h_num,int k_num,
		   DB_VECTOR *three_fold,VECTOR *five_fold, DB_VECTOR *six_fold, 
		   int *numfd3,int *numfd6,int numfd5)
{
  int i,j,k;
  float distance, dist;
  float xx,yy,zz;
  float x,y,z;
  int num,localfold;
  VECTOR *fold_tmp,*neighbor;
  float sym_rd, score,radius;
  int m,n,count, min_index;
  float min_dist;
  int total_num,u;
  float px,py,pz;
  float qx,qy,qz;
  int numfold3,numfold6;

  float temp;
  float nx,ny,nz;
  float ax,ay,az;
  VECTOR sv,sv1,sv2;
  float theta,phi;
  
  unsigned char *hex_lattice;
  double cx,cy,cz;
  double fx,fy,fz;
  double gx,gy,gz;
  double low_bound, high_bound;
  double a,b,t;
  double sqrtof3,ratio1,ratio2;
  double *angle1,*angle2;

  float max_tmp[3];
  max_tmp[0]=orig_tmp[0] + (xd-1)*span_tmp[0];
  max_tmp[1]=orig_tmp[1] + (yd-1)*span_tmp[1];
  max_tmp[2]=orig_tmp[2] + (zd-1)*span_tmp[2];

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  critical_start = critical;
  FiveFold = five_fold;
  SixFold = six_fold;
  ThreeFold = three_fold;
  numfold3 = *numfd3;
  numfold6 = *numfd6;
  sym_rd = (float)(min2(max_tmp[0],min2(max_tmp[1],max_tmp[2]))/2-2);
  
  //sym_rd = (float)(min2(XDIM,min2(YDIM,ZDIM))/2-2);
  
  dist = 2.0*sym_rd*sin(ANGL1*0.5);
  distance = (float)h_num*(float)h_num + (float)k_num*(float)k_num 
    + 0.5*(float)h_num*(float)k_num;
  min_radius = 0.5*dist/sqrt(distance);
  
 
  sqrtof3 = sqrt(3.0);
  gx = (double)h_num*sqrtof3/2.0;
  gy = (double)h_num*0.5 + (double)k_num;
  fx = -(double)k_num*sqrtof3/2.0;
  fy = (double)k_num*0.5 + (double)h_num;
  
  fold_tmp = (VECTOR*)malloc(sizeof(VECTOR)*6000);
  TempFold = (VECTOR*)malloc(sizeof(VECTOR)*6000); 
  neighbor = (VECTOR*)malloc(sizeof(VECTOR)*6); 
  hex_lattice = (unsigned char*)malloc(sizeof(unsigned char)*(h_num+k_num)*2*(h_num+k_num));
  for (i = 0; i < k_num+h_num; i++) 
    for (j = 0; j < 2*(h_num+k_num); j++) 
      hex_lattice[i*2*(h_num+k_num)+j] = 0;

  angle1 = (double*)malloc(sizeof(double)*100);
  angle2 = (double*)malloc(sizeof(double)*100);
  if (numfold6 > 0) {
    num = 0;
    temp = 0;
    for (i = -k_num; i <= h_num; i++) {
      if (i < 0) {
	cx = (double)i*sqrtof3/2.0;
	low_bound = fy*cx/fx+(double)i*0.5;
	high_bound = fy+(gy-fy)*(cx-fx)/(gx-fx)+(double)i*0.5;
	for (j = (int)(low_bound+0.5); j <= (int)(high_bound+0.5); j++) {
	  if ((j > low_bound-0.000001 && j < low_bound+0.000001 && i >= -k_num/2) ||
	      (j > low_bound+0.000001 && j < high_bound-0.000001)) {
	    if (hex_lattice[(i+k_num)*(h_num+k_num)+j] == 0) {
	      hex_lattice[(i+k_num)*(h_num+k_num)+j] = 1;
	      cy = -(double)i*0.5 + (double)j;
	      
	      //if (sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)) < 1.1 ||
	      //	  sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)) < 1.1 ||
	      //	  sqrt(cx*cx+cy*cy) < 1.1)
	      //	continue;
	      
	      distance = 99999.0;
	      if (distance > sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)))
		distance = sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy));
	      if (distance > sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)))
		distance = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
	      if (distance > sqrt(cx*cx+cy*cy))
		distance = sqrt(cx*cx+cy*cy);
	      if (distance > temp) {
		count = num;
		temp = distance;
	      }
	      
	      a = (gy-fy)/(gx-fx);
	      t = (fy-a*fx)/(cy/cx-a);
	      b = a*(t-fx)+fy;
	      a = t;
	      ratio1 = sqrt((fx-a)*(fx-a)+(fy-b)*(fy-b))/
		sqrt((gx-fx)*(gx-fx)+(gy-fy)*(gy-fy));
	      ratio2 = sqrt(cx*cx+cy*cy)/sqrt(a*a+b*b);
	      angle2[num] = acos(0.5*(cx*cx+cy*cy+gx*gx+gy*gy-(cx-gx)*(cx-gx)-(cy-gy)*(cy-gy))/
			    sqrt((cx*cx+cy*cy)*(gx*gx+gy*gy)));
	      angle1[num] = acos(0.5*(gx*gx+gy*gy+(cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)-cx*cx-cy*cy)/
			    sqrt(((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy))*(gx*gx+gy*gy)));
	      num++;
	      
	      xx = (1.0-ratio1)*gx;
	      yy = (1.0-ratio1)*gy;
	      a = fx + ratio2*(xx-fx);
	      b = fy + ratio2*(yy-fy);
	      if (a <= 0) {
		b += a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a-0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      else {
		b -= a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a+0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      
	      xx = ratio1*fx;
	      yy = ratio1*fy;
	      a = gx + ratio2*(xx-gx);
	      b = gy + ratio2*(yy-gy);
	      if (a <= 0) {
		b += a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a-0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      else {
		b -= a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a+0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	    }
	  }
	}
      }
      
      else if (i == 0) {
	cx = 0;
	high_bound = fy-fx*(gy-fy)/(gx-fx);
	for (j = 1; j <= (int)(high_bound+0.5); j++) {
	  if ((k_num > 0 && j < high_bound-0.000001) ||
	      (k_num == 0 && j <= (int)(0.5*high_bound))) {
	    if (hex_lattice[(k_num)*(h_num+k_num)+j] == 0) {
	      hex_lattice[(k_num)*(h_num+k_num)+j] = 1;
	      cy = (double)j;
	      
	      //if (sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)) < 1.1 ||
	      //	  sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)) < 1.1 ||
	      //	  sqrt(cx*cx+cy*cy) < 1.1)
	      //	continue;
	      
	      distance = 99999.0;
	      if (distance > sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)))
		distance = sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy));
	      if (distance > sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)))
		distance = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
	      if (distance > sqrt(cx*cx+cy*cy))
		distance = sqrt(cx*cx+cy*cy);
	      if (distance > temp) {
		count = num;
		temp = distance;
	      }
	      
	      a = 0;
	      b = high_bound;
	      ratio1 = sqrt((fx-a)*(fx-a)+(fy-b)*(fy-b))/
	             sqrt((gx-fx)*(gx-fx)+(gy-fy)*(gy-fy));
	      ratio2 = sqrt(cx*cx+cy*cy)/sqrt(a*a+b*b);
	      angle2[num] = acos(0.5*(cx*cx+cy*cy+gx*gx+gy*gy-(cx-gx)*(cx-gx)-(cy-gy)*(cy-gy))/
			    sqrt((cx*cx+cy*cy)*(gx*gx+gy*gy)));
	      angle1[num] = acos(0.5*(gx*gx+gy*gy+(cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)-cx*cx-cy*cy)/
			    sqrt(((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy))*(gx*gx+gy*gy)));
	      num++;
	      
	      xx = (1.0-ratio1)*gx;
	      yy = (1.0-ratio1)*gy;
	      a = fx + ratio2*(xx-fx);
	      b = fy + ratio2*(yy-fy);
	      if (a <= 0) {
		b += a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a-0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      else {
		b -= a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a+0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      
	      xx = ratio1*fx;
	      yy = ratio1*fy;
	      a = gx + ratio2*(xx-gx);
	      b = gy + ratio2*(yy-gy);
	      if (a <= 0) {
		b += a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a-0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      else {
		b -= a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a+0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	    }
	  }
	}
	
      }
      
      else {
	cx = (double)i*sqrtof3/2.0;
	low_bound = gy*cx/gx-(double)i*0.5;
	high_bound = fy+(gy-fy)*(cx-fx)/(gx-fx)-(double)i*0.5;
	for (j = (int)(low_bound+0.5); j <= (int)(high_bound+0.5); j++) {
	  if (j > low_bound+0.000001 && j < high_bound-0.000001) {
	    if (hex_lattice[(i+k_num)*(h_num+k_num)+j] == 0) {
	      hex_lattice[(i+k_num)*(h_num+k_num)+j] = 1;
	      cy = (double)i*0.5 + (double)j;
	      
	      //if (sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)) < 1.1 ||
	      //	  sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)) < 1.1 ||
	      //	  sqrt(cx*cx+cy*cy) < 1.1)
	      //	continue;
	      	      
	      distance = 99999.0;
	      if (distance > sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)))
		distance = sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy));
	      if (distance > sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)))
		distance = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
	      if (distance > sqrt(cx*cx+cy*cy))
		distance = sqrt(cx*cx+cy*cy);
	      if (distance > temp) {
		count = num;
		temp = distance;
	      }
	      
	      a = (gy-fy)/(gx-fx);
	      t = (fy-a*fx)/(cy/cx-a);
	      b = a*(t-fx)+fy;
	      a = t;
	      ratio1 = sqrt((fx-a)*(fx-a)+(fy-b)*(fy-b))/
		sqrt((gx-fx)*(gx-fx)+(gy-fy)*(gy-fy));
	      ratio2 = sqrt(cx*cx+cy*cy)/sqrt(a*a+b*b);
	      angle2[num] = acos(0.5*(cx*cx+cy*cy+gx*gx+gy*gy-(cx-gx)*(cx-gx)-(cy-gy)*(cy-gy))/
			    sqrt((cx*cx+cy*cy)*(gx*gx+gy*gy)));
	      angle1[num] = acos(0.5*(gx*gx+gy*gy+(cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)-cx*cx-cy*cy)/
			    sqrt(((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy))*(gx*gx+gy*gy)));
	      num++;
	      
	      xx = (1.0-ratio1)*gx;
	      yy = (1.0-ratio1)*gy;
	      a = fx + ratio2*(xx-fx);
	      b = fy + ratio2*(yy-fy);
	      if (a <= 0) {
		b += a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a-0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      else {
		b -= a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a+0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      
	      xx = ratio1*fx;
	      yy = ratio1*fy;
	      a = gx + ratio2*(xx-gx);
	      b = gy + ratio2*(yy-gy);
	      if (a <= 0) {
		b += a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a-0.5);
	      n = (int)(b+0.5);
	      hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	      else {
		b -= a*sqrtof3/3.0;
		a /= (sqrtof3/2.0);
		m = (int)(a+0.5);
		n = (int)(b+0.5);
		hex_lattice[(m+k_num)*(h_num+k_num)+n] = 1;
	      }
	    }
	  }
	}
      }
    }
  
    *numfd6 = num;
    total_num = num;
    printf("number of independent 6-fold subunits: %d \n",num);
     
    cx = angle1[0];
    cy = angle2[0];
    angle1[0] = angle1[count];
    angle2[0] = angle2[count];
    angle1[count] = cx;
    angle2[count] = cy;
    
    for (num = 0; num < total_num; num++)
      LocalSymmetryDetect(angle1[num],angle2[num],num, max_tmp);
	 
    if (numfd5 > 0) {
      num = total_num*60;
      for (n = 0; n < 12; n++) {
	cx = FiveFold[n].x;
	cy = FiveFold[n].y;
	cz = FiveFold[n].z;
	gx = max_tmp[0]/2; //XDIM/2;
	gy = max_tmp[1]/2; //YDIM/2;
	gz = max_tmp[2]/2; //ZDIM/2;
	
	dist = GetIntersection(cx,cy,cz, max_tmp);
	distance = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)+(cz-gz)*(cz-gz));
	xx = gx + dist*(cx-gx)/distance;
	yy = gy + dist*(cy-gy)/distance;
	zz = gz + dist*(cz-gz)/distance;
	
	TempFold[num].x = xx;
	TempFold[num].y = yy;
	TempFold[num].z = zz;
	num ++;
      }
    }
    
    localfold = total_num*60+12*numfd5;
    for (u = 0; u < total_num; u++) {
      num = u*60;
      
      x = TempFold[num].x;
      y = TempFold[num].y;
      z = TempFold[num].z;
      
      for (m = 0; m < localfold; m++) {
	fold_tmp[m].x = TempFold[m].x;
	fold_tmp[m].y = TempFold[m].y;
	fold_tmp[m].z = TempFold[m].z;
      }
      count = 0;
      while (count < 6) {
	min_dist = 999999.0f;
	for (m = 0; m < localfold; m++) {
	  if (m != num) {
	    xx = fold_tmp[m].x;
	    yy = fold_tmp[m].y;
	    zz = fold_tmp[m].z;
	    distance = sqrt((x-xx)*(x-xx)+(y-yy)*(y-yy)+(z-zz)*(z-zz));
	    if (distance < min_dist) {
	      min_index = m;
	      min_dist = distance;
	    }
	  }
	}
	
	neighbor[count].x = fold_tmp[min_index].x;
	neighbor[count].y = fold_tmp[min_index].y;
	neighbor[count].z = fold_tmp[min_index].z;
	fold_tmp[min_index].x = -9999.0;
	fold_tmp[min_index].y = -9999.0;
	fold_tmp[min_index].z = -9999.0;
	count++;
      }
      
      NeighborOrder(neighbor, 6);
      
      cx = 0;
      cy = 0;
      cz = 0;
      for (k = 0; k < 6; k++) {
	px = neighbor[k].x-x;
	py = neighbor[k].y-y;
	pz = neighbor[k].z-z;
	if (k < 5) {
	  qx = neighbor[k+1].x-x;
	  qy = neighbor[k+1].y-y;
	  qz = neighbor[k+1].z-z;
	}
	else {
	  qx = neighbor[0].x-x;
	  qy = neighbor[0].y-y;
	  qz = neighbor[0].z-z;
	}
	
	xx = py*qz-pz*qy;
	yy = pz*qx-px*qz;
	zz = px*qy-py*qx;
	distance = sqrt(xx*xx+yy*yy+zz*zz);
	cx += xx/distance;
	cy += yy/distance;
	cz += zz/distance;
      }
      
      distance = sqrt(cx*cx+cy*cy+cz*cz);
      cx /= distance;
      cy /= distance;
      cz /= distance;
      
      score = (x-max_tmp[0]/2/*XDIM/2*/)*cx + (y-max_tmp[1]/2/*YDIM/2*/)*cy + (z-max_tmp[2]/2/*ZDIM/2*/)*cz;
      if (score < 0) {
	cx = -cx;
	cy = -cy;
	cz = -cz;
      }
      
      px = x + 30*cx;
      py = y + 30*cy;
      pz = z + 30*cz;
      qx = x - 30*cx;
      qy = y - 30*cy;
      qz = z - 30*cz;

      //LocalRefinement(&px,&py,&pz,&qx,&qy,&qz,numfold6);
      
      score = -LocalSymmetryScore(px,py,pz, qx,qy,qz,orig_tmp, span_tmp, numfold6);
      radius = 2.0f;
      while (radius > 0) {
	score = -score;
	while (score > 0) 
	  SearchNeighbors(&px,&py,&pz, &qx,&qy,&qz, orig_tmp, span_tmp,&score, numfold6, radius);
	radius -= 0.2f;
      }
      
      fx = px;
      fy = py;
      fz = pz;
      gx = qx;
      gy = qy;
      gz = qz;
      
      
      SixFold[num].sx = fx;
      SixFold[num].sy = fy;
      SixFold[num].sz = fz;
      SixFold[num].ex = gx;
      SixFold[num].ey = gy;
      SixFold[num].ez = gz;
      num ++;
      
      
      cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
      cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
      cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
      theta = (float)atan2(cy,cx);
      phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
      for (n = 1; n < 5; n++) {
	sv = Rotate(fx,fy,fz,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	px = sv.x;
	py = sv.y;
	pz = sv.z;
	sv = Rotate(gx,gy,gz,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	qx = sv.x;
	qy = sv.y;
	qz = sv.z;
	temp = 999999.0;
	for (k=u*60; k<num; k++) {
	  a = (float)sqrt((px-SixFold[k].sx)*(px-SixFold[k].sx)+
			  (py-SixFold[k].sy)*(py-SixFold[k].sy)+
			  (pz-SixFold[k].sz)*(pz-SixFold[k].sz));
	  if (a < temp) 
	    temp = a;
	}
	
	if (temp > MINDIST) {
	  SixFold[num].sx = px;
	  SixFold[num].sy = py;
	  SixFold[num].sz = pz;
	  SixFold[num].ex = qx;
	  SixFold[num].ey = qy;
	  SixFold[num].ez = qz;
	  num ++;
	}
	else {
	  SixFold[num].sx = -9999.0;
	  SixFold[num].sy = -9999.0;
	  SixFold[num].sz = -9999.0;
	  SixFold[num].ex = -9999.0;
	  SixFold[num].ey = -9999.0;
	  SixFold[num].ez = -9999.0;
	  num++;
	}
      }
      
      for (m = 1; m < 11; m++) {
	nx = FiveFold[m].x-max_tmp[0]/2; //XDIM/2;
	ny = FiveFold[m].y-max_tmp[1]/2; //YDIM/2;
	nz = FiveFold[m].z-max_tmp[2]/2; //ZDIM/2;
	ax = nz*cy-ny*cz;
	ay = nx*cz-nz*cx;
	az = ny*cx-nx*cy;
	theta = (float)atan2(ay,ax);
	phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
	if (m < 6) {
	  sv1 = Rotate(fx,fy,fz,theta,phi,ANGL1,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	  sv2 = Rotate(gx,gy,gz,theta,phi,ANGL1,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	}
	else {
	  sv1 = Rotate(fx,fy,fz,theta,phi,ANGL2, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
	  sv2 = Rotate(gx,gy,gz,theta,phi,ANGL2, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
	}
	
	theta = (float)atan2(ny,nx);
	phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
	for (n = 0; n < 5; n++) {
	  sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	  px = sv.x;
	  py = sv.y;
	  pz = sv.z;
	  sv = Rotate(sv2.x,sv2.y,sv2.z,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	  qx = sv.x;
	  qy = sv.y;
	  qz = sv.z;
	  temp = 999999.0;
	  for (k=u*60; k<num; k++) {
	    a = (float)sqrt((px-SixFold[k].sx)*(px-SixFold[k].sx)+
			    (py-SixFold[k].sy)*(py-SixFold[k].sy)+
			    (pz-SixFold[k].sz)*(pz-SixFold[k].sz));
	    if (a < temp) 
	      temp = a;
	  }
	  
	  if (temp > MINDIST) {
	    SixFold[num].sx = px;
	    SixFold[num].sy = py;
	    SixFold[num].sz = pz;
	    SixFold[num].ex = qx;
	    SixFold[num].ey = qy;
	    SixFold[num].ez = qz;
	    num ++;
	  }
	  else {
	    SixFold[num].sx = -9999.0;
	    SixFold[num].sy = -9999.0;
	    SixFold[num].sz = -9999.0;
	    SixFold[num].ex = -9999.0;
	    SixFold[num].ey = -9999.0;
	    SixFold[num].ez = -9999.0;
	    num++;
	  }
	}
      }
      
      nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
      ax = nz*cy-ny*cz;
      ay = nx*cz-nz*cx;
      az = ny*cx-nx*cy;
      theta = (float)atan2(ay,ax);
      phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
      sv1 = Rotate(fx,fy,fz,theta,phi,PIE, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
      px = sv1.x;
      py = sv1.y;
      pz = sv1.z;
      sv2 = Rotate(gx,gy,gz,theta,phi,PIE, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
      qx = sv2.x;
      qy = sv2.y;
      qz = sv2.z; 
      temp = 999999.0;
      for (k=u*60; k<num; k++) {
	a = (float)sqrt((px-SixFold[k].sx)*(px-SixFold[k].sx)+
			(py-SixFold[k].sy)*(py-SixFold[k].sy)+
			(pz-SixFold[k].sz)*(pz-SixFold[k].sz));
	if (a < temp) 
	  temp = a;
      }
      
      if (temp > MINDIST) {
	SixFold[num].sx = px;
	SixFold[num].sy = py;
	SixFold[num].sz = pz;
	SixFold[num].ex = qx;
	SixFold[num].ey = qy;
	SixFold[num].ez = qz;
	num ++;
      }
      else {
	SixFold[num].sx = -9999.0;
	SixFold[num].sy = -9999.0;
	SixFold[num].sz = -9999.0;
	SixFold[num].ex = -9999.0;
	SixFold[num].ey = -9999.0;
	SixFold[num].ez = -9999.0;
	num++;
      }
      
      nx = FiveFold[11].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[11].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[11].z-max_tmp[2]/2; //ZDIM/2;
      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      for (n = 1; n < 5; n++) {
	sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
	px = sv.x;
	py = sv.y;
	pz = sv.z;
	sv = Rotate(sv2.x,sv2.y,sv2.z,theta,phi,n*2.0f*PIE/5.0f, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
	qx = sv.x;
	qy = sv.y;
	qz = sv.z;
	temp = 999999.0;
	for (k=u*60; k<num; k++) {
	  a = (float)sqrt((px-SixFold[k].sx)*(px-SixFold[k].sx)+
			  (py-SixFold[k].sy)*(py-SixFold[k].sy)+
			  (pz-SixFold[k].sz)*(pz-SixFold[k].sz));
	  if (a < temp) 
	    temp = a;
	}
	if (temp > MINDIST) {
	  SixFold[num].sx = px;
	  SixFold[num].sy = py;
	  SixFold[num].sz = pz;
	  SixFold[num].ex = qx;
	  SixFold[num].ey = qy;
	  SixFold[num].ez = qz;
	  num ++;
	}
	else {
	  SixFold[num].sx = -9999.0;
	  SixFold[num].sy = -9999.0;
	  SixFold[num].sz = -9999.0;
	  SixFold[num].ex = -9999.0;
	  SixFold[num].ey = -9999.0;
	  SixFold[num].ez = -9999.0;
	  num++;
	}
      }
    }
   
  }
 
 
  sqrtof3 = sqrt(3.0);
  gx = (double)h_num*sqrtof3/2.0;
  gy = (double)h_num*0.5 + (double)k_num;
  fx = -(double)k_num*sqrtof3/2.0;
  fy = (double)k_num*0.5 + (double)h_num;
  
  for (i = 0; i < k_num+h_num; i++) 
    for (j = 0; j < 2*(h_num+k_num); j++) 
      hex_lattice[i*2*(h_num+k_num)+j] = 0;

  if (numfold3 > 0) {
    num = 0;
    temp = 0;
    for (i = -k_num; i < h_num; i++) {
      for (j = 0; j < 2*(h_num+k_num); j++) {
	cy = (double)j*0.5;
	if ((i+j)%2 == 0) 
	  cx = (double)i*sqrtof3/2.0 + sqrtof3/3.0;
	else
	  cx = (double)i*sqrtof3/2.0 + sqrtof3/6.0;
	
	//if (sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)) < 1 ||
	//	  sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)) < 1 ||
	//	  sqrt(cx*cx+cy*cy) < 1)
	//continue;
		
	if (i < 0) 
	  low_bound = fy*cx/fx;
	else
	  low_bound = gy*cx/gx;
	high_bound = fy+(gy-fy)*(cx-fx)/(gx-fx);

	if ((cy > low_bound-0.000001 && cy < high_bound-0.000001 && cx <= fx/2) ||
	    (cy > low_bound+0.000001 && cy < high_bound-0.000001 && cx > fx/2)) {
	  
	  if (hex_lattice[(i+k_num)*2*(h_num+k_num)+j] == 0) {
	    hex_lattice[(i+k_num)*2*(h_num+k_num)+j] = 1;
	    
	    distance = 99999.0;
	    if (distance > sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy)))
	      distance = sqrt((cx-fx)*(cx-fx)+(cy-fy)*(cy-fy));
	    if (distance > sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)))
	      distance = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
	    if (distance > sqrt(cx*cx+cy*cy))
	      distance = sqrt(cx*cx+cy*cy);
	    if (distance > temp) {
	      count = num;
	      temp = distance;
	    }
	
	    a = (gy-fy)/(gx-fx);
	    t = (fy-a*fx)/(cy/cx-a);
	    b = a*(t-fx)+fy;
	    a = t;
	    ratio1 = sqrt((fx-a)*(fx-a)+(fy-b)*(fy-b))/
	      sqrt((gx-fx)*(gx-fx)+(gy-fy)*(gy-fy));
	    ratio2 = sqrt(cx*cx+cy*cy)/sqrt(a*a+b*b);
	    angle2[num] = acos(0.5*(cx*cx+cy*cy+gx*gx+gy*gy-(cx-gx)*(cx-gx)-(cy-gy)*(cy-gy))/
			  sqrt((cx*cx+cy*cy)*(gx*gx+gy*gy)));
	    angle1[num] = acos(0.5*(gx*gx+gy*gy+(cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)-cx*cx-cy*cy)/
			  sqrt(((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy))*(gx*gx+gy*gy)));
	    num++;
	    
	    xx = (1.0-ratio1)*gx;
	    yy = (1.0-ratio1)*gy;
	    a = (fx + ratio2*(xx-fx))/(sqrtof3/2.0);
	    b = fy + ratio2*(yy-fy);
	    if (a <= 0) 
	      m = (int)(a)-1;
	    else 
	      m = (int)(a);
	    n = (int)(2.0*b+0.5);
	    hex_lattice[(m+k_num)*2*(h_num+k_num)+n] = 1;
	    
	    xx = ratio1*fx;
	    yy = ratio1*fy;
	    a = (gx + ratio2*(xx-gx))/(sqrtof3/2.0);
	    b = gy + ratio2*(yy-gy);
	    if (a <= 0) 
	      m = (int)(a)-1;
	    else 
	      m = (int)(a);
	    n = (int)(2.0*b+0.5);
	    hex_lattice[(m+k_num)*2*(h_num+k_num)+n] = 1;
	    
	  }
	}
      }
    }
    
    *numfd3 = num;
    total_num = num;
    printf("number of independent 3-fold subunits: %d \n",num);
        
    cx = angle1[0];
    cy = angle2[0];
    angle1[0] = angle1[count];
    angle2[0] = angle2[count];
    angle1[count] = cx;
    angle2[count] = cy;
    
    for (num = 0; num < total_num; num++)
      LocalSymmetryDetect(angle1[num],angle2[num],num, max_tmp);
	  
    if (numfd5 > 0) {
      num = total_num*60;
      for (n = 0; n < 12; n++) {
	cx = FiveFold[n].x;
	cy = FiveFold[n].y;
	cz = FiveFold[n].z;
	gx = max_tmp[0]/2; //XDIM/2;
	gy = max_tmp[1]/2; //YDIM/2;
	gz = max_tmp[2]/2; //ZDIM/2;
	
	dist = GetIntersection(cx,cy,cz, max_tmp);
	distance = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy)+(cz-gz)*(cz-gz));
	xx = gx + dist*(cx-gx)/distance;
	yy = gy + dist*(cy-gy)/distance;
	zz = gz + dist*(cz-gz)/distance;
	
	TempFold[num].x = xx;
	TempFold[num].y = yy;
	TempFold[num].z = zz;
	num ++;
      }
    }
    
    
    localfold = total_num*60+12*numfd5;
    for (u = 0; u < total_num; u++) {
      num = u*60;
      
      x = TempFold[num].x;
      y = TempFold[num].y;
      z = TempFold[num].z;
      
      for (m = 0; m < localfold; m++) {
	fold_tmp[m].x = TempFold[m].x;
	fold_tmp[m].y = TempFold[m].y;
	fold_tmp[m].z = TempFold[m].z;
      }
      count = 0;
      while (count < 3) {
	min_dist = 999999.0;
	for (m = 0; m < localfold; m++) {
	  if (m != num) {
	    xx = fold_tmp[m].x;
	    yy = fold_tmp[m].y;
	    zz = fold_tmp[m].z;
	    distance = sqrt((x-xx)*(x-xx)+(y-yy)*(y-yy)+(z-zz)*(z-zz));
	    if (distance < min_dist) {
	      min_index = m;
	      min_dist = distance;
	    }
	  }
	}
	
	neighbor[count].x = fold_tmp[min_index].x;
	neighbor[count].y = fold_tmp[min_index].y;
	neighbor[count].z = fold_tmp[min_index].z;
	fold_tmp[min_index].x = -9999.0;
	fold_tmp[min_index].y = -9999.0;
	fold_tmp[min_index].z = -9999.0;
	count++;
      }
      
      NeighborOrder(neighbor, 3);
      
      cx = 0;
      cy = 0;
      cz = 0;
      for (k = 0; k < 3; k++) {
	px = neighbor[k].x-x;
	py = neighbor[k].y-y;
	pz = neighbor[k].z-z;
	if (k < 2) {
	  qx = neighbor[k+1].x-x;
	  qy = neighbor[k+1].y-y;
	  qz = neighbor[k+1].z-z;
	}
	else {
	  qx = neighbor[0].x-x;
	  qy = neighbor[0].y-y;
	  qz = neighbor[0].z-z;
	}
	
	xx = py*qz-pz*qy;
	yy = pz*qx-px*qz;
	zz = px*qy-py*qx;
	distance = sqrt(xx*xx+yy*yy+zz*zz);
	cx += xx/distance;
	cy += yy/distance;
	cz += zz/distance;
      }
      
      distance = sqrt(cx*cx+cy*cy+cz*cz);
      cx /= distance;
      cy /= distance;
      cz /= distance;
      
      score = (x-XDIM/2)*cx + (y-YDIM/2)*cy + (z-ZDIM/2)*cz;
      if (score < 0) {
	cx = -cx;
	cy = -cy;
	cz = -cz;
      }
      
      px = x + 30*cx;
      py = y + 30*cy;
      pz = z + 30*cz;
      qx = x - 30*cx;
      qy = y - 30*cy;
      qz = z - 30*cz;
      
      //LocalRefinement(&px,&py,&pz,&qx,&qy,&qz,numfold3);
            
      score = -LocalSymmetryScore(px,py,pz, qx,qy,qz, orig_tmp, span_tmp, numfold3);
      radius = 2.0f;
      while (radius > 0) {
	score = -score;
	while (score > 0) 
	  SearchNeighbors(&px,&py,&pz, &qx,&qy,&qz, orig_tmp, span_tmp,&score, numfold3, radius);
	radius -= 0.2f;
      }
      
      fx = px;
      fy = py;
      fz = pz;
      gx = qx;
      gy = qy;
      gz = qz;
      
      
      ThreeFold[num].sx = fx;
      ThreeFold[num].sy = fy;
      ThreeFold[num].sz = fz;
      ThreeFold[num].ex = gx;
      ThreeFold[num].ey = gy;
      ThreeFold[num].ez = gz;
      num ++;
      
      
      cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
      cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
      cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
      theta = (float)atan2(cy,cx);
      phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
      for (n = 1; n < 5; n++) {
	sv = Rotate(fx,fy,fz,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	px = sv.x;
	py = sv.y;
	pz = sv.z;
	sv = Rotate(gx,gy,gz,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	qx = sv.x;
	qy = sv.y;
	qz = sv.z;
	temp = 999999.0;
	for (k=u*60; k<num; k++) {
	  a = (float)sqrt((px-ThreeFold[k].sx)*(px-ThreeFold[k].sx)+
			  (py-ThreeFold[k].sy)*(py-ThreeFold[k].sy)+
			  (pz-ThreeFold[k].sz)*(pz-ThreeFold[k].sz));
	  if (a < temp) 
	    temp = a;
	}
	
	if (temp > MINDIST) {
	  ThreeFold[num].sx = px;
	  ThreeFold[num].sy = py;
	  ThreeFold[num].sz = pz;
	  ThreeFold[num].ex = qx;
	  ThreeFold[num].ey = qy;
	  ThreeFold[num].ez = qz;
	  num ++;
	}
	else {
	  ThreeFold[num].sx = -9999.0;
	  ThreeFold[num].sy = -9999.0;
	  ThreeFold[num].sz = -9999.0;
	  ThreeFold[num].ex = -9999.0;
	  ThreeFold[num].ey = -9999.0;
	  ThreeFold[num].ez = -9999.0;
	  num++;
	}
      }
      
      for (m = 1; m < 11; m++) {
	nx = FiveFold[m].x-max_tmp[0]/2; //XDIM/2;
	ny = FiveFold[m].y-max_tmp[1]/2; //YDIM/2;
	nz = FiveFold[m].z-max_tmp[2]/2; //ZDIM/2;
	ax = nz*cy-ny*cz;
	ay = nx*cz-nz*cx;
	az = ny*cx-nx*cy;
	theta = (float)atan2(ay,ax);
	phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
	if (m < 6) {
	  sv1 = Rotate(fx,fy,fz,theta,phi,ANGL1,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	  sv2 = Rotate(gx,gy,gz,theta,phi,ANGL1,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	}
	else {
	  sv1 = Rotate(fx,fy,fz,theta,phi,ANGL2, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
	  sv2 = Rotate(gx,gy,gz,theta,phi,ANGL2,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	}
	
	theta = (float)atan2(ny,nx);
	phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
	for (n = 0; n < 5; n++) {
	  sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	  px = sv.x;
	  py = sv.y;
	  pz = sv.z;
	  sv = Rotate(sv2.x,sv2.y,sv2.z,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
	  qx = sv.x;
	  qy = sv.y;
	  qz = sv.z;
	  temp = 999999.0;
	  for (k=u*60; k<num; k++) {
	    a = (float)sqrt((px-ThreeFold[k].sx)*(px-ThreeFold[k].sx)+
			    (py-ThreeFold[k].sy)*(py-ThreeFold[k].sy)+
			    (pz-ThreeFold[k].sz)*(pz-ThreeFold[k].sz));
	    if (a < temp) 
	      temp = a;
	  }
	  
	  if (temp > MINDIST) {
	    ThreeFold[num].sx = px;
	    ThreeFold[num].sy = py;
	    ThreeFold[num].sz = pz;
	    ThreeFold[num].ex = qx;
	    ThreeFold[num].ey = qy;
	    ThreeFold[num].ez = qz;
	    num ++;
	  }
	  else {
	    ThreeFold[num].sx = -9999.0;
	    ThreeFold[num].sy = -9999.0;
	    ThreeFold[num].sz = -9999.0;
	    ThreeFold[num].ex = -9999.0;
	    ThreeFold[num].ey = -9999.0;
	    ThreeFold[num].ez = -9999.0;
	    num++;
	  }
	}
      }
      
      nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
      ax = nz*cy-ny*cz;
      ay = nx*cz-nz*cx;
      az = ny*cx-nx*cy;
      theta = (float)atan2(ay,ax);
      phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
      sv1 = Rotate(fx,fy,fz,theta,phi,PIE,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      px = sv1.x;
      py = sv1.y;
      pz = sv1.z;
      sv2 = Rotate(gx,gy,gz,theta,phi,PIE,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      qx = sv2.x;
      qy = sv2.y;
      qz = sv2.z; 
      temp = 999999.0;
      for (k=u*60; k<num; k++) {
	a = (float)sqrt((px-ThreeFold[k].sx)*(px-ThreeFold[k].sx)+
			(py-ThreeFold[k].sy)*(py-ThreeFold[k].sy)+
			(pz-ThreeFold[k].sz)*(pz-ThreeFold[k].sz));
	if (a < temp) 
	  temp = a;
      }
      
      if (temp > MINDIST) {
	ThreeFold[num].sx = px;
	ThreeFold[num].sy = py;
	ThreeFold[num].sz = pz;
	ThreeFold[num].ex = qx;
	ThreeFold[num].ey = qy;
	ThreeFold[num].ez = qz;
	num ++;
      }
      else {
	ThreeFold[num].sx = -9999.0;
	ThreeFold[num].sy = -9999.0;
	ThreeFold[num].sz = -9999.0;
	ThreeFold[num].ex = -9999.0;
	ThreeFold[num].ey = -9999.0;
	ThreeFold[num].ez = -9999.0;
	num++;
      }
      
      nx = FiveFold[11].x-max_tmp[0]/2; //XDIM/2;
      ny = FiveFold[11].y-max_tmp[1]/2; //YDIM/2;
      nz = FiveFold[11].z-max_tmp[2]/2; //ZDIM/2;
      theta = (float)atan2(ny,nx);
      phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
      for (n = 1; n < 5; n++) {
	sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
	px = sv.x;
	py = sv.y;
	pz = sv.z;
	sv = Rotate(sv2.x,sv2.y,sv2.z,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
	qx = sv.x;
	qy = sv.y;
	qz = sv.z;
	temp = 999999.0;
	for (k=u*60; k<num; k++) {
	  a = (float)sqrt((px-ThreeFold[k].sx)*(px-ThreeFold[k].sx)+
			  (py-ThreeFold[k].sy)*(py-ThreeFold[k].sy)+
			  (pz-ThreeFold[k].sz)*(pz-ThreeFold[k].sz));
	  if (a < temp) 
	    temp = a;
	}
	
	if (temp > MINDIST) {
	  ThreeFold[num].sx = px;
	  ThreeFold[num].sy = py;
	  ThreeFold[num].sz = pz;
	  ThreeFold[num].ex = qx;
	  ThreeFold[num].ey = qy;
	  ThreeFold[num].ez = qz;
	  num ++;
	}
	else {
	  ThreeFold[num].sx = -9999.0;
	  ThreeFold[num].sy = -9999.0;
	  ThreeFold[num].sz = -9999.0;
	  ThreeFold[num].ex = -9999.0;
	  ThreeFold[num].ey = -9999.0;
	  ThreeFold[num].ez = -9999.0;
	  num++;
	}
      }
    }
   
  }


  free(fold_tmp);
  free(hex_lattice);
  free(TempFold);
  
}



void LocalSymmetryDetect(double angle1, double angle2, int index,float* max_tmp)
{
  int k,num;
  float fx,fy,fz;
  float gx,gy,gz;
  float temp, a, length;
  float cx,cy,cz;
  float nx,ny,nz;
  float ax,ay,az;
  float px,py,pz;
  float qx,qy,qz;
  VECTOR sv,sv1;
  float theta,phi;
  int m,n;
  
  
  angle1 *= 1.2;
  angle2 *= 1.2;
  cx = FiveFold[2].x-max_tmp[0]/2; //XDIM/2;
  cy = FiveFold[2].y-max_tmp[1]/2; //YDIM/2;
  cz = FiveFold[2].z-max_tmp[2]/2; //ZDIM/2;
  theta = (float)atan2(cy,cx);
  phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
  sv = Rotate(FiveFold[0].x,FiveFold[0].y,FiveFold[0].z,theta,phi,-angle1, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
  px = sv.x;
  py = sv.y;
  pz = sv.z;
  
  cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
  cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
  cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;


  theta = (float)atan2(cy,cx);
  phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
  sv = Rotate(FiveFold[2].x,FiveFold[2].y,FiveFold[2].z,theta,phi,angle2,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
  qx = sv.x;
  qy = sv.y;
  qz = sv.z;
  
  fx = (py-max_tmp[1]/2)*(FiveFold[2].z-max_tmp[2]/2 /*ZDIM/2*/)-(pz-max_tmp[2]/2)*(FiveFold[2].y-max_tmp[1]/2); //YDIM/2);
  fy = (pz-max_tmp[2]/2)*(FiveFold[2].x-max_tmp[0]/2 /*XDIM/2*/)-(px-max_tmp[0]/2)*(FiveFold[2].z-max_tmp[2]/2); //ZDIM/2);
  fz = (px-max_tmp[0]/2)*(FiveFold[2].y-max_tmp[1]/2 /*YDIM/2*/)-(py-max_tmp[1]/2)*(FiveFold[2].x-max_tmp[0]/2); //XDIM/2);
//  fx = (py-YDIM/2)*(FiveFold[2].z-max_tmp[2]/2 /*ZDIM/2*/)-(pz-ZDIM/2)*(FiveFold[2].y-max_tmp[1]/2); //YDIM/2);
//  fy = (pz-ZDIM/2)*(FiveFold[2].x-max_tmp[0]/2 /*XDIM/2*/)-(px-XDIM/2)*(FiveFold[2].z-max_tmp[2]/2); //ZDIM/2);
//  fz = (px-XDIM/2)*(FiveFold[2].y-max_tmp[1]/2 /*YDIM/2*/)-(py-YDIM/2)*(FiveFold[2].x-max_tmp[0]/2); //XDIM/2);
  temp = sqrt(fx*fx+fy*fy+fz*fz);
  fx /= temp;
  fy /= temp;
  fz /= temp;
  
  gx = (qy-max_tmp[1]/2 )*(FiveFold[0].z-max_tmp[2]/2 /*ZDIM/2*/)-(qz-max_tmp[2]/2 )*(FiveFold[0].y-max_tmp[1]/2); //YDIM/2);
  gy = (qz-max_tmp[2]/2 )*(FiveFold[0].x-max_tmp[0]/2 /*XDIM/2*/)-(qx-max_tmp[0]/2 )*(FiveFold[0].z-max_tmp[2]/2); //ZDIM/2);
  gz = (qx-max_tmp[0]/2 )*(FiveFold[0].y-max_tmp[1]/2 /*YDIM/2*/)-(qy-max_tmp[1]/2 )*(FiveFold[0].x-max_tmp[0]/2); //XDIM/2);
 // gx = (qy-YDIM/2)*(FiveFold[0].z-max_tmp[2]/2 /*ZDIM/2*/)-(qz-ZDIM/2)*(FiveFold[0].y-max_tmp[1]/2); //YDIM/2);
 // gy = (qz-ZDIM/2)*(FiveFold[0].x-max_tmp[0]/2 /*XDIM/2*/)-(qx-XDIM/2)*(FiveFold[0].z-max_tmp[2]/2); //ZDIM/2);
 // gz = (qx-XDIM/2)*(FiveFold[0].y-max_tmp[1]/2 /*YDIM/2*/)-(qy-YDIM/2)*(FiveFold[0].x-max_tmp[0]/2); //XDIM/2);
  temp = sqrt(gx*gx+gy*gy+gz*gz);
  gx /= temp;
  gy /= temp;
  gz /= temp;

  ax = gy*fz-gz*fy+max_tmp[0]/2; //XDIM/2;
  ay = gz*fx-gx*fz+max_tmp[1]/2; //YDIM/2;
  az = gx*fy-gy*fx+max_tmp[2]/2; //ZDIM/2;
  length = GetLength(ax,ay,az, max_tmp);
  gx = max_tmp[0]/2; //XDIM/2;
  gy = max_tmp[1]/2; //YDIM/2;
  gz = max_tmp[2]/2; //ZDIM/2;
  temp = sqrt((ax-gx)*(ax-gx)+(ay-gy)*(ay-gy)+(az-gz)*(az-gz));
  ax = (ax-gx)*length/temp + gx;
  ay = (ay-gy)*length/temp + gy;
  az = (az-gz)*length/temp + gz;
  
  length = GetIntersection(ax,ay,az, max_tmp);
  temp = sqrt((ax-gx)*(ax-gx)+(ay-gy)*(ay-gy)+(az-gz)*(az-gz));
  fx = gx + length*(ax-gx)/temp;
  fy = gy + length*(ay-gy)/temp;
  fz = gz + length*(az-gz)/temp;


  num = index*60;
  
  TempFold[num].x = fx;
  TempFold[num].y = fy;
  TempFold[num].z = fz;
  num ++;

  cx = FiveFold[0].x-max_tmp[0]/2; //XDIM/2;
  cy = FiveFold[0].y-max_tmp[1]/2; //YDIM/2;
  cz = FiveFold[0].z-max_tmp[2]/2; //ZDIM/2;
  theta = (float)atan2(cy,cx);
  phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
  for (n = 1; n < 5; n++) {
    sv = Rotate(fx,fy,fz,theta,phi,n*2.0f*PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    px = sv.x;
    py = sv.y;
    pz = sv.z;
    temp = 999999.0;
    for (k=index*60; k<num; k++) {
      a = (float)sqrt((px-TempFold[k].x)*(px-TempFold[k].x)+
		      (py-TempFold[k].y)*(py-TempFold[k].y)+
		      (pz-TempFold[k].z)*(pz-TempFold[k].z));
      if (a < temp) 
	temp = a;
    }
    
    if (temp > MINDIST) {
      TempFold[num].x = px;
      TempFold[num].y = py;
      TempFold[num].z = pz;
      num ++;
    }
    else {
      TempFold[num].x = -9999.0;
      TempFold[num].y = -9999.0;
      TempFold[num].z = -9999.0;
      num++;
    }
  }
  
  for (m = 1; m < 11; m++) {
    nx = FiveFold[m].x-max_tmp[0]/2; //XDIM/2;
    ny = FiveFold[m].y-max_tmp[1]/2; //YDIM/2;
    nz = FiveFold[m].z-max_tmp[2]/2; //ZDIM/2;
    ax = nz*cy-ny*cz;
    ay = nx*cz-nz*cx;
    az = ny*cx-nx*cy;
    theta = (float)atan2(ay,ax);
    phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
    if (m < 6) 
      sv1 = Rotate(fx,fy,fz,theta,phi,ANGL1,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    else 
      sv1 = Rotate(fx,fy,fz,theta,phi,ANGL2,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      
    theta = (float)atan2(ny,nx);
    phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
    for (n = 0; n < 5; n++) {
      sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f+PIE/5.0f,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      px = sv.x;
      py = sv.y;
      pz = sv.z;
      temp = 999999.0;
      for (k=index*60; k<num; k++) {
	a = (float)sqrt((px-TempFold[k].x)*(px-TempFold[k].x)+
			(py-TempFold[k].y)*(py-TempFold[k].y)+
			(pz-TempFold[k].z)*(pz-TempFold[k].z));
	if (a < temp) 
	  temp = a;
      }
      
      if (temp > MINDIST) {
		TempFold[num].x = px;
		TempFold[num].y = py;
		TempFold[num].z = pz;
		num ++;
      }
      else {
		TempFold[num].x = -9999.0;
		TempFold[num].y = -9999.0;
		TempFold[num].z = -9999.0;
		num++;
      }
    }
  }
        
  nx = FiveFold[1].x-max_tmp[0]/2; //XDIM/2;
  ny = FiveFold[1].y-max_tmp[1]/2; //YDIM/2;
  nz = FiveFold[1].z-max_tmp[2]/2; //ZDIM/2;
  ax = nz*cy-ny*cz;
  ay = nx*cz-nz*cx;
  az = ny*cx-nx*cy;
  theta = (float)atan2(ay,ax);
  phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
  sv1 = Rotate(fx,fy,fz,theta,phi,PIE,  max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
  px = sv1.x;
  py = sv1.y;
  pz = sv1.z;
  
  temp = 999999.0;
  for (k=index*60; k<num; k++) {
    a = (float)sqrt((px-TempFold[k].x)*(px-TempFold[k].x)+
		    (py-TempFold[k].y)*(py-TempFold[k].y)+
		    (pz-TempFold[k].z)*(pz-TempFold[k].z));
    if (a < temp) 
      temp = a;
  }
  
  if (temp > MINDIST) {
    TempFold[num].x = px;
    TempFold[num].y = py;
    TempFold[num].z = pz;
    num ++;
  }
  else {
    TempFold[num].x = -9999.0;
    TempFold[num].y = -9999.0;
    TempFold[num].z = -9999.0;
    num++;
  }
  
  nx = FiveFold[11].x-max_tmp[0]/2; //XDIM/2;
  ny = FiveFold[11].y-max_tmp[1]/2; //YDIM/2;
  nz = FiveFold[11].z-max_tmp[2]/2; //ZDIM/2;
  theta = (float)atan2(ny,nx);
  phi = (float)atan2(nz, sqrt(nx*nx+ny*ny));
  for (n = 1; n < 5; n++) {
    sv = Rotate(sv1.x,sv1.y,sv1.z,theta,phi,n*2.0f*PIE/5.0f, max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
    px = sv.x;
    py = sv.y;
    pz = sv.z;
    
    temp = 999999.0;
    for (k=index*60; k<num; k++) {
      a = (float)sqrt((px-TempFold[k].x)*(px-TempFold[k].x)+
		      (py-TempFold[k].y)*(py-TempFold[k].y)+
		      (pz-TempFold[k].z)*(pz-TempFold[k].z));
      if (a < temp) 
	temp = a;
    }
	
    if (temp > MINDIST) {
      TempFold[num].x = px;
      TempFold[num].y = py;
      TempFold[num].z = pz;
      num ++;
    }
    else {
      TempFold[num].x = -9999.0;
      TempFold[num].y = -9999.0;
      TempFold[num].z = -9999.0;
      num++;
    }
  }
}


float GetIntersection(float ex, float ey, float ez, float* max_tmp)
{
  int i,j;
  VECTOR dv1,dv2;
  float x,y,z,xx;
  float rf_sample = 2;
  int size,number;
  float avg_length, length;
  float sx,sy,sz;
  int max_size = 40;

  sx = max_tmp[0]/2; //(float)(XDIM/2);
  sy = max_tmp[1]/2; //(float)(YDIM/2);
  sz = max_tmp[2]/2; //(float)(ZDIM/2);
  xx = sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    dv1.x = 1;
    dv1.y = 0;
    dv1.z = 0;
  }
  else {
    dv1.x = (ey-sy)/xx;
    dv1.y = (sx-ex)/xx;
    dv1.z = 0;
  }
  x = (sy-ey)*dv1.z-(sz-ez)*dv1.y;
  y = (sz-ez)*dv1.x-(sx-ex)*dv1.z;
  z = (sx-ex)*dv1.y-(sy-ey)*dv1.x;
  xx = sqrt(x*x+y*y+z*z);
  dv2.x = x/xx;
  dv2.y = y/xx;
  dv2.z = z/xx;
  
  length = GetLength(ex,ey,ez, max_tmp);
  if (length >= 0) {
    avg_length = length;
    number = 1;
  }
  else {
    number = 0;
    avg_length = 0;
  }
  
  size = 1;
  while (number < max_size) {
    for (j = -size; j <= size; j++) {
      i = -size;
      x = ex + i*rf_sample*dv1.x + j*rf_sample*dv2.x;
      y = ey + i*rf_sample*dv1.y + j*rf_sample*dv2.y;
      z = ez + i*rf_sample*dv1.z + j*rf_sample*dv2.z;
	
      length = GetLength(x,y,z, max_tmp);
      if (length >= 0 && number < max_size) {
	avg_length += length;
	number++;
      }

      i = size;
      x = ex + i*rf_sample*dv1.x + j*rf_sample*dv2.x;
      y = ey + i*rf_sample*dv1.y + j*rf_sample*dv2.y;
      z = ez + i*rf_sample*dv1.z + j*rf_sample*dv2.z;
	
      length = GetLength(x,y,z,max_tmp);
      if (length >= 0 && number < max_size) {
	avg_length += length;
	number++;
      }
    }

    for (i = -size+1; i <= size-1; i++) {
      j = -size;
      x = ex + i*rf_sample*dv1.x + j*rf_sample*dv2.x;
      y = ey + i*rf_sample*dv1.y + j*rf_sample*dv2.y;
      z = ez + i*rf_sample*dv1.z + j*rf_sample*dv2.z;
	
      length = GetLength(x,y,z,max_tmp);
      if (length >= 0 && number < max_size) {
	avg_length += length;
	number++;
      }

      j = size;
      x = ex + i*rf_sample*dv1.x + j*rf_sample*dv2.x;
      y = ey + i*rf_sample*dv1.y + j*rf_sample*dv2.y;
      z = ez + i*rf_sample*dv1.z + j*rf_sample*dv2.z;
	
      length = GetLength(x,y,z,max_tmp);
      if (length >= 0 && number < max_size) {
	avg_length += length;
	number++;
      }
    }

    size++;
  }
  
  avg_length /= (float)(max_size);
  return(avg_length);

}


float GetLocalLength(float sx, float sy, float sz, float ex, float ey, float ez,float* orig_tmp, float* span_tmp, float tlow)
{
  float x,y,z;
  float xx,yy,zz;
  float xxx,yyy,zzz;
  float s_x,s_y,s_z;
  float e_x,e_y,e_z;
  float f_x,f_y,f_z;
  float g_x,g_y,g_z;
  float max_size;
  float length,max_length,step;
  float dx,dy,dz;
  char flag;
  float average;
  float a[3][3],b[3][3];
  float theta, phi;
  int m;
  VECTOR e[32];
  float x00,x01,x10,x11,y0,y1;
  

  theta = (float)atan2(ey-sy,ex-sx);
  phi = (float)atan2(ez-sz, sqrt((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)));
  
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

  xx = (float)sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    x = 0.5f*min_radius+sx;
    y = sy;
    z = sz;
  }
  else {
    x = 0.5f*min_radius*(sy-ey)/xx+sx;
    y = 0.5f*min_radius*(ex-sx)/xx+sy;
    z = sz;
  }

  e[0].x = x;
  e[0].y = y;
  e[0].z = z;

  x = x-sx;
  y = y-sy;
  z = z-sz;
  
  xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
  yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
  zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
  
  for (m = 1; m < 32; m++) {
    x = (float)(cos(PIE*(float)(m)/16.0f)*xx - 
      sin(PIE*(float)(m)/16.0f)*yy);
    y = (float)(sin(PIE*(float)(m)/16.0f)*xx + 
      cos(PIE*(float)(m)/16.0f)*yy);
    z = zz;
    
    e[m].x = b[0][0]*x+b[0][1]*y+b[0][2]*z+sx;
    e[m].y = b[1][0]*x+b[1][1]*y+b[1][2]*z+sy;
    e[m].z = b[2][0]*x+b[2][1]*y+b[2][2]*z+sz;
  }
    
    
  max_length = (float)sqrt((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)+(sz-ez)*(sz-ez));
  dx = (ex-sx)/max_length;
  dy = (ey-sy)/max_length;
  dz = (ez-sz)/max_length;

  max_size = 0;
  float stepp[3];
  stepp[0] = span_tmp[0];
  stepp[1] = span_tmp[1];
  stepp[2] = span_tmp[2];

//  step = 0;
  flag = 0;
  while (stepp[0] <= max_length && stepp[1]<= max_length && stepp[2]<= max_length) {
  //while(step<=max_length){

    average = 0;
    for (m = 0; m < 32; m++) {
//      xxx = e[m].x+step*dx;
//      yyy = e[m].y+step*dy;
//      zzz = e[m].z+step*dz;

	xxx = (e[m].x+ stepp[0]*dx - orig_tmp[0])/span_tmp[0];
	yyy = (e[m].y+ stepp[1]*dy - orig_tmp[1])/span_tmp[1];
	zzz = (e[m].z+ stepp[2]*dz - orig_tmp[2])/span_tmp[2];


      x00 = dataset[IndexVect((int)xxx,(int)yyy,(int)zzz)]*((int)xxx+1-xxx)+
	dataset[IndexVect((int)xxx+1,(int)yyy,(int)zzz)]*(xxx-(int)xxx);
      x01 = dataset[IndexVect((int)xxx,(int)yyy,(int)zzz+1)]*((int)xxx+1-xxx)+
	dataset[IndexVect((int)xxx+1,(int)yyy,(int)zzz+1)]*(xxx-(int)xxx);
      x10 = dataset[IndexVect((int)xxx,(int)yyy+1,(int)zzz)]*((int)xxx+1-xxx)+
	dataset[IndexVect((int)xxx+1,(int)yyy+1,(int)zzz)]*(xxx-(int)xxx);
      x11 = dataset[IndexVect((int)xxx,(int)yyy+1,(int)zzz+1)]*((int)xxx+1-xxx)+
	dataset[IndexVect((int)xxx+1,(int)yyy+1,(int)zzz+1)]*(xxx-(int)xxx);
      y0  = x00*((int)yyy+1-yyy) + x10*(yyy-(int)yyy);
      y1  = x01*((int)yyy+1-yyy) + x11*(yyy-(int)yyy);
      average += y0*((int)zzz+1-zzz) + y1*(zzz-(int)zzz);
    }
    average /= 32.0f;

    if (average >= tlow && flag == 0) {
      flag = 1;
 //     s_x = sx+step*dx;
 //     s_y = sy+step*dy;
 //     s_z = sz+step*dz;
 		s_x = sx + stepp[0]*dx;
		s_y = sy + stepp[1]*dy;
		s_z = sz + stepp[2]*dz;
    }
    if (average < tlow && flag == 1) {
      flag = 0;
  //    e_x = sx+step*dx;
   //   e_y = sy+step*dy;
   //   e_z = sz+step*dz;
      e_x = sx+stepp[0]*dx;
      e_y = sy+stepp[1]*dy;
      e_z = sz+stepp[2]*dz;
      length = sqrt((s_x-e_x)*(s_x-e_x)+(s_y-e_y)*(s_y-e_y)+(s_z-e_z)*(s_z-e_z));
      if (length > max_size) {
	max_size = length;
	f_x = s_x;
	f_y = s_y;
	f_z = s_z;
	g_x = e_x;
	g_y = e_y;
	g_z = e_z;
      }
    }
//    step = step + 1.0f;

	stepp[0]+= span_tmp[0];
	stepp[1]+= span_tmp[1];
	stepp[2]+= span_tmp[2];
  }
  
  if (max_size == 0) {
    length = -1.0f;
  }
  else {
    s_x = 0.5f*(f_x+g_x);
    s_y = 0.5f*(f_y+g_y);
    s_z = 0.5f*(f_z+g_z);
    length = (float)sqrt((sx-s_x)*(sx-s_x)+(sy-s_y)*(sy-s_y)+(sz-s_z)*(sz-s_z));
  }

  return(length);
  
}


float GetLength(float ex, float ey, float ez, float* max_tmp)
{
  int i,j,k;
  int m,n,l;
  float s_x,s_y,s_z;
  float e_x,e_y,e_z;
  float f_x,f_y,f_z;
  float g_x,g_y,g_z;
  float max_size;
  float length,step;
  float dx,dy,dz;
  float sx,sy,sz;
  char flag,alive;
  float radius = 5;

  
  sx = max_tmp[0]/2; //(float)(XDIM/2);
  sy = max_tmp[1]/2; //(float)(YDIM/2);
  sz = max_tmp[2]/2; //(float)(ZDIM/2);

  length = (float)sqrt((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)+(sz-ez)*(sz-ez));
  dx = (ex-sx)/length;
  dy = (ey-sy)/length;
  dz = (ez-sz)/length;

  max_size = 0;
//  step = 1;
  float stepp[3];
  stepp[0] = max_tmp[0]/(XDIM-1);
  stepp[1] = max_tmp[1]/(YDIM-1);
  stepp[2] = max_tmp[2]/(ZDIM-1);

  float unitstep[3];
  unitstep[0]= stepp[0];
  unitstep[1] = stepp[1];
  unitstep[2] = stepp[2];

  flag = 0;
  while (1) {
 //   i = (int)(sx+step*dx+0.5);
//    j = (int)(sy+step*dy+0.5);
 //   k = (int)(sz+step*dz+0.5);
    
    i = (int)((sx+stepp[0]*dx)/unitstep[0]+0.5);
    j = (int)((sy+stepp[1]*dy)/unitstep[1]+0.5);
    k = (int)((sz+stepp[2]*dz)/unitstep[2]+0.5);
    if (i < 0 || i >= XDIM ||
        j < 0 || j >= YDIM ||
        k < 0 || k >= ZDIM) 
      break;

    if (flag == 0) {
      alive = 0;
      for (l = max2(0,k-radius); l <= min2(ZDIM-1,k+radius); l++)
        for (n = max2(0,j-radius); n <= min2(YDIM-1,j+radius); n++)
          for (m = max2(0,i-radius); m <= min2(XDIM-1,i+radius); m++) {
            if (sqrt((m-i)*(m-i)+(n-j)*(n-j)+(l-k)*(l-k)) <= radius &&
                dataset[IndexVect(m,n,l)] > 0) { 
              alive = 1;
              break;
            }
          }
      if (alive) {
        flag = 1;
        s_x = sx+stepp[0]*dx;
        s_y = sy+stepp[1]*dy;
        s_z = sz+stepp[2]*dz;
//		s_x = sx+step*dx;
//        s_y = sy+step*dy;
//        s_z = sz+step*dz;

      }
    }
    else if (flag == 1) {
      alive = 1;
      for (l = max2(0,k-radius); l <= min2(ZDIM-1,k+radius); l++)
        for (n = max2(0,j-radius); n <= min2(YDIM-1,j+radius); n++)
          for (m = max2(0,i-radius); m <= min2(XDIM-1,i+radius); m++) {
            if (sqrt((m-i)*(m-i)+(n-j)*(n-j)+(l-k)*(l-k)) <= radius &&
                dataset[IndexVect(m,n,l)] > 0) 
              alive = 0;
          }
      if (!alive) {
        flag = 0;
        e_x = sx+stepp[0]*dx;
        e_y = sy+stepp[1]*dy;
        e_z = sz+stepp[2]*dz;

//		e_x = sx+step*dx;
//        e_y = sy+step*dy;
//        e_z = sz+step*dz;
        length = sqrt((s_x-e_x)*(s_x-e_x)+(s_y-e_y)*(s_y-e_y)+(s_z-e_z)*(s_z-e_z));
        if (length > max_size) {
          max_size = length;
          f_x = s_x;
          f_y = s_y;
          f_z = s_z;
          g_x = e_x;
          g_y = e_y;
          g_z = e_z;
        }
      }
    }
//    step = step + 1.0;
	  stepp[0]+= unitstep[0];
	  stepp[1]+= unitstep[1];
	  stepp[2]+= unitstep[2];
  }
  
  if (max_size == 0) {
    length = -1.0f;
  }
  else {
    s_x = 0.5*(f_x+g_x);
    s_y = 0.5*(f_y+g_y);
    s_z = 0.5*(f_z+g_z);
    length = (float)sqrt((sx-s_x)*(sx-s_x)+(sy-s_y)*(sy-s_y)+(sz-s_z)*(sz-s_z));
  }
   return(length);
  
}




void SearchNeighbors(float *s_x, float *s_y, float *s_z, 
		float *e_x, float *e_y, float *e_z, float* orig_tmp, float* span_tmp,
		float *min_score, int foldnum, float max_rd)
{
  float x,y,z;
  float xx,yy,zz;
  float xxx,yyy,zzz;
  float sx,sy,sz;
  float ex,ey,ez;
  float a[3][3],b[3][3];
  float score,mscore;
  float theta, phi;
  int m,n;
  VECTOR s[17],e[17];
  

  sx = *s_x;
  sy = *s_y;
  sz = *s_z;
  ex = *e_x;
  ey = *e_y;
  ez = *e_z;
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


  mscore = *min_score;
  
  xx = (float)sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    x = max_rd+ex;
    y = ey;
    z = ez;
  }
  else {
    x = max_rd*(ey-sy)/xx+ex;
    y = max_rd*(sx-ex)/xx+ey;
    z = ez;
  }

  e[0].x = x;
  e[0].y = y;
  e[0].z = z;

  x = x-ex;
  y = y-ey;
  z = z-ez;
  
  xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
  yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
  zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
  
  for (m = 1; m < 16; m++) {
    x = (float)(cos(PIE*(float)(m)/8.0f)*xx - 
      sin(PIE*(float)(m)/8.0f)*yy);
    y = (float)(sin(PIE*(float)(m)/8.0f)*xx + 
      cos(PIE*(float)(m)/8.0f)*yy);
    z = zz;
    
    xxx = b[0][0]*x+b[0][1]*y+b[0][2]*z+ex;
    yyy = b[1][0]*x+b[1][1]*y+b[1][2]*z+ey;
    zzz = b[2][0]*x+b[2][1]*y+b[2][2]*z+ez;
 
    e[m].x = xxx;
    e[m].y = yyy;
    e[m].z = zzz;
  
  }

  e[16].x = ex;
  e[16].y = ey;
  e[16].z = ez;

  xx = (float)sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    x = max_rd+sx;
    y = sy;
    z = sz;
  }
  else {
    x = max_rd*(sy-ey)/xx+sx;
    y = max_rd*(ex-sx)/xx+sy;
    z = sz;
  }

  s[0].x = x;
  s[0].y = y;
  s[0].z = z;

  x = x-sx;
  y = y-sy;
  z = z-sz;
  
  xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
  yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
  zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
  
  for (m = 1; m < 16; m++) {
    x = (float)(cos(PIE*(float)(m)/8.0f)*xx - 
      sin(PIE*(float)(m)/8.0f)*yy);
    y = (float)(sin(PIE*(float)(m)/8.0f)*xx + 
      cos(PIE*(float)(m)/8.0f)*yy);
    z = zz;
    
    xxx = b[0][0]*x+b[0][1]*y+b[0][2]*z+sx;
    yyy = b[1][0]*x+b[1][1]*y+b[1][2]*z+sy;
    zzz = b[2][0]*x+b[2][1]*y+b[2][2]*z+sz;
    
    s[m].x = xxx;
    s[m].y = yyy;
    s[m].z = zzz;
  
  }
  
  s[16].x = sx;
  s[16].y = sy;
  s[16].z = sz;

  
  for(n = 0; n < 17; n++)
    for(m = 0; m < 17; m++) {
      score = LocalSymmetryScore(s[n].x,s[n].y,s[n].z,e[m].x,e[m].y,e[m].z, orig_tmp, span_tmp, foldnum);
      if (score < mscore) {
	mscore = score;
	*s_x = s[n].x;
	*s_y = s[n].y;
	*s_z = s[n].z;
	*e_x = e[m].x;
	*e_y = e[m].y;
	*e_z = e[m].z;
	
      }
    }


  if (*min_score == mscore) 
    *min_score = -mscore;
  else
    *min_score = mscore;

}



void SearchNeighbors2(float *s_x, float *s_y, float *s_z, 
		float *e_x, float *e_y, float *e_z, float* orig_tmp, float* span_tmp,
		float *min_score, int foldnum, float max_rd,int index)
{
  float x,y,z;
  float xx,yy,zz;
  float xxx,yyy,zzz;
  float sx,sy,sz;
  float ex,ey,ez;
  float a[3][3],b[3][3];
  float score,mscore;
  float theta, phi;
  int m,n;
  VECTOR s[17],e[17];
  

  sx = *s_x;
  sy = *s_y;
  sz = *s_z;
  ex = *e_x;
  ey = *e_y;
  ez = *e_z;
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


  mscore = *min_score;
  
  xx = (float)sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    x = max_rd+ex;
    y = ey;
    z = ez;
  }
  else {
    x = max_rd*(ey-sy)/xx+ex;
    y = max_rd*(sx-ex)/xx+ey;
    z = ez;
  }

  e[0].x = x;
  e[0].y = y;
  e[0].z = z;

  x = x-ex;
  y = y-ey;
  z = z-ez;
  
  xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
  yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
  zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
  
  for (m = 1; m < 16; m++) {
    x = (float)(cos(PIE*(float)(m)/8.0f)*xx - 
      sin(PIE*(float)(m)/8.0f)*yy);
    y = (float)(sin(PIE*(float)(m)/8.0f)*xx + 
      cos(PIE*(float)(m)/8.0f)*yy);
    z = zz;
    
    xxx = b[0][0]*x+b[0][1]*y+b[0][2]*z+ex;
    yyy = b[1][0]*x+b[1][1]*y+b[1][2]*z+ey;
    zzz = b[2][0]*x+b[2][1]*y+b[2][2]*z+ez;
 
    e[m].x = xxx;
    e[m].y = yyy;
    e[m].z = zzz;
  
  }

  e[16].x = ex;
  e[16].y = ey;
  e[16].z = ez;

  xx = (float)sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    x = max_rd+sx;
    y = sy;
    z = sz;
  }
  else {
    x = max_rd*(sy-ey)/xx+sx;
    y = max_rd*(ex-sx)/xx+sy;
    z = sz;
  }

  s[0].x = x;
  s[0].y = y;
  s[0].z = z;

  x = x-sx;
  y = y-sy;
  z = z-sz;
  
  xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
  yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
  zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
  
  for (m = 1; m < 16; m++) {
    x = (float)(cos(PIE*(float)(m)/8.0f)*xx - 
      sin(PIE*(float)(m)/8.0f)*yy);
    y = (float)(sin(PIE*(float)(m)/8.0f)*xx + 
      cos(PIE*(float)(m)/8.0f)*yy);
    z = zz;
    
    xxx = b[0][0]*x+b[0][1]*y+b[0][2]*z+sx;
    yyy = b[1][0]*x+b[1][1]*y+b[1][2]*z+sy;
    zzz = b[2][0]*x+b[2][1]*y+b[2][2]*z+sz;
    
    s[m].x = xxx;
    s[m].y = yyy;
    s[m].z = zzz;
  
  }
  
  s[16].x = sx;
  s[16].y = sy;
  s[16].z = sz;

  
  for(n = 0; n < 17; n++)
    for(m = 0; m < 17; m++) {
      score = LocalSymmetryScore2(s[n].x,s[n].y,s[n].z,
				  e[m].x,e[m].y,e[m].z, orig_tmp, span_tmp, foldnum,index);
      
      if (score < mscore) {
	mscore = score;
	*s_x = s[n].x;
	*s_y = s[n].y;
	*s_z = s[n].z;
	*e_x = e[m].x;
	*e_y = e[m].y;
	*e_z = e[m].z;
	
      }
    }


  if (*min_score == mscore) 
    *min_score = -mscore;
  else
    *min_score = mscore;
}



float LocalSymmetryScore(float sx, float sy, float sz, 
			 float ex, float ey, float ez,float* orig_tmp, float* span_tmp,  int foldnum)
{
  int i,j,k;
  float d3,d2;
  int m, num;
  float x,y,z;
  float xx,yy,zz;
  float average,variance;
  float a[3][3],b[3][3];
  CPNT *critical_tmp;
  float asymmetry;
  float theta, phi;
  float distance;
  float a1,b1,c1;
  float x00,x01,x10,x11,y0,y1;
  float dx,dy,dz;


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


  a1 = sx-ex;
  b1 = sy-ey;
  c1 = sz-ez;
  d3 = (float)sqrt(a1*a1+b1*b1+c1*c1);
  distance = min_radius;
  
  num = 0;
  asymmetry = 0;
  critical_tmp = critical_start;
  while (critical_tmp != NULL) {
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;

    x = orig_tmp[0]+span_tmp[0]*i-ex;
    y = orig_tmp[1]+span_tmp[1]*j-ey;
    z = orig_tmp[2]+span_tmp[2]*k-ez;
  //  x = i-ex;
  //  y = j-ey;
  // z = k-ez;
    xx = b1*z-c1*y;
    yy = x*c1-a1*z;
    zz = a1*y-x*b1;
    d2 = (float)(sqrt(xx*xx+yy*yy+zz*zz)/d3);
    
    if (d2 > distance ||
	a1*x+b1*y+c1*z < 0) 
      critical_tmp = critical_tmp->next;
    else {
      density[0] = dataset[IndexVect(i,j,k)];
      
      xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
      yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
      zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
      
      average = density[0];
      for (m = 1; m < foldnum; m++) {
	x = (float)(cos(2*PIE*(float)(m)/(float)(foldnum))*xx - 
		    sin(2*PIE*(float)(m)/(float)(foldnum))*yy);
	y = (float)(sin(2*PIE*(float)(m)/(float)(foldnum))*xx + 
		    cos(2*PIE*(float)(m)/(float)(foldnum))*yy);
	z = zz;

	dx = (b[0][0]*x+b[0][1]*y+b[0][2]*z+ex-orig_tmp[0])/span_tmp[0];
	dy = (b[1][0]*x+b[1][1]*y+b[1][2]*z+ey-orig_tmp[1])/span_tmp[1];
	dz = (b[2][0]*x+b[2][1]*y+b[2][2]*z+ez-orig_tmp[2])/span_tmp[2];
    
//	dx = b[0][0]*x+b[0][1]*y+b[0][2]*z+ex;
//	dy = b[1][0]*x+b[1][1]*y+b[1][2]*z+ey;
//	dz = b[2][0]*x+b[2][1]*y+b[2][2]*z+ez;
	
	x00 = dataset[IndexVect((int)dx,(int)dy,(int)dz)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy,(int)dz)]*(dx-(int)dx);
	x01 = dataset[IndexVect((int)dx,(int)dy,(int)dz+1)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy,(int)dz+1)]*(dx-(int)dx);
	x10 = dataset[IndexVect((int)dx,(int)dy+1,(int)dz)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy+1,(int)dz)]*(dx-(int)dx);
	x11 = dataset[IndexVect((int)dx,(int)dy+1,(int)dz+1)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy+1,(int)dz+1)]*(dx-(int)dx);
	y0  = x00*((int)dy+1-dy) + x10*(dy-(int)dy);
	y1  = x01*((int)dy+1-dy) + x11*(dy-(int)dy);
	density[m] = y0*((int)dz+1-dz) + y1*(dz-(int)dz);
	
	average += density[m];
      }

      average /= (float)(foldnum);
      
      variance = 0;
      for (m = 0; m < foldnum; m++) 
	variance += (float)fabs(density[m]-average);
      
      asymmetry += variance/(float)(foldnum);
      num ++;
      
      critical_tmp = critical_tmp->next;
    }
    
  }
  
  if (num > 0)
    return(asymmetry/(float)(num));
  else 
    return(999999.0);
  
}


float LocalSymmetryScore2(float sx, float sy, float sz, 
			 float ex, float ey, float ez, float* orig_tmp, float* span_tmp, int foldnum,int index)
{
  int i,j,k;
  int m, num;
  float x,y,z;
  float xx,yy,zz;
  float average,variance;
  float a[3][3],b[3][3];
  CPNT *critical_tmp;
  float asymmetry;
  float theta, phi;
  float x00,x01,x10,x11,y0,y1;
  float dx,dy,dz;
  

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

  num = 0;
  asymmetry = 0;
  symmetry_score = 0;
  critical_tmp = critical_start;
  while (critical_tmp != NULL) {
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;

    x = orig_tmp[0]+span_tmp[0]*i-ex;
    y = orig_tmp[1]+span_tmp[1]*j-ey;
    z = orig_tmp[2]+span_tmp[2]*k-ez;

//    x = i-ex;
//    y = j-ey;
//    z = k-ez;
    
    if (classify[IndexVect(i,j,k)] != index) 
      critical_tmp = critical_tmp->next;
    else {
      density[0] = dataset[IndexVect(i,j,k)];
      
      xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
      yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
      zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
      
      average = density[0];
      for (m = 1; m < foldnum; m++) {
	x = (float)(cos(2*PIE*(float)(m)/(float)(foldnum))*xx - 
	     sin(2*PIE*(float)(m)/(float)(foldnum))*yy);
	y = (float)(sin(2*PIE*(float)(m)/(float)(foldnum))*xx + 
	     cos(2*PIE*(float)(m)/(float)(foldnum))*yy);
	z = zz;
	
	dx = (b[0][0]*x+b[0][1]*y+b[0][2]*z+ex-orig_tmp[0])/span_tmp[0];
	dy = (b[1][0]*x+b[1][1]*y+b[1][2]*z+ey-orig_tmp[1])/span_tmp[1];
	dz = (b[2][0]*x+b[2][1]*y+b[2][2]*z+ez-orig_tmp[2])/span_tmp[2];
    
//	dx = b[0][0]*x+b[0][1]*y+b[0][2]*z+ex;
//	dy = b[1][0]*x+b[1][1]*y+b[1][2]*z+ey;
//	dz = b[2][0]*x+b[2][1]*y+b[2][2]*z+ez;
    
	x00 = dataset[IndexVect((int)dx,(int)dy,(int)dz)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy,(int)dz)]*(dx-(int)dx);
	x01 = dataset[IndexVect((int)dx,(int)dy,(int)dz+1)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy,(int)dz+1)]*(dx-(int)dx);
	x10 = dataset[IndexVect((int)dx,(int)dy+1,(int)dz)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy+1,(int)dz)]*(dx-(int)dx);
	x11 = dataset[IndexVect((int)dx,(int)dy+1,(int)dz+1)]*((int)dx+1-dx)+
	  dataset[IndexVect((int)dx+1,(int)dy+1,(int)dz+1)]*(dx-(int)dx);
	y0  = x00*((int)dy+1-dy) + x10*(dy-(int)dy);
	y1  = x01*((int)dy+1-dy) + x11*(dy-(int)dy);
	density[m] = y0*((int)dz+1-dz) + y1*(dz-(int)dz);
	
	average += density[m];
      }

      
      average /= (float)(foldnum);
	
      variance = 0;
      for (m = 0; m < foldnum; m++) 
	variance += (float)fabs(density[m]-average);
      
      asymmetry += variance/(float)(foldnum);
      num ++;
      
      symmetry_score += 1.0-variance/(average*(float)(foldnum));
      
      critical_tmp = critical_tmp->next;
    }
  }

  if (num > 0) {
    symmetry_score /= (float)(num);
    return(asymmetry/(float)(num));
  }
  else {
    symmetry_score = 999999.0;
    return(999999.0);
  }
  
}



void LocalRefinement(float *fx, float *fy, float *fz,
		     float *gx, float *gy, float *gz, float* orig_tmp, float* span_tmp, int numfold)
{
  int i,j;
  int m,n;
  float sx,sy,sz;
  float ex,ey,ez;
  VECTOR dv1,dv2;
  float x,y,z;
  float xx,yy,zz;
  float min_rotation, temp;
  int rf_Rd = 5;
  float rf_sample = 0.5;


  ex = *gx;
  ey = *gy;
  ez = *gz;
  sx = *fx;
  sy = *fy;
  sz = *fz;
  

  xx = sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    dv1.x = 1;
    dv1.y = 0;
    dv1.z = 0;
  }
  else {
    dv1.x = (ey-sy)/xx;
    dv1.y = (sx-ex)/xx;
    dv1.z = 0;
  }
  x = (sy-ey)*dv1.z-(sz-ez)*dv1.y;
  y = (sz-ez)*dv1.x-(sx-ex)*dv1.z;
  z = (sx-ex)*dv1.y-(sy-ey)*dv1.x;
  xx = sqrt(x*x+y*y+z*z);
  dv2.x = x/xx;
  dv2.y = y/xx;
  dv2.z = z/xx;
  
  min_rotation = 999999.0;
  for (j = -rf_Rd; j <= rf_Rd; j++) 
    for (i = -rf_Rd; i <= rf_Rd; i++) {
      x = sx + i*rf_sample*dv1.x + j*rf_sample*dv2.x;
      y = sy + i*rf_sample*dv1.y + j*rf_sample*dv2.y;
      z = sz + i*rf_sample*dv1.z + j*rf_sample*dv2.z;

      for (n = -rf_Rd; n <= rf_Rd; n++)
	for (m = -rf_Rd; m <= rf_Rd; m++) {
	  xx = ex + m*rf_sample*dv1.x + n*rf_sample*dv2.x;
	  yy = ey + m*rf_sample*dv1.y + n*rf_sample*dv2.y;
	  zz = ez + m*rf_sample*dv1.z + n*rf_sample*dv2.z;
    
	  temp = LocalSymmetryScore(x,y,z,xx,yy,zz,orig_tmp, span_tmp, numfold);
	  if (temp < min_rotation) {
	    min_rotation = temp;
	    *fx = x;
	    *fy = y;
	    *fz = z;
	    *gx = xx;
	    *gy = yy;
	    *gz = zz;
	  }
	}
    }
    
}



void NeighborOrder(VECTOR *neighbor, int polynum)
{
  float xx,yy,zz;
  float dist,min_dist;
  int i,j, neigh_index;

  
  j = 0;
  while (j < polynum-1) {
    xx = neighbor[j].x;
    yy = neighbor[j].y;
    zz = neighbor[j].z;
    j++;
    
    min_dist = 999999.0f;
    for (i = j; i < polynum; i++) {
      dist = (float)sqrt((xx-neighbor[i].x)*(xx-neighbor[i].x)+
		  (yy-neighbor[i].y)*(yy-neighbor[i].y)+
		  (zz-neighbor[i].z)*(zz-neighbor[i].z));
      if (dist < min_dist) {
	min_dist = dist;
	neigh_index = i;
      }
    }
    xx = neighbor[j].x;
    yy = neighbor[j].y;
    zz = neighbor[j].z;
    neighbor[j].x = neighbor[neigh_index].x;
    neighbor[j].y = neighbor[neigh_index].y;
    neighbor[j].z = neighbor[neigh_index].z;
    neighbor[neigh_index].x = xx;
    neighbor[neigh_index].y = yy;
    neighbor[neigh_index].z = zz;
  }
  
}




void DrawLine(float sx, float sy, float sz, float ex, float ey, float ez, float radius)
{
  float x,y,z;
  float xx,yy,zz;
  float xxx,yyy,zzz;
  float a[3][3],b[3][3];
  float theta, phi;
  int m;
  

  theta = (float)atan2(sy-ey,sx-ex);
  phi = (float)atan2(sz-ez, sqrt((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)));
  
  a[0][0] = (float)(cos(0.5*PIE-phi)*cos(theta));
  a[0][1] = (float)(cos(0.5*PIE-phi)*sin(theta));
  a[0][2] = (float)-sin(0.5*PIE-phi);
  a[1][0] = (float)-sin(theta);
  a[1][1] = (float)cos(theta);
  a[1][2] = 0;
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
  b[2][1] = 0;
  b[2][2] = (float)cos(0.5*PIE-phi);

  
  xx = (float)sqrt((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex));
  if (xx == 0) {
    x = radius+ex;
    y = ey;
    z = ez;
  }
  else {
    x = radius*(ey-sy)/xx+ex;
    y = radius*(sx-ex)/xx+ey;
    z = ez;
  }
  
  vertex[v_num].x = x;
  vertex[v_num].y = y;
  vertex[v_num].z = z;
  vertex[v_num+1].x = x+sx-ex;
  vertex[v_num+1].y = y+sy-ey;
  vertex[v_num+1].z = z+sz-ez;
    
  
  x = x-ex;
  y = y-ey;
  z = z-ez;
  
  xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
  yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
  zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
  
  for (m = 1; m < PolyNum; m++) {
    x = (float)(cos(2*PIE*(float)(m)/(float)(PolyNum))*xx - 
      sin(2*PIE*(float)(m)/(float)(PolyNum))*yy);
    y = (float)(sin(2*PIE*(float)(m)/(float)(PolyNum))*xx + 
      cos(2*PIE*(float)(m)/(float)(PolyNum))*yy);
    z = zz;
    
    xxx = b[0][0]*x+b[0][1]*y+b[0][2]*z+ex;
    yyy = b[1][0]*x+b[1][1]*y+b[1][2]*z+ey;
    zzz = b[2][0]*x+b[2][1]*y+b[2][2]*z+ez;
    
    vertex[v_num+2*m].x = xxx;
    vertex[v_num+2*m].y = yyy;
    vertex[v_num+2*m].z = zzz;
    vertex[v_num+2*m+1].x = xxx+sx-ex;
    vertex[v_num+2*m+1].y = yyy+sy-ey;
    vertex[v_num+2*m+1].z = zzz+sz-ez;
    
  }


  for (m = 0; m < PolyNum-1; m++) {
    triangle[t_num+2*m].x = v_num+2*m;
    triangle[t_num+2*m].y = v_num+2*m+1;
    triangle[t_num+2*m].z = v_num+2*m+2;
    triangle[t_num+2*m+1].x = v_num+2*m+1;
    triangle[t_num+2*m+1].y = v_num+2*m+2;
    triangle[t_num+2*m+1].z = v_num+2*m+3;
  }

  triangle[t_num+2*PolyNum-2].x = v_num+2*PolyNum-2;
  triangle[t_num+2*PolyNum-2].y = v_num+2*PolyNum-1;
  triangle[t_num+2*PolyNum-2].z = v_num;
  triangle[t_num+2*PolyNum-1].x = v_num+2*PolyNum-1;
  triangle[t_num+2*PolyNum-1].y = v_num;
  triangle[t_num+2*PolyNum-1].z = v_num+1;

  v_num += 2*PolyNum;
  t_num += 2*PolyNum;

}
};

