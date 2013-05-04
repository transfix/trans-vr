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


typedef struct {
  float trans;
  float rotat;
  float angle;
}CVM;


static CPNT *critical_start;
static int XDIM, YDIM, ZDIM;
static float *dataset;
static unsigned short *classify;


VECTOR AlignmentScore2(DB_VECTOR,DB_VECTOR,float*,int,int);
VECTOR CoVarRotate(float, float, float, float, float, float);
VECTOR AlignmentScoreRefine(DB_VECTOR,DB_VECTOR,float*,int,int,float,float);





void CoVarianceRefine(int xd,int yd,int zd,float *data,CPNT *critical, 
		      unsigned short *p_classify,DB_VECTOR *local_fold,
		      int numaxis,CVM* coVar,int numfd5,FILE *fp)
{
  int m,n;
  DB_VECTOR *temp;
  VECTOR sv;
  float score;
  float x,y,z;
  


  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  critical_start = critical;
  classify = p_classify;
  

  
  temp = (DB_VECTOR*)malloc(sizeof(DB_VECTOR)*numaxis);

  for (n = 0; n < numaxis; n++) {
    temp[n].sx = local_fold[n*60].sx;
    temp[n].sy = local_fold[n*60].sy;
    temp[n].sz = local_fold[n*60].sz;
    temp[n].ex = local_fold[n*60].ex;
    temp[n].ey = local_fold[n*60].ey;
    temp[n].ez = local_fold[n*60].ez;
  }
  
  for (n = 0; n < numaxis; n++) {
    for (m = 0; m < numaxis; m++) {
      if (n == m) {
	coVar[n*numaxis+m].trans = 0.0;
	coVar[n*numaxis+m].rotat = 0.0;
	coVar[n*numaxis+m].angle = 0.0;
	
	score = 1.0f;
	x = 0;
	y = 0;
	z = 0;
	fprintf(fp,"%f %f %f %f\n",score,x,y,z);
      }
      else if (n == 0) {
	sv = AlignmentScore2(temp[m],temp[n],&score,m*60+numfd5*12,n*60+numfd5*12);
	coVar[n*numaxis+m].trans = sv.x;
	coVar[n*numaxis+m].rotat = sv.y;
	coVar[n*numaxis+m].angle = sv.z;
	
	x = sv.x;
	y = sv.y;
	z = sv.z;
	fprintf(fp,"%f %f %f %f\n",score,x,y,z);
      }
      else if (n < m) {
	sv = AlignmentScoreRefine(temp[m],temp[n],&score,m*60+numfd5*12,n*60+numfd5*12,
		           coVar[m].trans-coVar[n].trans,coVar[m].rotat-coVar[n].rotat);
	coVar[n*numaxis+m].trans = sv.x;
	coVar[n*numaxis+m].rotat = sv.y;
	coVar[n*numaxis+m].angle = sv.z;
	
	x = sv.x;
	y = sv.y;
	z = sv.z;
	fprintf(fp,"%f %f %f %f\n",score,x,y,z);
      }
      else {
	x = -coVar[m*numaxis+n].trans;
	y = -coVar[m*numaxis+n].rotat;
	z = coVar[m*numaxis+n].angle;
	coVar[n*numaxis+m].trans = x;
	coVar[n*numaxis+m].rotat = y;
	coVar[n*numaxis+m].angle = z;
	fprintf(fp,"%f %f %f %f\n",-1.0f,x,y,z);
      }
      
    }
    fprintf(fp,"\n");
  }
  
  
  fclose(fp);
  free(temp);

}


VECTOR AlignmentScore2(DB_VECTOR temp1, DB_VECTOR temp2, float* p_score,
		      int index1,int index2)
{
  int i,j,k;
  int u,num;
  float x,y,z;
  float rotat,trans,max_rotat=0.f,max_trans=0.f;
  float tmp_rotat,tmp_trans;
  float theta,phi;
  float t_theta,t_phi;
  float fx,fy,fz;
  float gx,gy,gz;
  float px,py,pz;
  float qx,qy,qz;
  float dx,dy,dz;
  float t1,t2,t3;
  float score,max_score;
  VECTOR sv;
  float x00,x01,x10,x11,y0,y1;
  float tmp;
  float avg1,avg2,alpha;
  CPNT *critical_tmp;


  
  theta = (float)atan2(temp1.sy-temp1.ey,temp1.sx-temp1.ex);
  phi = (float)atan2(temp1.sz-temp1.ez, sqrt((temp1.sx-temp1.ex)*
	(temp1.sx-temp1.ex)+(temp1.sy-temp1.ey)*(temp1.sy-temp1.ey)));
  
  fx = temp2.sx-temp2.ex;
  fy = temp2.sy-temp2.ey;
  fz = temp2.sz-temp2.ez;
  gx = temp1.sx-temp1.ex;
  gy = temp1.sy-temp1.ey;
  gz = temp1.sz-temp1.ez;
  px = fy*gz-fz*gy;
  py = fz*gx-fx*gz;
  pz = fx*gy-fy*gx;
  t_theta = (float)atan2(py,px);
  t_phi = (float)atan2(pz, sqrt(px*px+py*py));
  tmp = (float)sqrt(gx*gx+gy*gy+gz*gz);
  dx = gx/tmp;
  dy = gy/tmp;
  dz = gz/tmp;
  
  qx = temp1.ex+temp2.sx-temp2.ex;
  qy = temp1.ey+temp2.sy-temp2.ey;
  qz = temp1.ez+temp2.sz-temp2.ez;
  t1 = (float)sqrt(fx*fx+fy*fy+fz*fz);
  t2 = (float)sqrt(gx*gx+gy*gy+gz*gz);
  t3 = (float)sqrt((qx-temp1.sx)*(qx-temp1.sx)+
	    (qy-temp1.sy)*(qy-temp1.sy)+
	    (qz-temp1.sz)*(qz-temp1.sz));
  alpha = (float)(acos((t1*t1+t2*t2-t3*t3)/(2.0*t1*t2)));
  
 
  max_score = -999.0f;
  trans = -10.0;
  while (trans <= 10.0) {
    for (u = 0; u < 360; u+=5) {
      rotat = PIE*(float)u/180.0f;
      
      avg1 = 0;
      avg2 = 0;
      num = 0;
      critical_tmp = critical_start;
      while (critical_tmp != NULL) {
	i = critical_tmp->x;
	j = critical_tmp->y;
	k = critical_tmp->z;
	
	x = i-temp2.ex;
	y = j-temp2.ey;
	z = k-temp2.ez;
	
	if (classify[IndexVect(i,j,k)] != index2) 
	  critical_tmp = critical_tmp->next;
	else {
	  t1 = dataset[IndexVect(i,j,k)];
	  
	  sv = CoVarRotate(x,y,z,t_theta,t_phi,alpha);
	  x = sv.x;
	  y = sv.y;
	  z = sv.z;
	  sv = CoVarRotate(x,y,z,theta,phi,rotat);
	  x = sv.x+temp1.ex+dx*trans;
	  y = sv.y+temp1.ey+dy*trans;
	  z = sv.z+temp1.ez+dz*trans;
	  
	  if (x <= XDIM-2 && x > 0 &&
	      y <= YDIM-2 && y > 0 &&
	      z <= ZDIM-2 && z > 0) {
	    x00 = dataset[IndexVect((int)x,(int)y,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z)]*(x-(int)x);
	    x01 = dataset[IndexVect((int)x,(int)y,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z+1)]*(x-(int)x);
	    x10 = dataset[IndexVect((int)x,(int)y+1,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z)]*(x-(int)x);
	    x11 = dataset[IndexVect((int)x,(int)y+1,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z+1)]*(x-(int)x);
	    y0  = x00*((int)y+1-y) + x10*(y-(int)y);
	    y1  = x01*((int)y+1-y) + x11*(y-(int)y);
	    t2 = y0*((int)z+1-z) + y1*(z-(int)z);
	    
	    avg1 += (float)fabs(t1-t2);
	    avg2 += (float)max2(t1,t2);
	  }
	  num++;
	  critical_tmp = critical_tmp->next;
	}
      }
      
      critical_tmp = critical_start;
      while (critical_tmp != NULL) {
	i = critical_tmp->x;
	j = critical_tmp->y;
	k = critical_tmp->z;
	
	x = i-temp1.ex-dx*trans;
	y = j-temp1.ey-dy*trans;
	z = k-temp1.ez-dz*trans;
	
	
	if (classify[IndexVect(i,j,k)] != index1) 
	  critical_tmp = critical_tmp->next;
	else {
	  t1 = dataset[IndexVect(i,j,k)];
	  
	  sv = CoVarRotate(x,y,z,theta,phi,-rotat);
	  x = sv.x;
	  y = sv.y;
	  z = sv.z;
	  sv = CoVarRotate(x,y,z,t_theta,t_phi,-alpha);
	  x = sv.x+temp2.ex;
	  y = sv.y+temp2.ey;
	  z = sv.z+temp2.ez;
	  
	  if (x <= XDIM-2 && x > 0 &&
	      y <= YDIM-2 && y > 0 &&
	      z <= ZDIM-2 && z > 0) {
	    x00 = dataset[IndexVect((int)x,(int)y,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z)]*(x-(int)x);
	    x01 = dataset[IndexVect((int)x,(int)y,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z+1)]*(x-(int)x);
	    x10 = dataset[IndexVect((int)x,(int)y+1,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z)]*(x-(int)x);
	    x11 = dataset[IndexVect((int)x,(int)y+1,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z+1)]*(x-(int)x);
	    y0  = x00*((int)y+1-y) + x10*(y-(int)y);
	    y1  = x01*((int)y+1-y) + x11*(y-(int)y);
	    t2 = y0*((int)z+1-z) + y1*(z-(int)z);
	    
	    avg1 += (float)fabs(t1-t2);
	    avg2 += (float)max2(t1,t2);
	  }
	  num++;
	  critical_tmp = critical_tmp->next;
	}
      }
      
      score = 1-avg1/avg2;

      if (score > max_score) {
	max_score = score;
	max_rotat = rotat;
	max_trans = trans;
      }
      
    }
    trans += 1;
  }
  
  tmp_trans = max_trans;
  tmp_rotat = max_rotat;
  tmp = PIE/180.0f;
  for (trans = tmp_trans-0.9f; trans <= tmp_trans+0.9f; trans += 0.1f) {
    for (rotat = tmp_rotat-4*tmp; rotat < tmp_rotat+4*tmp; rotat+=tmp) {
      
      avg1 = 0;
      avg2 = 0;
      num = 0;
      critical_tmp = critical_start;
      while (critical_tmp != NULL) {
	i = critical_tmp->x;
	j = critical_tmp->y;
	k = critical_tmp->z;
	
	x = i-temp2.ex;
	y = j-temp2.ey;
	z = k-temp2.ez;
	
	if (classify[IndexVect(i,j,k)] != index2) 
	  critical_tmp = critical_tmp->next;
	else {
	  t1 = dataset[IndexVect(i,j,k)];
	  
	  sv = CoVarRotate(x,y,z,t_theta,t_phi,alpha);
	  x = sv.x;
	  y = sv.y;
	  z = sv.z;
	  sv = CoVarRotate(x,y,z,theta,phi,rotat);
	  x = sv.x+temp1.ex+dx*trans;
	  y = sv.y+temp1.ey+dy*trans;
	  z = sv.z+temp1.ez+dz*trans;
	  
	  if (x <= XDIM-2 && x > 0 &&
	      y <= YDIM-2 && y > 0 &&
	      z <= ZDIM-2 && z > 0) {
	    x00 = dataset[IndexVect((int)x,(int)y,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z)]*(x-(int)x);
	    x01 = dataset[IndexVect((int)x,(int)y,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z+1)]*(x-(int)x);
	    x10 = dataset[IndexVect((int)x,(int)y+1,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z)]*(x-(int)x);
	    x11 = dataset[IndexVect((int)x,(int)y+1,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z+1)]*(x-(int)x);
	    y0  = x00*((int)y+1-y) + x10*(y-(int)y);
	    y1  = x01*((int)y+1-y) + x11*(y-(int)y);
	    t2 = y0*((int)z+1-z) + y1*(z-(int)z);
	    
	    avg1 += (float)fabs(t1-t2);
	    avg2 += (float)max2(t1,t2);
	  }
	  num++;
	  critical_tmp = critical_tmp->next;
	}
      }
      
      critical_tmp = critical_start;
      while (critical_tmp != NULL) {
	i = critical_tmp->x;
	j = critical_tmp->y;
	k = critical_tmp->z;
	
	x = i-temp1.ex-dx*trans;
	y = j-temp1.ey-dy*trans;
	z = k-temp1.ez-dz*trans;
	
	if (classify[IndexVect(i,j,k)] != index1) 
	  critical_tmp = critical_tmp->next;
	else {
	  t1 = dataset[IndexVect(i,j,k)];
	  
	  sv = CoVarRotate(x,y,z,theta,phi,-rotat);
	  x = sv.x;
	  y = sv.y;
	  z = sv.z;
	  sv = CoVarRotate(x,y,z,t_theta,t_phi,-alpha);
	  x = sv.x+temp2.ex;
	  y = sv.y+temp2.ey;
	  z = sv.z+temp2.ez;
	  
	  if (x <= XDIM-2 && x > 0 &&
	      y <= YDIM-2 && y > 0 &&
	      z <= ZDIM-2 && z > 0) {
	    x00 = dataset[IndexVect((int)x,(int)y,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z)]*(x-(int)x);
	    x01 = dataset[IndexVect((int)x,(int)y,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z+1)]*(x-(int)x);
	    x10 = dataset[IndexVect((int)x,(int)y+1,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z)]*(x-(int)x);
	    x11 = dataset[IndexVect((int)x,(int)y+1,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z+1)]*(x-(int)x);
	    y0  = x00*((int)y+1-y) + x10*(y-(int)y);
	    y1  = x01*((int)y+1-y) + x11*(y-(int)y);
	    t2 = y0*((int)z+1-z) + y1*(z-(int)z);
	    
	    avg1 += (float)fabs(t1-t2);
	    avg2 += max2(t1,t2);
	  }
	  num++;
	  critical_tmp = critical_tmp->next;
	}
      }
      
      score = 1-avg1/avg2;

      if (score > max_score) {
	max_score = score;
	max_rotat = rotat;
	max_trans = trans;
      }
      
    }
  }
  
  *p_score = max_score;
  sv.x = max_trans;
  sv.y = max_rotat;
  sv.z = alpha;
  return(sv);
}


VECTOR AlignmentScoreRefine(DB_VECTOR temp1, DB_VECTOR temp2, float* p_score,
		      int index1,int index2, float tmp_trans,float tmp_rotat)
{
  int i,j,k;
  int num;
  float x,y,z;
  float rotat,trans;
  float max_rotat=0.f,max_trans=0.f;
  float theta,phi;
  float t_theta,t_phi;
  float fx,fy,fz;
  float gx,gy,gz;
  float px,py,pz;
  float qx,qy,qz;
  float dx,dy,dz;
  float t1,t2,t3;
  float score,max_score;
  VECTOR sv;
  float x00,x01,x10,x11,y0,y1;
  float tmp;
  float avg1,avg2,alpha;
  CPNT *critical_tmp;


  
  theta = (float)atan2(temp1.sy-temp1.ey,temp1.sx-temp1.ex);
  phi = (float)atan2(temp1.sz-temp1.ez, sqrt((temp1.sx-temp1.ex)*
	(temp1.sx-temp1.ex)+(temp1.sy-temp1.ey)*(temp1.sy-temp1.ey)));
  
  fx = temp2.sx-temp2.ex;
  fy = temp2.sy-temp2.ey;
  fz = temp2.sz-temp2.ez;
  gx = temp1.sx-temp1.ex;
  gy = temp1.sy-temp1.ey;
  gz = temp1.sz-temp1.ez;
  px = fy*gz-fz*gy;
  py = fz*gx-fx*gz;
  pz = fx*gy-fy*gx;
  t_theta = (float)atan2(py,px);
  t_phi = (float)atan2(pz, sqrt(px*px+py*py));
  tmp = (float)sqrt(gx*gx+gy*gy+gz*gz);
  dx = gx/tmp;
  dy = gy/tmp;
  dz = gz/tmp;
  
  qx = temp1.ex+temp2.sx-temp2.ex;
  qy = temp1.ey+temp2.sy-temp2.ey;
  qz = temp1.ez+temp2.sz-temp2.ez;
  t1 = (float)sqrt(fx*fx+fy*fy+fz*fz);
  t2 = (float)sqrt(gx*gx+gy*gy+gz*gz);
  t3 = (float)sqrt((qx-temp1.sx)*(qx-temp1.sx)+
	    (qy-temp1.sy)*(qy-temp1.sy)+
	    (qz-temp1.sz)*(qz-temp1.sz));
  alpha = (float)(acos((t1*t1+t2*t2-t3*t3)/(2.0*t1*t2)));
  
 
  max_score = -999.0f;
  tmp = PIE/180.0f;
  for (trans = tmp_trans-0.9f; trans <= tmp_trans+0.9f; trans += 0.1f) {
    for (rotat = tmp_rotat-4*tmp; rotat < tmp_rotat+4*tmp; rotat+=tmp) {
      
      avg1 = 0;
      avg2 = 0;
      num = 0;
      critical_tmp = critical_start;
      while (critical_tmp != NULL) {
	i = critical_tmp->x;
	j = critical_tmp->y;
	k = critical_tmp->z;
	
	x = i-temp2.ex;
	y = j-temp2.ey;
	z = k-temp2.ez;
	
	if (classify[IndexVect(i,j,k)] != index2) 
	  critical_tmp = critical_tmp->next;
	else {
	  t1 = dataset[IndexVect(i,j,k)];
	  
	  sv = CoVarRotate(x,y,z,t_theta,t_phi,alpha);
	  x = sv.x;
	  y = sv.y;
	  z = sv.z;
	  sv = CoVarRotate(x,y,z,theta,phi,rotat);
	  x = sv.x+temp1.ex+dx*trans;
	  y = sv.y+temp1.ey+dy*trans;
	  z = sv.z+temp1.ez+dz*trans;
	  
	  if (x <= XDIM-2 && x > 0 &&
	      y <= YDIM-2 && y > 0 &&
	      z <= ZDIM-2 && z > 0) {
	    x00 = dataset[IndexVect((int)x,(int)y,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z)]*(x-(int)x);
	    x01 = dataset[IndexVect((int)x,(int)y,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z+1)]*(x-(int)x);
	    x10 = dataset[IndexVect((int)x,(int)y+1,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z)]*(x-(int)x);
	    x11 = dataset[IndexVect((int)x,(int)y+1,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z+1)]*(x-(int)x);
	    y0  = x00*((int)y+1-y) + x10*(y-(int)y);
	    y1  = x01*((int)y+1-y) + x11*(y-(int)y);
	    t2 = y0*((int)z+1-z) + y1*(z-(int)z);
	    
	    avg1 += (float)fabs(t1-t2);
	    avg2 += (float)max2(t1,t2);
	  }
	  num++;
	  critical_tmp = critical_tmp->next;
	}
      }
      
      critical_tmp = critical_start;
      while (critical_tmp != NULL) {
	i = critical_tmp->x;
	j = critical_tmp->y;
	k = critical_tmp->z;
	
	x = i-temp1.ex-dx*trans;
	y = j-temp1.ey-dy*trans;
	z = k-temp1.ez-dz*trans;
	
	if (classify[IndexVect(i,j,k)] != index1) 
	  critical_tmp = critical_tmp->next;
	else {
	  t1 = dataset[IndexVect(i,j,k)];
	  
	  sv = CoVarRotate(x,y,z,theta,phi,-rotat);
	  x = sv.x;
	  y = sv.y;
	  z = sv.z;
	  sv = CoVarRotate(x,y,z,t_theta,t_phi,-alpha);
	  x = sv.x+temp2.ex;
	  y = sv.y+temp2.ey;
	  z = sv.z+temp2.ez;
	  
	  if (x <= XDIM-2 && x > 0 &&
	      y <= YDIM-2 && y > 0 &&
	      z <= ZDIM-2 && z > 0) {
	    x00 = dataset[IndexVect((int)x,(int)y,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z)]*(x-(int)x);
	    x01 = dataset[IndexVect((int)x,(int)y,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y,(int)z+1)]*(x-(int)x);
	    x10 = dataset[IndexVect((int)x,(int)y+1,(int)z)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z)]*(x-(int)x);
	    x11 = dataset[IndexVect((int)x,(int)y+1,(int)z+1)]*((int)x+1-x)+
	      dataset[IndexVect((int)x+1,(int)y+1,(int)z+1)]*(x-(int)x);
	    y0  = x00*((int)y+1-y) + x10*(y-(int)y);
	    y1  = x01*((int)y+1-y) + x11*(y-(int)y);
	    t2 = y0*((int)z+1-z) + y1*(z-(int)z);
	    
	    avg1 += (float)fabs(t1-t2);
	    avg2 += (float)max2(t1,t2);
	  }
	  num++;
	  critical_tmp = critical_tmp->next;
	}
      }
      
      score = 1-avg1/avg2;

      if (score > max_score) {
	max_score = score;
	max_rotat = rotat;
	max_trans = trans;
      }
      
    }
  }
  
  *p_score = max_score;
  sv.x = max_trans;
  sv.y = max_rotat;
  sv.z = alpha;
  return(sv);
}




VECTOR CoVarRotate(float sx, float sy, float sz,
  float theta, float phi, float angle)
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


  x = a[0][0]*sx+a[0][1]*sy+a[0][2]*sz;
  y = a[1][0]*sx+a[1][1]*sy+a[1][2]*sz;
  z = a[2][0]*sx+a[2][1]*sy+a[2][2]*sz;
      
  xx = (float)(cos(angle)*x - sin(angle)*y);
  yy = (float)(sin(angle)*x + cos(angle)*y);
  zz = z;
	
  tmp.x = b[0][0]*xx+b[0][1]*yy+b[0][2]*zz;
  tmp.y = b[1][0]*xx+b[1][1]*yy+b[1][2]*zz;
  tmp.z = b[2][0]*xx+b[2][1]*yy+b[2][2]*zz;
  
  return(tmp);
  
}

};
