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

#define _LITTLE_ENDIAN 1

#define max2(x, y) ((x>y) ? (x):(y))
#define min2(x, y) ((x<y) ? (x):(y))
#define PIE           3.1415926f
#define IndexVect(i,j,k) ((k)*xdim*ydim + (j)*xdim + (i))

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
  float trans;
  float rotat;
  float angle;
}CVM;


VECTOR CoVarRotate(float, float, float, float, float, float);
void swap_buffer(char *buffer, int count, int typesize);


void ComputeSubAverage(int xdim, int ydim, int zdim, float *dataset,
	 unsigned short *result,DB_VECTOR *lcfold,int lcfdnum,int lcaxisnum,
	 CVM* covmatrix,int fvfdnum,float *span_tmp,float *orig_tmp,FILE *fp)
{
  int i,j,k;
  int m,n;
  float *data;
  float average,tmp;
  float x,y,z;
  float xx,yy,zz;
  float x00,x01,x10,x11,y0,y1;
  float alpha,rotat,trans;
  float theta,phi;
  float t_theta,t_phi;
  float fx,fy,fz;
  float gx,gy,gz;
  float px,py,pz;
  float dx,dy,dz;
  VECTOR sv;
  float a[3][3],b[3][3];

  float p_minext[3], p_maxext[3];
  int p_nverts, p_ncells;
  unsigned int p_dim[3];
  float p_orig[3], p_span[3];
  
  int lx,ly,lz;
  int hx,hy,hz;
  int xd,yd,zd;

	size_t fwrite_return = 0;

  data = (float *)malloc(sizeof(float)*xdim*ydim*zdim);

  lx = 9999;
  ly = 9999;
  lz = 9999;
  hx = 0;
  hy = 0;
  hz = 0;
  for (k=0; k<zdim; k++)
    for (j=0; j<ydim; j++) 
      for (i=0; i<xdim; i++) {
	if (result[IndexVect(i,j,k)] == fvfdnum*12) {
	  average = 0;
	  for (n = 0; n < lcaxisnum; n++) {	    
	    alpha = covmatrix[n].angle;
	    rotat = covmatrix[n].rotat;
	    trans = covmatrix[n].trans;
  	    gx = lcfold[n*60].sx-lcfold[n*60].ex;
	    gy = lcfold[n*60].sy-lcfold[n*60].ey;
	    gz = lcfold[n*60].sz-lcfold[n*60].ez;
	    fx = lcfold[0].sx-lcfold[0].ex;
	    fy = lcfold[0].sy-lcfold[0].ey;
	    fz = lcfold[0].sz-lcfold[0].ez;
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
	    x = orig_tmp[0] + span_tmp[0]*i-lcfold[0].ex;
	    y = orig_tmp[1] + span_tmp[1]*j-lcfold[0].ey;
	    z = orig_tmp[2] + span_tmp[2]*k-lcfold[0].ez;
//	    x = (float)i-lcfold[0].ex;
//	    y = (float)j-lcfold[0].ey;
//	    z = (float)k-lcfold[0].ez;
	    sv = CoVarRotate(x,y,z,t_theta,t_phi,alpha);
	    x = sv.x;
	    y = sv.y;
	    z = sv.z;
	    sv = CoVarRotate(x,y,z,theta,phi,rotat);
	    x = sv.x+dx*trans;
	    y = sv.y+dy*trans;
	    z = sv.z+dz*trans;
	    
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
	    	    
	    xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
	    yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
	    zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
	    
	    for (m = 0; m < lcfdnum; m++) {
	      x = (float)(cos(2*PIE*(float)(m)/(float)(lcfdnum))*xx - 
			  sin(2*PIE*(float)(m)/(float)(lcfdnum))*yy);
	      y = (float)(sin(2*PIE*(float)(m)/(float)(lcfdnum))*xx + 
			  cos(2*PIE*(float)(m)/(float)(lcfdnum))*yy);
	      z = zz;
	      dx = (b[0][0]*x+b[0][1]*y+b[0][2]*z+lcfold[n*60].ex-orig_tmp[0])/span_tmp[0];
	      dy = (b[1][0]*x+b[1][1]*y+b[1][2]*z+lcfold[n*60].ey-orig_tmp[1])/span_tmp[1];
	      dz = (b[2][0]*x+b[2][1]*y+b[2][2]*z+lcfold[n*60].ez-orig_tmp[2])/span_tmp[2];
//	      dx = b[0][0]*x+b[0][1]*y+b[0][2]*z+lcfold[n*60].ex;
//	      dy = b[1][0]*x+b[1][1]*y+b[1][2]*z+lcfold[n*60].ey;
//	      dz = b[2][0]*x+b[2][1]*y+b[2][2]*z+lcfold[n*60].ez;
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
	      average += y0*((int)dz+1-dz) + y1*(dz-(int)dz);
	    }
	  }
	  average /= (float)(lcaxisnum*lcfdnum);
	  data[IndexVect(i,j,k)] = average;

	  if (i < lx)
	    lx = i;
	  if (i > hx)
	    hx = i;
	  if (j < ly)
	    ly = j;
	  if (j > hy)
	    hy = j;
	  if (k < lz)
	    lz = k;
	  if (k > hz)
	    hz = k;
	}
	else
	  data[IndexVect(i,j,k)] = 0;
     }



  lx = max2(0,lx-10);
  ly = max2(0,ly-10);
  lz = max2(0,lz-10);
  hx = min2(xdim-1,hx+10);
  hy = min2(ydim-1,hy+10);
  hz = min2(zdim-1,hz+10);
  xd = hx-lx+1;
  yd = hy-ly+1;
  zd = hz-lz+1;
  p_minext[0]=orig_tmp[0]+lx*span_tmp[0];
  p_minext[1]=orig_tmp[1]+ly*span_tmp[1];
  p_minext[2]=orig_tmp[2]+lz*span_tmp[2];
  p_maxext[0]=orig_tmp[0]+hx*span_tmp[0];
  p_maxext[1]=orig_tmp[1]+hy*span_tmp[1];
  p_maxext[2]=orig_tmp[2]+hz*span_tmp[2];
  p_nverts = xd*yd*zd;
  p_ncells = (xd-1)*(yd-1)*(zd-1);
  p_dim[0]=xd;
  p_dim[1]=yd;
  p_dim[2]=zd;
  p_orig[0]=p_minext[0];
  p_orig[1]=p_minext[1];
  p_orig[2]=p_minext[2];
  p_span[0]=span_tmp[0];
  p_span[1]=span_tmp[1];
  p_span[2]=span_tmp[2];

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)p_minext, 3, sizeof(float));
  swap_buffer((char *)p_maxext, 3, sizeof(float));
  swap_buffer((char *)&p_nverts, 1, sizeof(int));
  swap_buffer((char *)&p_ncells, 1, sizeof(int));
  swap_buffer((char *)p_dim, 3, sizeof(unsigned int));
  swap_buffer((char *)p_orig, 3, sizeof(float));
  swap_buffer((char *)p_span, 3, sizeof(float));
#endif 
  fwrite_return = fwrite(p_minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(p_maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&p_nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&p_ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(p_dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(p_orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(p_span, sizeof(float), 3, fp);

  for (k=lz; k<=hz; k++) 
    for (j=ly; j<=hy; j++)
      for (i=lx; i<=hx; i++) {
	x = data[IndexVect(i,j,k)];
#ifdef _LITTLE_ENDIAN
	swap_buffer((char *)&x, 1, sizeof(float));
#endif
	fwrite_return = fwrite(&x, sizeof(float), 1, fp);
      }          
  fclose(fp);
  
  free(data);
}



void Write5fSubunit(int xdim, int ydim, int zdim, float *data,
	 unsigned short *result, float *span_tmp,float *orig_tmp,FILE *fp)
{
  int i,j,k;
  float tmp;

  float p_minext[3], p_maxext[3];
  int p_nverts, p_ncells;
  unsigned int p_dim[3];
  float p_orig[3], p_span[3];
  
  int lx,ly,lz;
  int hx,hy,hz;
  int xd,yd,zd;
  
  size_t fwrite_return = 0;

  
  lx = 9999;
  ly = 9999;
  lz = 9999;
  hx = 0;
  hy = 0;
  hz = 0;
  for (k=0; k<zdim; k++)
    for (j=0; j<ydim; j++) 
      for (i=0; i<xdim; i++) {
	if (result[IndexVect(i,j,k)] == 0) {
	  if (i < lx)
	    lx = i;
	  if (i > hx)
	    hx = i;
	  if (j < ly)
	    ly = j;
	  if (j > hy)
	    hy = j;
	  if (k < lz)
	    lz = k;
	  if (k > hz)
	    hz = k; 
	}
	else
	  data[IndexVect(i,j,k)] = 0;
      }

  lx = max2(0,lx-10);
  ly = max2(0,ly-10);
  lz = max2(0,lz-10);
  hx = min2(xdim-1,hx+10);
  hy = min2(ydim-1,hy+10);
  hz = min2(zdim-1,hz+10);

 
  xd = hx-lx+1;
  yd = hy-ly+1;
  zd = hz-lz+1;
  p_minext[0]=orig_tmp[0]+lx*span_tmp[0];
  p_minext[1]=orig_tmp[1]+ly*span_tmp[1];
  p_minext[2]=orig_tmp[2]+lz*span_tmp[2];
  p_maxext[0]=orig_tmp[0]+hx*span_tmp[0];
  p_maxext[1]=orig_tmp[1]+hy*span_tmp[1];
  p_maxext[2]=orig_tmp[2]+hz*span_tmp[2];
  p_nverts = xd*yd*zd;
  p_ncells = (xd-1)*(yd-1)*(zd-1);
  p_dim[0]=xd;
  p_dim[1]=yd;
  p_dim[2]=zd;
  p_orig[0]=p_minext[0];
  p_orig[1]=p_minext[1];
  p_orig[2]=p_minext[2];
  p_span[0]=span_tmp[0];
  p_span[1]=span_tmp[1];
  p_span[2]=span_tmp[2];

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)p_minext, 3, sizeof(float));
  swap_buffer((char *)p_maxext, 3, sizeof(float));
  swap_buffer((char *)&p_nverts, 1, sizeof(int));
  swap_buffer((char *)&p_ncells, 1, sizeof(int));
  swap_buffer((char *)p_dim, 3, sizeof(unsigned int));
  swap_buffer((char *)p_orig, 3, sizeof(float));
  swap_buffer((char *)p_span, 3, sizeof(float));
#endif 
  fwrite_return = fwrite(p_minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(p_maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&p_nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&p_ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(p_dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(p_orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(p_span, sizeof(float), 3, fp);

  for (k=lz; k<=hz; k++) 
    for (j=ly; j<=hy; j++)
      for (i=lx; i<=hx; i++) {
	tmp = data[IndexVect(i,j,k)];
#ifdef _LITTLE_ENDIAN
	swap_buffer((char *)&tmp, 1, sizeof(float));
#endif
	fwrite_return = fwrite(&tmp, sizeof(float), 1, fp);
      }          
  fclose(fp);
  
}

};
