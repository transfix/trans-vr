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
// Add dimension doesn't make sense. Needs to be fixed.

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
#define MINDIST       10
#define PolyNum       15

#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))

namespace SegCapsid {

typedef struct {
  float x;
  float y;
  float z;
}VECTOR;


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
static int Rd = 200;
static int sample = 2;
static float density[10];
static float sym_rd;

float GlobalSymmetryScore(float, float, float);
VECTOR Rotate(float, float, float, float, float, float, int, int, int);
void GlobalRefinement(float *, float *, float *);


void GlobalSymmetry(int xd, int yd, int zd, float *data, CPNT *critical,
		    VECTOR *FiveFold, FILE* fp)
{
  int x,y,z;
  int i=0,j=0,k=0;
  int ii=0,jj=0,kk=0;
  int iii=0,jjj=0,kkk=0;
  unsigned char flag;
  float distance, dist;
  float xx,yy,zz;
  float cx,cy,cz;
  float nx,ny,nz;
  float ax,ay,az;
  unsigned char *tmp;
  float temp,minsym,minsym2;
  int total_num,num_save;
  float *temp_asymm;
  VECTOR sv;
  float theta,phi;
  int n;
  float a,b,c,d;
  float fx,fy,fz;
  float gx,gy,gz;
  float px,py,pz;
  double aa,bb, aa_angle;
  
  
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  critical_start = critical;
  sym_rd = min2(XDIM,min2(YDIM,ZDIM))/2.f-2.f;
  

  
  temp_asymm = (float*)malloc(sizeof(float)*(2*Rd+1)*(2*Rd+1)*(2*Rd+1));
  tmp= (unsigned char*)malloc(sizeof(unsigned char)*(2*Rd+1)*(2*Rd+1)*(2*Rd+1));
  total_num = 0;
  for (z=0; z<2*Rd+1; z++)
    for (y=0; y<2*Rd+1; y++) 
      for (x=0; x<2*Rd+1; x++) {
	tmp[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] = 0;
	distance = (float)sqrt(double((x-Rd)*(x-Rd)+(y-Rd)*(y-Rd)+(z-Rd)*(z-Rd)));
	if (distance > Rd-1 && distance < Rd+1) {
	  flag = 0;
	  for (k=max2(z-sample,0); k<=min2(z+sample,2*Rd); k++)
	    for (j=max2(y-sample,0); j<=min2(y+sample,2*Rd); j++) 
	      for (i=max2(x-sample,0); i<=min2(x+sample,2*Rd); i++) {
		dist = (float)sqrt(double((x-i)*(x-i)+(y-j)*(y-j)+(z-k)*(z-k)));
		if (dist <= sample && 
		    tmp[k*(2*Rd+1)*(2*Rd+1)+j*(2*Rd+1)+i] == 255) 
		  flag = 1;
	      }
	  if (flag == 0) {
	    tmp[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] = 255;
	    total_num++;
	  }
	}
      }
  printf("total sampling number: %d \n",total_num);

 
  num_save = 0;
  aa = 2*sin(0.5*ANGL1);
  aa_angle = asin(aa*sqrt(3.0)/3.0);
  minsym = 9999999;
  for (z=0; z<2*Rd+1; z++)
    for (y=0; y<2*Rd+1; y++) 
      for (x=0; x<2*Rd+1; x++) {
	temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] = 9999999;
	
	if (tmp[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] == 255) {

	  aa = sqrt(double((x-Rd)*(x-Rd)+(y-Rd)*(y-Rd)+(z-Rd)*(z-Rd)));
	  bb = acos((z-Rd)/aa);
	  if (bb <= aa_angle+0.05) {
	    num_save++;
	    temp = GlobalSymmetryScore((float)x,(float)y,(float)z);
	    if (temp < minsym) {
	      minsym = temp;
	      i = x;
	      j = y;
	      k = z;
	    }
	    temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] = temp; 
	   
	  }
	}
      }
  
  xx = (float)((i-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
    +XDIM/2);
  yy = (float)((j-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
    +YDIM/2);
  zz = (float)((k-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
    +ZDIM/2);
  
  temp_asymm[k*(2*Rd+1)*(2*Rd+1)+j*(2*Rd+1)+i] = 8888888;
  FiveFold[0].x = xx;
  FiveFold[0].y = yy;
  FiveFold[0].z = zz;
  
  printf("FiveFold[0]: %f %f %f \n",FiveFold[0].x,FiveFold[0].y,FiveFold[0].z);
 

  minsym = 9999999;
  minsym2 = 9999999;
  for (z=0; z<2*Rd+1; z++)
    for (y=0; y<2*Rd+1; y++) 
      for (x=0; x<2*Rd+1; x++) {
	if (tmp[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] == 255) {
	  if (temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] < 8888888) {
	    a = (float)sqrt(double((i-x)*(i-x)+(j-y)*(j-y)+(k-z)*(k-z)));
	    if (a >= MINDIST &&
		temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] < minsym2) {
	      minsym2 = temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x];
	      ii = x;
	      jj = y;
	      kk = z;
	    }
	  }
	
	  else if (temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] == 9999999) {
	    
	    if (i != Rd || j != Rd) {
	      a = (float)((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd));
	      b = ((x-Rd)*(i-Rd)+(y-Rd)*(j-Rd)+(z-Rd)*(k-Rd))/a;
	      fx = Rd+b*(i-Rd);
	      fy = Rd+b*(j-Rd);
	      fz = Rd+b*(k-Rd);
	      gx = (float)sqrt(double((fx-x)*(fx-x)+(fy-y)*(fy-y)+(fz-z)*(fz-z)));
	      
	      b = (Rd*(k-Rd))/a;
	      px = Rd+b*(i-Rd);
	      py = Rd+b*(j-Rd);
	      pz = Rd+b*(k-Rd);
	      gz = (Rd-px)*(x-px)+(Rd-py)*(y-py)+(2*Rd-pz)*(z-pz);
	      
	      a = (float)((j-Rd)*Rd);
	      b = (float)(-(i-Rd)*Rd);
	      c = 0;
	      d = -a*Rd-b*Rd-2*c*Rd;
	      gy = (float)(fabs(a*x+b*y+c*z+d)/sqrt(a*a+b*b+c*c));
	      bb = asin(gy/gx);

	      a = (float)sqrt(double((x-Rd)*(x-Rd)+(y-Rd)*(y-Rd)+(z-Rd)*(z-Rd)));
	      b = (float)sqrt(double((Rd-i)*(Rd-i)+(Rd-j)*(Rd-j)+(Rd-k)*(Rd-k)));
	      c = (float)sqrt(double((x-i)*(x-i)+(y-j)*(y-j)+(z-k)*(z-k)));
	      aa = acos((a*a+b*b-c*c)/(2*a*b));

	      
	      if (gz >= 0 && bb <= 0.2*PIE+0.05 && aa <= ANGL1+0.05 ) {
		num_save++;
		temp = GlobalSymmetryScore((float)x,(float)y,(float)z);
		if (temp < minsym) {
		  minsym = temp;
		  iii = x;
		  jjj = y;
		  kkk = z;
		}
		temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] = temp; 
		
	      }
	    }
	    else {
	      a = (float)((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd));
	      b = ((x-Rd)*(i-Rd)+(y-Rd)*(j-Rd)+(z-Rd)*(k-Rd))/a;
	      fx = Rd+b*(i-Rd);
	      fy = Rd+b*(j-Rd);
	      fz = Rd+b*(k-Rd);
	      gx = (float)sqrt(double((fx-x)*(fx-x)+(fy-y)*(fy-y)+(fz-z)*(fz-z)));
	      
	      b = (Rd*(j-Rd)+Rd*(k-Rd))/a;
	      px = Rd+b*(i-Rd);
	      py = Rd+b*(j-Rd);
	      pz = Rd+b*(k-Rd);
	      gz = (Rd-px)*(x-px)+(2*Rd-py)*(y-py)+(2*Rd-pz)*(z-pz);
	      
	      a = (float)(-Rd*(k-2*Rd) + (j-2*Rd)*Rd);
	      b = (float)(-(i-Rd)*Rd);
	      c = (float)((i-Rd)*Rd);
	      d = -a*Rd - 2*b*Rd-2*c*Rd;
	      gy = (float)(fabs(a*x+b*y+c*z+d)/sqrt(a*a+b*b+c*c));
	      
	      a = (float)sqrt(double((x-Rd)*(x-Rd)+(y-Rd)*(y-Rd)+(z-Rd)*(z-Rd)));
	      b = (float)sqrt(double((Rd-fx)*(Rd-fx)+(Rd-fy)*(Rd-fy)+(Rd-fz)*(Rd-fz)));
	      aa = acos(b/a);
	      bb = asin(gy/gx);
	      if (gz >= 0 && bb <= 0.2*PIE+0.05 && aa <= ANGL1+0.05 ) {
		num_save++;
		temp = GlobalSymmetryScore((float)x,(float)y,(float)z);
		if (temp < minsym) {
		  minsym = temp;
		  iii = x;
		  jjj = y;
		  kkk = z;
		}
		temp_asymm[z*(2*Rd+1)*(2*Rd+1)+y*(2*Rd+1)+x] = temp; 
		
	      }
	    }
	    
	  }
	}
      }
  

  printf("total:%d  used:%d \n",total_num,num_save);

  if (minsym <= minsym2) {
    i = iii;
    j = jjj;
    k = kkk;
    xx = (float)((i-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
      +XDIM/2);
    yy = (float)((j-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
      +YDIM/2);
    zz = (float)((k-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
      +ZDIM/2);
    
    temp_asymm[k*(2*Rd+1)*(2*Rd+1)+j*(2*Rd+1)+i] = 8888888;
    FiveFold[1].x = xx;
    FiveFold[1].y = yy;
    FiveFold[1].z = zz;
  }
  else {
    i = ii;
    j = jj;
    k = kk;
    xx = (float)((i-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
      +XDIM/2);
    yy = (float)((j-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
      +YDIM/2);
    zz = (float)((k-Rd)*sym_rd/sqrt(double((i-Rd)*(i-Rd)+(j-Rd)*(j-Rd)+(k-Rd)*(k-Rd)))
      +ZDIM/2);
    
    temp_asymm[k*(2*Rd+1)*(2*Rd+1)+j*(2*Rd+1)+i] = 8888888;
    FiveFold[1].x = xx;
    FiveFold[1].y = yy;
    FiveFold[1].z = zz;
  }
  

  fx = FiveFold[0].x;
  fy = FiveFold[0].y;
  fz = FiveFold[0].z;
  GlobalRefinement(&fx, &fy, &fz);
  FiveFold[0].x = fx;
  FiveFold[0].y = fy;
  FiveFold[0].z = fz;
  fx = FiveFold[1].x;
  fy = FiveFold[1].y;
  fz = FiveFold[1].z;
  GlobalRefinement(&fx, &fy, &fz);
  FiveFold[1].x = fx;
  FiveFold[1].y = fy;
  FiveFold[1].z = fz;
  
  cx = FiveFold[0].x-XDIM/2;
  cy = FiveFold[0].y-YDIM/2;
  cz = FiveFold[0].z-ZDIM/2;
  nx = FiveFold[1].x-XDIM/2;
  ny = FiveFold[1].y-YDIM/2;
  nz = FiveFold[1].z-ZDIM/2;
  ax = nz*cy-ny*cz;
  ay = nx*cz-nz*cx;
  az = ny*cx-nx*cy;
  theta = (float)atan2(ay,ax);
  phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
  sv = Rotate(FiveFold[0].x,FiveFold[0].y,FiveFold[0].z,theta,phi,ANGL1,XDIM,YDIM,ZDIM);
  FiveFold[1].x = sv.x;
  FiveFold[1].y = sv.y;
  FiveFold[1].z = sv.z;

  nx = FiveFold[1].x-XDIM/2;
  ny = FiveFold[1].y-YDIM/2;
  nz = FiveFold[1].z-ZDIM/2;
  ax = nz*cy-ny*cz;
  ay = nx*cz-nz*cx;
  az = ny*cx-nx*cy;
  theta = (float)atan2(ay,ax);
  phi = (float)atan2(az,sqrt(ax*ax+ay*ay));
  sv = Rotate(FiveFold[0].x,FiveFold[0].y,FiveFold[0].z,theta,phi,PIE,XDIM,YDIM,ZDIM);
  FiveFold[11].x = sv.x;
  FiveFold[11].y = sv.y;
  FiveFold[11].z = sv.z;

  theta = (float)atan2(cy,cx);
  phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
  for (n = 1; n < 5; n++) { 
    sv = Rotate(FiveFold[1].x,FiveFold[1].y,FiveFold[1].z,theta,phi,-n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
    FiveFold[n+1].x = sv.x;
    FiveFold[n+1].y = sv.y;
    FiveFold[n+1].z = sv.z;
  }
  
  cx = FiveFold[1].x-XDIM/2;
  cy = FiveFold[1].y-YDIM/2;
  cz = FiveFold[1].z-ZDIM/2;
  theta = (float)atan2(cy,cx);
  phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
  sv = Rotate(FiveFold[2].x,FiveFold[2].y,FiveFold[2].z,theta,phi,2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
  FiveFold[6].x = sv.x;
  FiveFold[6].y = sv.y;
  FiveFold[6].z = sv.z;

  cx = FiveFold[0].x-XDIM/2;
  cy = FiveFold[0].y-YDIM/2;
  cz = FiveFold[0].z-ZDIM/2;
  theta = (float)atan2(cy,cx);
  phi = (float)atan2(cz, sqrt(cx*cx+cy*cy));
  for (n = 1; n < 5; n++) { 
    sv = Rotate(FiveFold[6].x,FiveFold[6].y,FiveFold[6].z,theta,phi,-n*2.0f*PIE/5.0f,XDIM,YDIM,ZDIM);
    FiveFold[n+6].x = sv.x;
    FiveFold[n+6].y = sv.y;
    FiveFold[n+6].z = sv.z;
  }

   
  for(i=0; i<12; i++) 
    fprintf(fp, "%f %f %f \n",FiveFold[i].x,FiveFold[i].y,FiveFold[i].z);
  fclose(fp);


  free(tmp);
  free(temp_asymm);
  
}



float GlobalSymmetryScore(float sx, float sy, float sz)
{
  int i,j,k;
  float x3,y3,z3;
  float d3,d2;
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
  float distance;
  int count = 0;


  theta = (float)atan2(sy-Rd,sx-Rd);
  phi = (float)atan2(double(sz-Rd), sqrt(double((sx-Rd)*(sx-Rd)+(sy-Rd)*(sy-Rd))));
  distance = (float)(min2(XDIM,min2(YDIM,ZDIM))/4);

  
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


  x3 = sx-Rd;
  y3 = sy-Rd;
  z3 = sz-Rd;
  d3 = (float)sqrt(x3*x3+y3*y3+z3*z3);

  num = 0;
  asymmetry = 0;
  critical_tmp = critical_start;
  while (critical_tmp != NULL) {
    count = 0;  // record the number of rotated critical points that were in-bounds
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;

    x = (float)(i-XDIM/2);
    y = (float)(j-YDIM/2);
    z = (float)(k-ZDIM/2);
    xx = y3*z-z3*y;
    yy = x*z3-x3*z;
    zz = x3*y-x*y3;
    d2 = (float)(sqrt(xx*xx+yy*yy+zz*zz)/d3);
    
    if (d2 > distance) 
      critical_tmp = critical_tmp->next;
    else {
      density[0] = dataset[IndexVect(i,j,k)];
      
      xx = a[0][0]*x+a[0][1]*y+a[0][2]*z;
      yy = a[1][0]*x+a[1][1]*y+a[1][2]*z;
      zz = a[2][0]*x+a[2][1]*y+a[2][2]*z;
      
      average = density[0];
      for (m = 1; m < 5; m++) {
	x = (float)(cos(2*PIE*(float)(m)/5.0)*xx - 
		    sin(2*PIE*(float)(m)/5.0)*yy);
	y = (float)(sin(2*PIE*(float)(m)/5.0)*xx + 
		    cos(2*PIE*(float)(m)/5.0)*yy);
	z = zz;
	
	dx = b[0][0]*x+b[0][1]*y+b[0][2]*z+XDIM/2;
	dy = b[1][0]*x+b[1][1]*y+b[1][2]*z+YDIM/2;
	dz = b[2][0]*x+b[2][1]*y+b[2][2]*z+ZDIM/2;
	
	if(dx > 0.0 && dy > 0.0 && dz > 0.0 &&
	   dx < XDIM-1 && dy < YDIM-1 && dz < ZDIM-1)
	  {
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
	    
	    average += density[count++];
	  }
      }
      
      average /= count;
      
      variance = 0;
      for (m = 0; m < count; m++) 
	variance += (float)fabs(density[m]-average);
      
      asymmetry += variance/float(count);
      num ++;
      
      critical_tmp = critical_tmp->next;
    }
  }

  
  if (num > 0)
    return(asymmetry/(float)(num));
  else {
    printf("wrong ???? \n");
    return(9999999);
  }
}



void GlobalRefinement(float *fx, float *fy, float *fz)
{
  int i,j;
  float sx,sy,sz;
  float ex,ey,ez;
  VECTOR dv1,dv2;
  float x,y,z;
  float min_rotation,radius,temp;
  int rddd;
  float rf_sample;

  rddd = 10;
  rf_sample = 0.1f*sample*sym_rd/(float)Rd; 
  sx = *fx;
  sy = *fy;
  sz = *fz;
  ex = XDIM/2.f;
  ey = YDIM/2.f;
  ez = ZDIM/2.f;

  temp = (float)sqrt(double((sy-ey)*(sy-ey)+(sx-ex)*(sx-ex)));
  if (temp == 0) {
    dv1.x = 1;
    dv1.y = 0;
    dv1.z = 0;
  }
  else {
    dv1.x = (ey-sy)/temp;
    dv1.y = (sx-ex)/temp;
    dv1.z = 0;
  }
  x = (sy-ey)*dv1.z-(sz-ez)*dv1.y;
  y = (sz-ez)*dv1.x-(sx-ex)*dv1.z;
  z = (sx-ex)*dv1.y-(sy-ey)*dv1.x;
  temp = (float)sqrt(x*x+y*y+z*z);
  dv2.x = x/temp;
  dv2.y = y/temp;
  dv2.z = z/temp;
  
  min_rotation = 9999999.0f;
  for (j = -rddd; j <= rddd; j++) 
    for (i = -rddd; i <= rddd; i++) {
      x = sx + i*rf_sample*dv1.x + j*rf_sample*dv2.x;
      y = sy + i*rf_sample*dv1.y + j*rf_sample*dv2.y;
      z = sz + i*rf_sample*dv1.z + j*rf_sample*dv2.z;

      radius = (float)sqrt((x-XDIM/2)*(x-XDIM/2)+
		    (y-YDIM/2)*(y-YDIM/2)+
		    (z-ZDIM/2)*(z-ZDIM/2));
      x = (x-XDIM/2)*Rd/radius+Rd;
      y = (y-YDIM/2)*Rd/radius+Rd;
      z = (z-ZDIM/2)*Rd/radius+Rd;
      
      temp = GlobalSymmetryScore(x,y,z);
      if (temp < min_rotation) {
	min_rotation = temp;
	ex = x;
	ey = y;
	ez = z;
      }
    }

  *fx = (ex-Rd)*sym_rd/Rd+XDIM/2;
  *fy = (ey-Rd)*sym_rd/Rd+YDIM/2;
  *fz = (ez-Rd)*sym_rd/Rd+ZDIM/2;

}

};

