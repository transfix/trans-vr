/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/**************************************
 * Zeyun Yu (zeyun@cs.utexas.edu)    *
 * Department of Computer Science    *
 * University of Texas at Austin     *
 **************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <TightCocone/segment.h>

namespace TightCocone
{
 
void gradient();
void gvfflow();
void GetGradient();

void GVF_Compute()
{

  ImgGrad = (float *)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
  GetGradient();
  /*
  gradient();
  */
  gvfflow();
  
}


void gvfflow()
{
  float *u, *v, *w;
  float *tempx, *tempy, *tempz;
  int m,i,j,k;
  float gf,hf;
  float cx,cy,cz;
  float dt = 0.1666;
  float Kama = 1.0;
  float maxgrad, gradient;

  if (
      (u = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM)) == NULL) {
    printf("not enough memory 111...\n");
    exit(0);
  }
  if (
      (v = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM)) == NULL) {
    printf("not enough memory 222...\n");
    exit(0);
  }
  if (
      (w = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM)) == NULL) {
    printf("not enough memory 333...\n");
    exit(0);
  }
  
  if (
      (tempx = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM)) == NULL) {
    printf("not enough memory 444...\n");
    exit(0);
  }
  if (
      (tempy = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM)) == NULL) {
    printf("not enough memory 555...\n");
    exit(0);
  }
  if (
      (tempz = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM)) == NULL) {
    printf("not enough memory 666...\n");
    exit(0);
  }
  
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	u[IndexVect(i,j,k)] = velocity[IndexVect(i,j,k)].x;
	v[IndexVect(i,j,k)] = velocity[IndexVect(i,j,k)].y;
	w[IndexVect(i,j,k)] = velocity[IndexVect(i,j,k)].z;
      }
  
 
  for (m = 0; m<5; m++) {

    if (m % 2 == 0)
      printf("Iteration = %d \n",m);
    
    /* Normalize the vector field 
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++)
	for (i=0; i<XDIM; i++) { 
	  gradient = sqrt(u[IndexVect(i,j,k)]*u[IndexVect(i,j,k)] + 
			v[IndexVect(i,j,k)]*v[IndexVect(i,j,k)] + 
			w[IndexVect(i,j,k)]*w[IndexVect(i,j,k)] );
	  if (gradient > 0) {
	    u[IndexVect(i,j,k)] = u[IndexVect(i,j,k)]/gradient;
	    v[IndexVect(i,j,k)] = v[IndexVect(i,j,k)]/gradient;
	    w[IndexVect(i,j,k)] = w[IndexVect(i,j,k)]/gradient;
	  }
	}
    */
    /* Diffusing the vector field */
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++) {
	  
	  gf = dt*exp(-sqrt(
		   velocity[IndexVect(i,j,k)].x * velocity[IndexVect(i,j,k)].x
		 + velocity[IndexVect(i,j,k)].y * velocity[IndexVect(i,j,k)].y
		 + velocity[IndexVect(i,j,k)].z * velocity[IndexVect(i,j,k)].z)/Kama);
	  
	  hf = dt-gf;
	  cx = hf * velocity[IndexVect(i,j,k)].x;
	  cy = hf * velocity[IndexVect(i,j,k)].y;
	  cz = hf * velocity[IndexVect(i,j,k)].z;
	  
	  
	  tempx[IndexVect(i,j,k)] = (1.0-hf)*u[IndexVect(i,j,k)] + 
	    gf*(u[IndexVect(min(i+1,XDIM-1),j,k)] + u[IndexVect(max(i-1,0),j,k)]
		+ u[IndexVect(i,min(j+1,YDIM-1),k)] + u[IndexVect(i,max(j-1,0),k)]
		+ u[IndexVect(i,j,min(k+1,ZDIM-1))] + u[IndexVect(i,j,max(k-1,0))]
		- 6.0*u[IndexVect(i,j,k)]) + cx;
	  tempy[IndexVect(i,j,k)] = (1.0-hf)*v[IndexVect(i,j,k)] + 
	    gf*(v[IndexVect(min(i+1,XDIM-1),j,k)] + v[IndexVect(max(i-1,0),j,k)]
		+ v[IndexVect(i,min(j+1,YDIM-1),k)] + v[IndexVect(i,max(j-1,0),k)]
		+ v[IndexVect(i,j,min(k+1,ZDIM-1))] + v[IndexVect(i,j,max(k-1,0))]
		- 6.0*v[IndexVect(i,j,k)]) + cy;
	  tempz[IndexVect(i,j,k)] = (1.0-hf)*w[IndexVect(i,j,k)] + 
	    gf*(w[IndexVect(min(i+1,XDIM-1),j,k)] + w[IndexVect(max(i-1,0),j,k)]
		+ w[IndexVect(i,min(j+1,YDIM-1),k)] + w[IndexVect(i,max(j-1,0),k)]
		+ w[IndexVect(i,j,min(k+1,ZDIM-1))] + w[IndexVect(i,j,max(k-1,0))]
		- 6.0*w[IndexVect(i,j,k)]) + cz;
	 
	}
    
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++) {
	  u[IndexVect(i,j,k)] = tempx[IndexVect(i,j,k)];
	  v[IndexVect(i,j,k)] = tempy[IndexVect(i,j,k)];
	  w[IndexVect(i,j,k)] = tempz[IndexVect(i,j,k)];
	}

  }
 
  maxgrad = 0.0;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	gradient = u[IndexVect(i,j,k)] * u[IndexVect(i,j,k)] +
	  v[IndexVect(i,j,k)] * v[IndexVect(i,j,k)] +
	  w[IndexVect(i,j,k)] * w[IndexVect(i,j,k)];
	if (gradient > maxgrad)
	  maxgrad = gradient;
      }

  maxgrad = sqrt(maxgrad);
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	velocity[IndexVect(i,j,k)].x = u[IndexVect(i,j,k)] / maxgrad;
	velocity[IndexVect(i,j,k)].y = v[IndexVect(i,j,k)] / maxgrad;
	velocity[IndexVect(i,j,k)].z = w[IndexVect(i,j,k)] / maxgrad;
      }

  free(u);
  free(v);
  free(w);
  free(tempx);
  free(tempy);
  free(tempz);
}


/*
void gradient(int dir)
{
  int i,j,k;
  int m,n,l;
  int max_x,max_y,max_z;
  float max_value, tmp;
  float maxgrad, gradient;
  
  
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	max_value = -99999;
	for (l=max(0,k-WINDOW); l<=min(ZDIM-1,k+WINDOW); l++) 
	  for (n=max(0,j-WINDOW); n<=min(YDIM-1,j+WINDOW); n++) 
	    for (m=max(0,i-WINDOW); m<=min(XDIM-1,i+WINDOW); m++) {
	      if (dataset->data[IndexVect(m,n,l)] > max_value) {
		max_value = dataset->data[IndexVect(m,n,l)];
		max_x = m;
		max_y = n;
		max_z = l;
	      }
	    }
	
	if(max_value <= dataset->data[IndexVect(i,j,k)]) {
	  velocity[IndexVect(i,j,k)].x = 0;
	  velocity[IndexVect(i,j,k)].y = 0;
	  velocity[IndexVect(i,j,k)].z = 0;
	}
	else {
	  tmp = (dataset->data[IndexVect(max_x,max_y,max_z)]-
		 dataset->data[IndexVect(i,j,k)])/
	         sqrt((max_x - i)*(max_x - i)+
		      (max_y - j)*(max_y - j)+
		      (max_z - k)*(max_z - k));
	  velocity[IndexVect(i,j,k)].x = (max_x - i) * tmp;
	  velocity[IndexVect(i,j,k)].y = (max_y - j) * tmp;
	  velocity[IndexVect(i,j,k)].z = (max_z - k) * tmp;
	  
	}
      }

  maxgrad = -999;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	gradient = velocity[IndexVect(i,j,k)].x * velocity[IndexVect(i,j,k)].x +
	           velocity[IndexVect(i,j,k)].y * velocity[IndexVect(i,j,k)].y +
	           velocity[IndexVect(i,j,k)].z * velocity[IndexVect(i,j,k)].z;
	if (gradient > maxgrad)
	  maxgrad = gradient;
      }
  
  maxgrad = sqrt(maxgrad);
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) {
	velocity[IndexVect(i,j,k)].x = velocity[IndexVect(i,j,k)].x / maxgrad;
	velocity[IndexVect(i,j,k)].y = velocity[IndexVect(i,j,k)].y / maxgrad;
	velocity[IndexVect(i,j,k)].z = velocity[IndexVect(i,j,k)].z / maxgrad;
      }
 
}
*/



void GetGradient()
{
  int x,y,z;
  int i,j,k;
  float gradient, grad_x, grad_y, grad_z;
  float maxgrad, mingrad;


  maxgrad = 0;
  mingrad = 999999;
  for (z=0; z<ZDIM; z++)
    for (y=0; y<YDIM; y++) 
      for (x=0; x<XDIM; x++) {
	grad_x=0.0;
	for (j=max(y-1,0); j<=min(y+1,YDIM-1); j++) 
	  for (k=max(z-1,0); k<=min(z+1,ZDIM-1); k++) {
	    grad_x += dataset->data[IndexVect(min(x+1,XDIM-1),j,k)]-
	              dataset->data[IndexVect(max(x-1,0),j,k)];
	    if (j==y || k==z)
	      grad_x += dataset->data[IndexVect(min(x+1,XDIM-1),j,k)]-
		        dataset->data[IndexVect(max(x-1,0),j,k)];
	    if (j==y && k==z)
	      grad_x += 2.0*(dataset->data[IndexVect(min(x+1,XDIM-1),j,k)]-
			     dataset->data[IndexVect(max(x-1,0),j,k)]);
	  }
	
	grad_y=0.0;
	for (i=max(x-1,0); i<=min(x+1,XDIM-1); i++) 
	  for (k=max(z-1,0); k<=min(z+1,ZDIM-1); k++) {
	    grad_y += dataset->data[IndexVect(i,min(y+1,YDIM-1),k)]-
	              dataset->data[IndexVect(i,max(y-1,0),k)];
	    if (i==x || k==z)
	      grad_y += dataset->data[IndexVect(i,min(y+1,YDIM-1),k)]-
		        dataset->data[IndexVect(i,max(y-1,0),k)];
	    if (i==x && k==z)
	      grad_y += 2.0*(dataset->data[IndexVect(i,min(y+1,YDIM-1),k)]-
			     dataset->data[IndexVect(i,max(y-1,0),k)]);
	  }
	
	grad_z=0.0;
	for (i=max(x-1,0); i<=min(x+1,XDIM-1); i++) 
	  for (j=max(y-1,0); j<=min(y+1,YDIM-1); j++) { 
	    grad_z += dataset->data[IndexVect(i,j,min(z+1,ZDIM-1))]-
	              dataset->data[IndexVect(i,j,max(z-1,0))];
	    if (i==x || j==y)
	      grad_z += dataset->data[IndexVect(i,j,min(z+1,ZDIM-1))]-
		        dataset->data[IndexVect(i,j,max(z-1,0))];
	    if (i==x && j==y)
	      grad_z += 2.0*(dataset->data[IndexVect(i,j,min(z+1,ZDIM-1))]-
			     dataset->data[IndexVect(i,j,max(z-1,0))]);
	  }
        

	velocity[IndexVect(x,y,z)].x = grad_x;
	velocity[IndexVect(x,y,z)].y = grad_y;
	velocity[IndexVect(x,y,z)].z = grad_z;

	gradient=sqrt(grad_x*grad_x+grad_y*grad_y+grad_z*grad_z);
	ImgGrad[IndexVect(x,y,z)] = gradient;

	if (gradient < mingrad)
	  mingrad = gradient;
	if (gradient > maxgrad)
	  maxgrad = gradient;
	
      }
  
  if (mingrad < maxgrad) {
    for (z=0; z<ZDIM; z++)
      for (y=0; y<YDIM; y++) 
	for (x=0; x<XDIM; x++) {
	  ImgGrad[IndexVect(x,y,z)] = (ImgGrad[IndexVect(x,y,z)]-mingrad)/
	                              (maxgrad - mingrad);
	  velocity[IndexVect(x,y,z)].x /= maxgrad;
	  velocity[IndexVect(x,y,z)].y /= maxgrad;
	  velocity[IndexVect(x,y,z)].z /= maxgrad;
	}
  }
  
} 

}
